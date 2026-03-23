#!/usr/bin/env python3
# Lproject_sim/src/nodes/slope_costmap_node.py
"""
Slope-based Costmap Publisher Node

이 노드는 지형의 기울기(slope)를 계산하여 OccupancyGrid로 발행합니다.
크레이터와 같은 오목한 지형을 장애물로 인식하게 해줍니다.

작동 방식:
1. NVBlox의 depth map 또는 pointcloud를 구독
2. 로봇 주변 영역의 고도(elevation) 맵 생성
3. 인접 셀 간의 기울기 계산
4. 임계값 이상의 기울기를 가진 영역을 장애물로 표시
5. OccupancyGrid로 발행 → Nav2 StaticLayer에서 사용
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
import numpy as np
import tf2_ros
from tf2_ros import Buffer, TransformListener
import struct
import math


class SlopeCostmapNode(Node):
    def __init__(self):
        super().__init__('slope_costmap_node')
        
        # Parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('map_size', 20.0)  # meters
        self.declare_parameter('resolution', 0.1)  # meters per cell
        self.declare_parameter('max_slope_degrees', 25.0)  # degrees - 이 이상이면 장애물
        self.declare_parameter('slope_cost_scale', 1.0)  # 기울기에 대한 비용 스케일
        self.declare_parameter('update_rate', 2.0)  # Hz
        self.declare_parameter('pointcloud_topic', '/nvblox_node/static_pointcloud')
        self.declare_parameter('unknown_cost', -1)  # 알 수 없는 영역의 비용 (-1 = unknown)
        self.declare_parameter('min_points_per_cell', 1)  # 셀당 최소 포인트 수
        
        # Get parameters
        self.map_frame = self.get_parameter('map_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.map_size = self.get_parameter('map_size').value
        self.resolution = self.get_parameter('resolution').value
        self.max_slope_deg = self.get_parameter('max_slope_degrees').value
        self.slope_cost_scale = self.get_parameter('slope_cost_scale').value
        self.update_rate = self.get_parameter('update_rate').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.unknown_cost = self.get_parameter('unknown_cost').value
        self.min_points_per_cell = self.get_parameter('min_points_per_cell').value
        
        # 최대 기울기를 라디안으로 변환
        self.max_slope_rad = math.radians(self.max_slope_deg)
        
        # Grid dimensions
        self.grid_size = int(self.map_size / self.resolution)
        
        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Elevation map (stores height values)
        self.elevation_map = np.full((self.grid_size, self.grid_size), np.nan, dtype=np.float32)
        self.point_count_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Robot position
        self.robot_x = 0.0
        self.robot_y = 0.0
        
        # QoS for pointcloud
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        costmap_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/slope_costmap', costmap_qos)
        
        # Subscribers
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            qos
        )
        
        # Timer for publishing costmap
        self.timer = self.create_timer(1.0 / self.update_rate, self.publish_costmap)
        
        self.last_pointcloud_time = None
        
        self.get_logger().info(f'Slope Costmap Node Started')
        self.get_logger().info(f'  Map size: {self.map_size}m x {self.map_size}m')
        self.get_logger().info(f'  Resolution: {self.resolution}m')
        self.get_logger().info(f'  Max slope: {self.max_slope_deg}°')
        self.get_logger().info(f'  Subscribing to: {self.pointcloud_topic}')
    
    def get_robot_position(self):
        """로봇의 현재 위치를 TF에서 가져옴"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            self.robot_x = transform.transform.translation.x
            self.robot_y = transform.transform.translation.y
            return True
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return False
    
    def pointcloud_callback(self, msg: PointCloud2):
        """PointCloud2 메시지에서 elevation map 업데이트"""
        if not self.get_robot_position():
            return
        
        # Reset maps
        self.elevation_map.fill(np.nan)
        self.point_count_map.fill(0)
        
        # Parse PointCloud2
        points = self.parse_pointcloud2(msg)
        if points is None or len(points) == 0:
            return
        
        # Map origin (bottom-left corner)
        origin_x = self.robot_x - self.map_size / 2.0
        origin_y = self.robot_y - self.map_size / 2.0
        
        # Populate elevation map
        for point in points:
            px, py, pz = point[:3]
            
            # Convert to grid coordinates
            gx = int((px - origin_x) / self.resolution)
            gy = int((py - origin_y) / self.resolution)
            
            # Check bounds
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                # 평균 높이 계산을 위해 누적
                if np.isnan(self.elevation_map[gy, gx]):
                    self.elevation_map[gy, gx] = pz
                    self.point_count_map[gy, gx] = 1
                else:
                    # Incremental mean
                    n = self.point_count_map[gy, gx]
                    self.elevation_map[gy, gx] = (self.elevation_map[gy, gx] * n + pz) / (n + 1)
                    self.point_count_map[gy, gx] = n + 1
        
        self.last_pointcloud_time = self.get_clock().now()
    
    def parse_pointcloud2(self, msg: PointCloud2):
        """PointCloud2 메시지를 numpy 배열로 변환"""
        try:
            # Find field offsets
            x_offset = y_offset = z_offset = None
            for field in msg.fields:
                if field.name == 'x':
                    x_offset = field.offset
                elif field.name == 'y':
                    y_offset = field.offset
                elif field.name == 'z':
                    z_offset = field.offset
            
            if x_offset is None or y_offset is None or z_offset is None:
                self.get_logger().warn('PointCloud2 missing xyz fields')
                return None
            
            # Parse points
            points = []
            point_step = msg.point_step
            data = msg.data
            
            for i in range(msg.width * msg.height):
                offset = i * point_step
                x = struct.unpack_from('f', data, offset + x_offset)[0]
                y = struct.unpack_from('f', data, offset + y_offset)[0]
                z = struct.unpack_from('f', data, offset + z_offset)[0]
                
                # Filter NaN and Inf
                if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                    points.append([x, y, z])
            
            return np.array(points, dtype=np.float32) if points else None
            
        except Exception as e:
            self.get_logger().error(f'Failed to parse PointCloud2: {e}')
            return None
    
    def compute_slope_costmap(self):
        """Elevation map에서 기울기 기반 costmap 계산"""
        costmap = np.full((self.grid_size, self.grid_size), self.unknown_cost, dtype=np.int8)
        
        # 유효한 elevation 데이터가 있는 셀만 처리
        valid_mask = (self.point_count_map >= self.min_points_per_cell) & ~np.isnan(self.elevation_map)
        
        if not np.any(valid_mask):
            return costmap
        
        # Sobel-like gradient calculation
        for y in range(1, self.grid_size - 1):
            for x in range(1, self.grid_size - 1):
                if not valid_mask[y, x]:
                    continue
                
                # 3x3 이웃 확인
                neighbors_valid = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if not valid_mask[y + dy, x + dx]:
                            neighbors_valid = False
                            break
                    if not neighbors_valid:
                        break
                
                if not neighbors_valid:
                    # 일부 이웃만 있어도 계산 시도
                    costmap[y, x] = 0  # Free space로 가정
                    continue
                
                # X 방향 기울기 (dz/dx)
                dz_dx = (self.elevation_map[y, x+1] - self.elevation_map[y, x-1]) / (2 * self.resolution)
                
                # Y 방향 기울기 (dz/dy)
                dz_dy = (self.elevation_map[y+1, x] - self.elevation_map[y-1, x]) / (2 * self.resolution)
                
                # 전체 기울기 (magnitude)
                slope = math.sqrt(dz_dx**2 + dz_dy**2)
                slope_angle = math.atan(slope)  # radians
                
                # 기울기를 비용으로 변환
                if slope_angle >= self.max_slope_rad:
                    # 최대 기울기 초과 → 장애물 (100)
                    costmap[y, x] = 100
                else:
                    # 기울기에 비례하는 비용 (0-99)
                    normalized_slope = slope_angle / self.max_slope_rad
                    cost = int(normalized_slope * 99 * self.slope_cost_scale)
                    costmap[y, x] = min(cost, 99)
        
        return costmap
    
    def publish_costmap(self):
        """OccupancyGrid로 costmap 발행"""
        if self.last_pointcloud_time is None:
            return
        
        # Check if data is stale (more than 5 seconds old)
        age = (self.get_clock().now() - self.last_pointcloud_time).nanoseconds / 1e9
        if age > 5.0:
            self.get_logger().debug('PointCloud data is stale, skipping publish')
            return
        
        if not self.get_robot_position():
            return
        
        # Compute slope costmap
        costmap_data = self.compute_slope_costmap()
        
        # Create OccupancyGrid message
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = self.robot_x - self.map_size / 2.0
        msg.info.origin.position.y = self.robot_y - self.map_size / 2.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Flatten and convert to list
        msg.data = costmap_data.flatten().tolist()
        
        self.costmap_pub.publish(msg)
        self.get_logger().debug(f'Published slope costmap ({self.grid_size}x{self.grid_size})')


def main(args=None):
    rclpy.init(args=args)
    node = SlopeCostmapNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
