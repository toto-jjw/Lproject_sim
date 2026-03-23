# Lproject_sim/src/core/ros_manager.py
import rclpy
from rclpy.node import Node
import numpy as np
import threading

# ROS 2 Messages
from geometry_msgs.msg import Twist, PoseStamped, Vector3, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Header, Float32, Float64, Empty
from std_srvs.srv import Trigger
#from cv_bridge import CvBridge
import tf2_ros
from tf2_msgs.msg import TFMessage

class ROSManager(Node):
    """
    Native ROS 2 integration for Isaac Sim.
    Running DIRECTLY inside the simulation process (Python 3.11).
    """
    
    def __init__(self, robot_name="husky", publish_map_tf=True):
        # Initialize Node
        super().__init__(f'isaac_ros_bridge_{robot_name}')
        self.robot_name = robot_name # Use passed argument
        self.publish_map_tf = publish_map_tf

        # --- Publishers ---

        # 1. Logic/Status Publishers
        # Sun sensor is custom logic, not in USD
        self.sun_pub = self.create_publisher(Vector3, f'/{self.robot_name}/sun_vector', 10)
        self.battery_pub = self.create_publisher(BatteryState, f'/{self.robot_name}/battery_state', 10)
        
        # 센서 온도 발행 (camera_noise_node에서 구독)
        self.temperature_pub = self.create_publisher(Float64, '/rover/sensor_temperature', 10)
        
        # 2. GT TF Publisher (separated topic to avoid conflict with Nav2/SLAM)
        if self.publish_map_tf:
            self.tf_gt_pub = self.create_publisher(TFMessage, '/tf_gt', 10)
            print(f"[ROSManager] map→base_link GT TF broadcasting enabled on /tf_gt")
        else:
            self.tf_gt_pub = None
        
        # Note: cmd_vel is handled by USD OmniGraph subscriber
        # Robot control: ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/{robot_name}/cmd_vel
        
        
        #self.cv_bridge = CvBridge()        
       
        
        #self.create_subscription(Twist, f'/{self.robot_name}/cmd_vel', self.cmd_vel_cb, 10)
        self.create_subscription(Float32, f'/{self.robot_name}/solar_panel/cmd_angle', self.solar_cb, 10)
        #self.create_subscription(PoseStamped, f'/{self.robot_name}/mission/goal', self.goal_cb, 10)
        
        # Reset Pose Service
        self._reset_pose_callback = None
        self.reset_service = self.create_service(
            Trigger, f'/{self.robot_name}/reset_pose', self._reset_pose_srv_cb
        )
        # Reset Pose Topic (간편한 호출용)
        self.create_subscription(Empty, f'/{self.robot_name}/reset_pose_cmd', self._reset_pose_topic_cb, 10)
        # Custom Pose Reset Topic (지정 위치로 리셋)
        self.create_subscription(PoseStamped, f'/{self.robot_name}/reset_pose_target', self._reset_pose_target_cb, 10)
        # Custom Reset Target 설정 (지속적인 리셋 목표 위치 변경)
        self.create_subscription(PoseStamped, f'/{self.robot_name}/set_reset_target', self._set_reset_target_cb, 10)
        self._reset_requested = False
        self._reset_target_pose = None  # None이면 초기 위치, (position, orientation)이면 지정 위치
        self._set_target_callback = None  # robot_context.set_custom_reset_target 콜백
        
        # State
        #self.current_cmd_vel = (0.0, 0.0)
        self.current_solar_cmd = 0.0
        #self.current_goal_pose = None
        
        #print(f"[ROSManager] Initialized for {self.robot_name} (Mode: {self.bridge_type})")
        print(f"[ROSManager] Initialized for {self.robot_name}")


    def set_reset_pose_callback(self, callback):
        """로버 리셋 콜백 함수를 등록합니다."""
        self._reset_pose_callback = callback
    
    def set_target_callback(self, callback):
        """커스텀 리셋 타겟 설정 콜백을 등록합니다. (robot_context.set_custom_reset_target)"""
        self._set_target_callback = callback
    
    def _reset_pose_srv_cb(self, request, response):
        """ROS2 Service 콜백: 로버를 초기 위치로 리셋"""
        self._reset_requested = True
        self._reset_target_pose = None  # 초기 위치로
        response.success = True
        response.message = "Reset pose requested. Will be applied on next simulation step."
        return response
    
    def _reset_pose_topic_cb(self, msg):
        """ROS2 Topic 콜백: 로버를 초기 위치로 리셋"""
        self._reset_requested = True
        self._reset_target_pose = None  # 초기 위치로
    
    def _reset_pose_target_cb(self, msg: PoseStamped):
        """
        ROS2 Topic 콜백: 로버를 지정된 위치로 리셋
        PoseStamped 메시지의 pose를 사용합니다.
        
        사용법:
          ros2 topic pub --once /{robot_name}/reset_pose_target geometry_msgs/msg/PoseStamped \
            "{header: {frame_id: 'world'}, pose: {position: {x: 1.0, y: 2.0, z: 0.5}, orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}}}"
        """
        p = msg.pose.position
        q = msg.pose.orientation
        position = np.array([p.x, p.y, p.z], dtype=np.float64)
        # ROS quaternion (x,y,z,w) → Isaac Sim quaternion (w,x,y,z)
        orientation = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
        
        self._reset_requested = True
        self._reset_target_pose = (position, orientation)
        self.get_logger().info(
            f"Reset to custom pose requested: pos=[{p.x:.2f}, {p.y:.2f}, {p.z:.2f}]"
        )
    
    def _set_reset_target_cb(self, msg: PoseStamped):
        """
        ROS2 Topic 콜백: 커스텀 리셋 목표 위치를 설정 (즉시 리셋하지 않음)
        이후 reset_pose_cmd / R키 / Trigger 서비스 호출 시 이 위치로 리셋됩니다.
        
        사용법:
          ros2 topic pub --once /{robot_name}/set_reset_target geometry_msgs/msg/PoseStamped \
            "{header: {frame_id: 'world'}, pose: {position: {x: 5.0, y: 3.0, z: 0.5}, orientation: {w: 1.0}}}"
        """
        p = msg.pose.position
        q = msg.pose.orientation
        position = np.array([p.x, p.y, p.z], dtype=np.float64)
        orientation = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
        
        if self._set_target_callback:
            self._set_target_callback(position, orientation)
        
        self.get_logger().info(
            f"Custom reset target set: pos=[{p.x:.2f}, {p.y:.2f}, {p.z:.2f}]"
        )
    
    def is_reset_requested(self):
        """
        리셋 요청이 있는지 확인하고 플래그를 초기화합니다.
        
        Returns:
            tuple: (requested: bool, target_pose: tuple or None)
                   target_pose가 None이면 초기 위치로 리셋,
                   (position, orientation)이면 해당 위치로 리셋
        """
        if self._reset_requested:
            self._reset_requested = False
            target = self._reset_target_pose
            self._reset_target_pose = None
            return True, target
        return False, None
    
    def solar_cb(self, msg):
        self.current_solar_cmd = msg.data


    def get_solar_cmd(self):
        return self.current_solar_cmd


    def publish_sun_vector(self, vector):
        msg = Vector3()
        msg.x = float(vector[0])
        msg.y = float(vector[1])
        msg.z = float(vector[2])
        self.sun_pub.publish(msg)


    def publish_battery_state(self, voltage, percentage, current):
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.voltage = float(voltage)
        msg.percentage = float(percentage)
        msg.current = float(current)
        self.battery_pub.publish(msg)


    def publish_sensor_temperature(self, temperature_c: float):
        """
        센서 온도를 발행합니다.
        camera_noise_node에서 Dark Current 계산에 사용됩니다.
        
        Args:
            temperature_c: 온도 (섭씨)
        """
        msg = Float64()
        # Kelvin으로 변환하여 발행 (camera_noise_node에서 자동 감지)
        msg.data = float(temperature_c + 273.15)  # °C → K
        self.temperature_pub.publish(msg)


    def publish_map_to_base_tf(self, position, orientation):
        """
        map → base_link GT TF를 /tf_gt 토픽으로 발행합니다.
        시뮬레이션 월드 좌표를 절대 좌표(map 프레임)로 발행합니다.
        
        Args:
            position: [x, y, z] 월드 좌표
            orientation: [w, x, y, z] 쿼터니언 (Isaac Core format)
        """
        if not self.publish_map_tf or self.tf_gt_pub is None:
            return
            
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        
        # Isaac Core (w, x, y, z) -> ROS (x, y, z, w)
        t.transform.rotation.w = float(orientation[0])
        t.transform.rotation.x = float(orientation[1])
        t.transform.rotation.y = float(orientation[2])
        t.transform.rotation.z = float(orientation[3])
        
        # Manually create TFMessage and publish to /tf_gt
        tf_msg = TFMessage()
        tf_msg.transforms = [t]
        self.tf_gt_pub.publish(tf_msg)


    def spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.0)

    def shutdown(self):
        self.destroy_node()

