
#!/usr/bin/env python3
# Lproject_sim/src/nodes/solar_control_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32
import math
import numpy as np
import time

class SolarControlNode(Node):
    def __init__(self):
        super().__init__('solar_control_node')
        
        self.sun_sub = self.create_subscription(Vector3, '/husky_1/sun_vector', self.sun_callback, 10)
        self.cmd_pub = self.create_publisher(Float32, '/husky_1/solar_panel/cmd_angle', 10)
        
        self.current_sun_vector = None
        self.current_cmd_angle = 0.0  # 현재 패널 각도 (점진적 회전용)
        self.target_cmd_angle = 0.0   # 목표 각도
        
        # 시작 시간 기록 (2초 대기용)
        self.start_time = time.time()
        self.startup_delay = 2.0  # 2초 대기
        self.is_initialized = False
        
        # 회전 속도 (rad/s) - 부드러운 추적을 위해 느리게
        self.rotation_speed = 0.3  # 약 17도/초
        
        self.timer = self.create_timer(0.1, self.control_loop) # 10Hz
        
        self.get_logger().info("Solar Control Node Started - Waiting 2 seconds before tracking...")

    def sun_callback(self, msg):
        self.current_sun_vector = msg

    def control_loop(self):
        # 2초 대기 체크
        elapsed = time.time() - self.start_time
        if elapsed < self.startup_delay:
            return
        
        if not self.is_initialized:
            self.is_initialized = True
            self.get_logger().info("Starting solar panel tracking...")
        
        if self.current_sun_vector is None:
            return

        # Sun vector in Sensor Frame (assumed aligned with Robot Base Link)
        s_x = self.current_sun_vector.x
        s_y = self.current_sun_vector.y
        
        # Calculate angle to sun in Robot Frame
        sun_angle = math.atan2(s_y, s_x)
        
        # Panel Default (0 deg command) faces Backwards (180 deg relative to Front)
        # Panel Normal Angle = cmd + pi
        # We want Panel Normal Angle = sun_angle
        # cmd = sun_angle - pi
        
        self.target_cmd_angle = sun_angle - math.pi
        
        # Normalize to +/- pi
        while self.target_cmd_angle > math.pi:
            self.target_cmd_angle -= 2 * math.pi
        while self.target_cmd_angle < -math.pi:
            self.target_cmd_angle += 2 * math.pi
            
        # 점진적 회전: 현재 각도에서 목표 각도로 서서히 이동
        angle_diff = self.target_cmd_angle - self.current_cmd_angle
        
        # 각도 차이를 -pi ~ pi 범위로 정규화 (최단 경로 회전)
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # 회전 속도 제한 적용
        dt = 0.1  # 타이머 주기
        max_rotation = self.rotation_speed * dt
        
        if abs(angle_diff) < max_rotation:
            # 목표에 거의 도달
            self.current_cmd_angle = self.target_cmd_angle
        else:
            # 점진적 회전
            self.current_cmd_angle += max_rotation * (1.0 if angle_diff > 0 else -1.0)
        
        # 정규화
        while self.current_cmd_angle > math.pi:
            self.current_cmd_angle -= 2 * math.pi
        while self.current_cmd_angle < -math.pi:
            self.current_cmd_angle += 2 * math.pi
            
        # Publish
        msg = Float32()
        msg.data = float(self.current_cmd_angle)
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SolarControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
