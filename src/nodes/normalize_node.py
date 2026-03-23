#!/usr/bin/env python3
"""
Normalize Node (Simple Brightness Enhancement)
단순 밝기 향상 (Gamma Correction / Linear Scaling)으로 DimCam과 비교용

Subscribed Topics:
    /stereo/left/rgb_noisy/compressed (sensor_msgs/CompressedImage): Left camera RGB image
    /stereo/right/rgb_noisy/compressed (sensor_msgs/CompressedImage): Right camera RGB image

Published Topics:
    /stereo/left/brightened/compressed (sensor_msgs/CompressedImage): Brightened left image
    /stereo/right/brightened/compressed (sensor_msgs/CompressedImage): Brightened right image
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import numpy as np
import cv2
import time


class SimpleBrightnessNode(Node):
    """Simple Brightness Enhancement ROS2 Node"""
    
    def __init__(self):
        super().__init__('simple_brightness_node')
        
        # --- 파라미터 선언 ---
        self.declare_parameter('method', 'normalize')  # 'normalize', 'gamma', 'linear', 'clahe'
        self.declare_parameter('gamma', 0.4)  # gamma < 1 = 밝아짐 (0.4 권장)
        self.declare_parameter('norm_percentile', 1.0)  # 정규화시 상/하위 percentile 제외 (outlier 제거)
        self.declare_parameter('norm_gamma', 0.5)  # 정규화 후 적용할 gamma (1.0 = 비활성화)
        self.declare_parameter('linear_scale', 3.0)  # 밝기 배수
        self.declare_parameter('clahe_clip', 2.0)  # CLAHE clip limit
        self.declare_parameter('clahe_grid', 8)  # CLAHE grid size
        self.declare_parameter('max_rate', 30.0)  # 목표 처리 Hz
        
        # 파라미터 로드
        self.method = self.get_parameter('method').value
        self.gamma = self.get_parameter('gamma').value
        self.norm_percentile = self.get_parameter('norm_percentile').value
        self.norm_gamma = self.get_parameter('norm_gamma').value
        self.linear_scale = self.get_parameter('linear_scale').value
        self.clahe_clip = self.get_parameter('clahe_clip').value
        self.clahe_grid = self.get_parameter('clahe_grid').value
        self.max_rate = self.get_parameter('max_rate').value
        
        # --- CV Bridge ---
        self.bridge = CvBridge()
        
        # --- 처리 상태 ---
        self.process_count = 0
        self.start_time = time.time()
        self.last_process_time = 0.0
        self.is_processing = False
        
        # --- Gamma LUT (속도 최적화) ---
        self.gamma_lut = self._create_gamma_lut(self.gamma)
        self.norm_gamma_lut = self._create_gamma_lut(self.norm_gamma)  # 정규화용 gamma LUT
        
        # --- CLAHE 객체 ---
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip, 
            tileGridSize=(self.clahe_grid, self.clahe_grid)
        )
        
        # --- QoS 설정 ---
        # Subscriber QoS: BEST_EFFORT (속도 향상을 위해 가벼운 토픽 구독)
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5  # BEST_EFFORT는 작은 버퍼로 충분
        )
        
        # --- Synchronized Subscribers (message_filters) ---
        self.left_sub = Subscriber(self, CompressedImage, '/stereo/left/rgb_noisy/compressed', qos_profile=sub_qos)
        self.right_sub = Subscriber(self, CompressedImage, '/stereo/right/rgb_noisy/compressed', qos_profile=sub_qos)
        
        # ApproximateTimeSynchronizer: 타임스탬프가 비슷한 Left/Right 이미지를 매칭
        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=0.1  # 100ms 이내의 타임스탬프 차이 허용
        )
        self.sync.registerCallback(self.synced_callback)
        
        # --- Publishers ---
        # Compressed: BEST_EFFORT QoS (효율적 전송용)
        compressed_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Raw Image: RELIABLE QoS (RViz 호환용)
        raw_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        # Compressed Image publishers (for efficient network transfer)
        self.left_pub = self.create_publisher(CompressedImage, '/stereo/left/brightened/compressed', compressed_qos)
        self.right_pub = self.create_publisher(CompressedImage, '/stereo/right/brightened/compressed', compressed_qos)
        # Raw Image publishers (for RViz - RELIABLE)
        self.left_raw_pub = self.create_publisher(Image, '/stereo/left/brightened', raw_qos)
        self.right_raw_pub = self.create_publisher(Image, '/stereo/right/brightened', raw_qos)
        
        self.get_logger().info('Simple Brightness Node initialized')
        self.get_logger().info(f'  - Method: {self.method}')
        if self.method == 'normalize':
            self.get_logger().info(f'  - Percentile: {self.norm_percentile}% (outlier removal)')
            self.get_logger().info(f'  - Post-Gamma: {self.norm_gamma} (< 1 = brighter)')
        elif self.method == 'gamma':
            self.get_logger().info(f'  - Gamma: {self.gamma} (< 1 = brighter)')
        elif self.method == 'linear':
            self.get_logger().info(f'  - Scale: {self.linear_scale}x')
        elif self.method == 'clahe':
            self.get_logger().info(f'  - CLAHE clip: {self.clahe_clip}, grid: {self.clahe_grid}')
        self.get_logger().info(f'  - Target rate: {self.max_rate} Hz')
    
    def _create_gamma_lut(self, gamma: float) -> np.ndarray:
        """Gamma correction용 LUT 생성 (속도 최적화)"""
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 
                        for i in range(256)]).astype(np.uint8)
        return lut
    
    def _apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """Gamma correction 적용 (LUT 사용으로 빠름)"""
        return cv2.LUT(image, self.gamma_lut)
    
    def _apply_normalize(self, image: np.ndarray) -> np.ndarray:
        """Min-Max 정규화 + Gamma: 픽셀값을 0~255 범위로 스트레칭 후 gamma 적용
        
        공식: 
        1. normalized = (input - min) / (max - min) * 255
        2. output = normalized ^ (1/gamma)
        
        percentile 옵션으로 극단값(노이즈) 제외 가능
        """
        # Grayscale로 변환하여 밝기 기준 계산
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Percentile 기반 min/max (outlier 제거)
        if self.norm_percentile > 0:
            min_val = np.percentile(gray, self.norm_percentile)
            max_val = np.percentile(gray, 100 - self.norm_percentile)
        else:
            min_val = gray.min()
            max_val = gray.max()
        
        # 분모가 0이 되는 것 방지
        if max_val - min_val < 1:
            max_val = min_val + 1
        
        # 1. 정규화 적용 (채널별로)
        result = image.astype(np.float32)
        result = (result - min_val) / (max_val - min_val) * 255.0
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # 2. Gamma 적용 (norm_gamma != 1.0 일 때만)
        if abs(self.norm_gamma - 1.0) > 0.01:
            result = cv2.LUT(result, self.norm_gamma_lut)
        
        return result
    
    def _apply_linear(self, image: np.ndarray) -> np.ndarray:
        """Linear scaling 적용"""
        result = image.astype(np.float32) * self.linear_scale
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용"""
        # LAB 색공간에서 L 채널에만 적용
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """선택된 방법으로 이미지 밝기 향상"""
        if self.method == 'normalize':
            return self._apply_normalize(image)
        elif self.method == 'gamma':
            return self._apply_gamma(image)
        elif self.method == 'linear':
            return self._apply_linear(image)
        elif self.method == 'clahe':
            return self._apply_clahe(image)
        else:
            self.get_logger().warn(f'Unknown method: {self.method}, using normalize')
            return self._apply_normalize(image)
    
    def _decode_compressed(self, msg: CompressedImage) -> np.ndarray:
        """CompressedImage를 numpy array로 디코딩"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    def _encode_compressed(self, image: np.ndarray, header) -> CompressedImage:
        """numpy array를 CompressedImage로 인코딩"""
        # RGB -> BGR for cv2.imencode
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # JPEG 압축 (quality=90)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, compressed_data = cv2.imencode('.jpg', image_bgr, encode_param)
        
        msg = CompressedImage()
        msg.header = header
        msg.format = 'jpeg'
        msg.data = compressed_data.tobytes()
        return msg
    
    def synced_callback(self, left_msg: CompressedImage, right_msg: CompressedImage):
        """동기화된 Left/Right 이미지 처리 콜백"""
        # Rate limiting
        current_time = time.time()
        min_interval = 1.0 / self.max_rate
        if (current_time - self.last_process_time) < min_interval:
            return
        
        # 이미 처리 중이면 스킵
        if self.is_processing:
            return
        
        self.is_processing = True
        self.last_process_time = current_time
        
        try:
            start_time = time.time()
            
            # Compressed 이미지 디코딩
            left_image = self._decode_compressed(left_msg)
            right_image = self._decode_compressed(right_msg)
            
            # 밝기 향상
            left_enhanced = self._enhance_image(left_image)
            right_enhanced = self._enhance_image(right_image)
            
            # ROS CompressedImage로 변환 및 퍼블리시
            left_out_msg = self._encode_compressed(left_enhanced, left_msg.header)
            right_out_msg = self._encode_compressed(right_enhanced, right_msg.header)
            
            # Compressed 발행
            self.left_pub.publish(left_out_msg)
            self.right_pub.publish(right_out_msg)
            
            # Raw Image 발행 (RViz용)
            left_raw_msg = self.bridge.cv2_to_imgmsg(left_enhanced, encoding='rgb8')
            left_raw_msg.header = left_msg.header
            right_raw_msg = self.bridge.cv2_to_imgmsg(right_enhanced, encoding='rgb8')
            right_raw_msg.header = right_msg.header
            self.left_raw_pub.publish(left_raw_msg)
            self.right_raw_pub.publish(right_raw_msg)
            
            self.process_count += 1
            
            # 성능 로깅 (5초마다)
            process_time = time.time() - start_time
            elapsed = current_time - self.start_time
            if elapsed > 5.0 and self.process_count > 0:
                avg_fps = self.process_count / elapsed
                self.get_logger().info(
                    f'Process time: {process_time*1000:.1f}ms, '
                    f'Output: {avg_fps:.1f} FPS'
                )
                self.start_time = current_time
                self.process_count = 0
                
        except Exception as e:
            self.get_logger().error(f'Error processing images: {e}')
        finally:
            self.is_processing = False


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SimpleBrightnessNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
