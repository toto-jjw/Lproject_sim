#!/usr/bin/env python3
"""
Adjust Node (Stereo Image Synchronizer & FPS Throttler)

스테레오 좌/우 이미지를 동기화하고, 지정된 FPS로 출력을 조절하는 노드입니다.
노이즈/향상 등 업스트림 노드의 출력 FPS가 불안정하거나 너무 높을 때,
다운스트림 (SLAM, Navigation 등)에 안정적인 프레임 속도를 제공합니다.

기능:
  - ApproximateTimeSynchronizer로 L/R 스테레오 페어 매칭
  - 지정된 target FPS로 출력 조절 (기본 5 Hz)
  - 통일된 타임스탬프로 L/R 동시 발행 (noise_node/enhance_node와 동일 패턴)
  - Raw + Compressed 이미지 모두 지원

Input Topics (configurable):
  /stereo/left/enhanced  (sensor_msgs/Image)
  /stereo/right/enhanced (sensor_msgs/Image)

Output Topics:
  /stereo/left/adjusted              (sensor_msgs/Image)
  /stereo/right/adjusted             (sensor_msgs/Image)
  /stereo/left/adjusted/compressed   (sensor_msgs/CompressedImage)
  /stereo/right/adjusted/compressed  (sensor_msgs/CompressedImage)

Usage:
  python3 -m src.nodes.adjust_node --ros-args -p target_fps:=5.0
  
  또는:
  bash run_adjust.sh --fps 5 --input enhanced
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import numpy as np
import cv2
import time


# --- QoS Profiles (noise_node과 동일한 패턴) ---
# Subscriber: input에 따라 RELIABLE 또는 BEST_EFFORT
RAW_SUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE
)

BEST_EFFORT_SUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE
)

# Publisher (Raw Image): RELIABLE
RAW_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE
)

# Publisher (Compressed Image): BEST_EFFORT
COMPRESSED_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE
)


class StereoAdjustNode(Node):
    """
    스테레오 이미지 동기화 + FPS 조절 노드
    
    업스트림 노드(noise/enhance)의 불안정한 출력을 받아서
    안정적인 동기화된 스테레오 페어를 일정 FPS로 발행합니다.
    """
    
    def __init__(self):
        super().__init__('stereo_adjust_node')
        
        # --- 파라미터 선언 ---
        self.declare_parameter('target_fps', 5.0)
        self.declare_parameter('input_type', 'enhanced')  # 'raw', 'noisy', 'enhanced'
        self.declare_parameter('left_topic', '')  # 비어있으면 input_type에 따라 자동 결정
        self.declare_parameter('right_topic', '')
        self.declare_parameter('output_prefix', '/stereo')
        self.declare_parameter('output_suffix', 'adjusted')
        self.declare_parameter('sync_slop', 0.05)  # 타임스탬프 매칭 허용 오차 (초)
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('publish_raw', True)
        
        # 파라미터 로드
        self.target_fps = self.get_parameter('target_fps').value
        self.input_type = self.get_parameter('input_type').value
        self.left_topic_param = self.get_parameter('left_topic').value
        self.right_topic_param = self.get_parameter('right_topic').value
        self.output_prefix = self.get_parameter('output_prefix').value
        self.output_suffix = self.get_parameter('output_suffix').value
        self.sync_slop = self.get_parameter('sync_slop').value
        self.sync_queue_size = self.get_parameter('sync_queue_size').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.publish_compressed = self.get_parameter('publish_compressed').value
        self.publish_raw = self.get_parameter('publish_raw').value
        
        # --- CV Bridge ---
        self.bridge = CvBridge()
        
        # --- 처리 상태 ---
        self.last_publish_time = 0.0
        self.min_interval = 1.0 / self.target_fps
        self.is_processing = False
        
        # --- 성능 통계 ---
        self.input_count = 0
        self.output_count = 0
        self.dropped_count = 0
        self.start_time = time.time()
        self.last_log_time = self.get_clock().now()
        
        # --- 토픽 결정 ---
        left_topic, right_topic, msg_type, sub_qos = self._resolve_topics()
        
        # --- Publishers ---
        left_out = f'{self.output_prefix}/left/{self.output_suffix}'
        right_out = f'{self.output_prefix}/right/{self.output_suffix}'
        
        if self.publish_raw:
            self.left_raw_pub = self.create_publisher(Image, left_out, RAW_PUB_QOS)
            self.right_raw_pub = self.create_publisher(Image, right_out, RAW_PUB_QOS)
        
        if self.publish_compressed:
            self.left_compressed_pub = self.create_publisher(
                CompressedImage, f'{left_out}/compressed', COMPRESSED_PUB_QOS)
            self.right_compressed_pub = self.create_publisher(
                CompressedImage, f'{right_out}/compressed', COMPRESSED_PUB_QOS)
        
        # --- Synchronized Subscribers ---
        self.left_sub = Subscriber(self, msg_type, left_topic, qos_profile=sub_qos)
        self.right_sub = Subscriber(self, msg_type, right_topic, qos_profile=sub_qos)
        
        if msg_type == Image:
            callback = self._synced_callback_raw
        else:
            callback = self._synced_callback_compressed
        
        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop
        )
        self.sync.registerCallback(callback)
        
        # --- 로깅 ---
        self.get_logger().info('=== Stereo Adjust Node ===')
        self.get_logger().info(f'  Input type: {self.input_type}')
        self.get_logger().info(f'  Left input:  {left_topic}')
        self.get_logger().info(f'  Right input: {right_topic}')
        self.get_logger().info(f'  Left output:  {left_out} (+ /compressed)')
        self.get_logger().info(f'  Right output: {right_out} (+ /compressed)')
        self.get_logger().info(f'  Target FPS: {self.target_fps} Hz (interval: {self.min_interval*1000:.1f}ms)')
        self.get_logger().info(f'  Sync slop: {self.sync_slop*1000:.1f}ms')
        self.get_logger().info(f'  Publish raw: {self.publish_raw}, compressed: {self.publish_compressed}')
    
    def _resolve_topics(self):
        """input_type에 따른 토픽, 메시지 타입, QoS 결정"""
        
        # 사용자가 직접 지정한 경우
        if self.left_topic_param and self.right_topic_param:
            left = self.left_topic_param
            right = self.right_topic_param
            # compressed 토픽이면 CompressedImage, 아니면 Image
            if '/compressed' in left:
                return left, right, CompressedImage, BEST_EFFORT_SUB_QOS
            else:
                return left, right, Image, RAW_SUB_QOS
        
        # input_type에 따라 자동 결정
        topic_map = {
            'raw': {
                'left': '/stereo/left/rgb',
                'right': '/stereo/right/rgb',
                'msg_type': Image,
                'qos': RAW_SUB_QOS,
            },
            'noisy': {
                'left': '/stereo/left/rgb_noisy',
                'right': '/stereo/right/rgb_noisy',
                'msg_type': Image,
                'qos': RAW_SUB_QOS,
            },
            'noisy_compressed': {
                'left': '/stereo/left/rgb_noisy/compressed',
                'right': '/stereo/right/rgb_noisy/compressed',
                'msg_type': CompressedImage,
                'qos': BEST_EFFORT_SUB_QOS,
            },
            'enhanced': {
                'left': '/stereo/left/enhanced',
                'right': '/stereo/right/enhanced',
                'msg_type': Image,
                'qos': RAW_SUB_QOS,
            },
            'enhanced_compressed': {
                'left': '/stereo/left/enhanced/compressed',
                'right': '/stereo/right/enhanced/compressed',
                'msg_type': CompressedImage,
                'qos': BEST_EFFORT_SUB_QOS,
            },
        }
        
        cfg = topic_map.get(self.input_type)
        if cfg is None:
            self.get_logger().warn(
                f'Unknown input_type "{self.input_type}", falling back to "enhanced"')
            cfg = topic_map['enhanced']
        
        return cfg['left'], cfg['right'], cfg['msg_type'], cfg['qos']
    
    # ==================== Callbacks ====================
    
    def _synced_callback_raw(self, left_msg: Image, right_msg: Image):
        """동기화된 Raw 스테레오 이미지 콜백"""
        self.input_count += 1
        
        # Rate limiting
        current_time = time.time()
        if self.is_processing or (current_time - self.last_publish_time) < self.min_interval:
            self.dropped_count += 1
            return
        
        self.is_processing = True
        self.last_publish_time = current_time
        
        try:
            # 통일 타임스탬프 (noise_node과 동일 패턴)
            sync_stamp = self.get_clock().now().to_msg()
            
            # Raw 발행
            if self.publish_raw:
                left_out = Image()
                left_out.header.stamp = sync_stamp
                left_out.header.frame_id = left_msg.header.frame_id
                left_out.height = left_msg.height
                left_out.width = left_msg.width
                left_out.encoding = left_msg.encoding
                left_out.is_bigendian = left_msg.is_bigendian
                left_out.step = left_msg.step
                left_out.data = left_msg.data
                self.left_raw_pub.publish(left_out)
                
                right_out = Image()
                right_out.header.stamp = sync_stamp
                right_out.header.frame_id = right_msg.header.frame_id
                right_out.height = right_msg.height
                right_out.width = right_msg.width
                right_out.encoding = right_msg.encoding
                right_out.is_bigendian = right_msg.is_bigendian
                right_out.step = right_msg.step
                right_out.data = right_msg.data
                self.right_raw_pub.publish(right_out)
            
            # Compressed 발행
            if self.publish_compressed:
                left_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='passthrough')
                right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='passthrough')
                
                self._publish_compressed_pair(left_cv, right_cv, 
                                               left_msg.header, right_msg.header, sync_stamp)
            
            self.output_count += 1
            self._log_stats()
            
        except Exception as e:
            self.get_logger().error(f'Raw callback error: {e}')
        finally:
            self.is_processing = False
    
    def _synced_callback_compressed(self, left_msg: CompressedImage, right_msg: CompressedImage):
        """동기화된 Compressed 스테레오 이미지 콜백"""
        self.input_count += 1
        
        # Rate limiting
        current_time = time.time()
        if self.is_processing or (current_time - self.last_publish_time) < self.min_interval:
            self.dropped_count += 1
            return
        
        self.is_processing = True
        self.last_publish_time = current_time
        
        try:
            # 통일 타임스탬프
            sync_stamp = self.get_clock().now().to_msg()
            
            # 디코딩
            left_cv = cv2.imdecode(np.frombuffer(left_msg.data, np.uint8), cv2.IMREAD_COLOR)
            right_cv = cv2.imdecode(np.frombuffer(right_msg.data, np.uint8), cv2.IMREAD_COLOR)
            
            if left_cv is None or right_cv is None:
                self.get_logger().warn('Failed to decode compressed image')
                return
            
            # Raw 발행 (BGR → RGB)
            if self.publish_raw:
                left_rgb = cv2.cvtColor(left_cv, cv2.COLOR_BGR2RGB)
                right_rgb = cv2.cvtColor(right_cv, cv2.COLOR_BGR2RGB)
                
                left_raw_msg = self.bridge.cv2_to_imgmsg(left_rgb, encoding='rgb8')
                left_raw_msg.header.stamp = sync_stamp
                left_raw_msg.header.frame_id = left_msg.header.frame_id
                self.left_raw_pub.publish(left_raw_msg)
                
                right_raw_msg = self.bridge.cv2_to_imgmsg(right_rgb, encoding='rgb8')
                right_raw_msg.header.stamp = sync_stamp
                right_raw_msg.header.frame_id = right_msg.header.frame_id
                self.right_raw_pub.publish(right_raw_msg)
            
            # Compressed 발행 (re-stamp)
            if self.publish_compressed:
                left_comp_out = CompressedImage()
                left_comp_out.header.stamp = sync_stamp
                left_comp_out.header.frame_id = left_msg.header.frame_id
                left_comp_out.format = left_msg.format
                left_comp_out.data = left_msg.data  # 재압축 없이 원본 데이터 전달
                self.left_compressed_pub.publish(left_comp_out)
                
                right_comp_out = CompressedImage()
                right_comp_out.header.stamp = sync_stamp
                right_comp_out.header.frame_id = right_msg.header.frame_id
                right_comp_out.format = right_msg.format
                right_comp_out.data = right_msg.data
                self.right_compressed_pub.publish(right_comp_out)
            
            self.output_count += 1
            self._log_stats()
            
        except Exception as e:
            self.get_logger().error(f'Compressed callback error: {e}')
        finally:
            self.is_processing = False
    
    # ==================== Helpers ====================
    
    def _publish_compressed_pair(self, left_cv: np.ndarray, right_cv: np.ndarray,
                                  left_header, right_header, sync_stamp):
        """Raw 이미지를 JPEG 압축하여 Compressed 발행"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        
        # Left
        if len(left_cv.shape) == 3 and left_cv.shape[2] == 3:
            # RGB → BGR for imencode
            left_bgr = cv2.cvtColor(left_cv, cv2.COLOR_RGB2BGR)
        else:
            left_bgr = left_cv
        _, left_data = cv2.imencode('.jpg', left_bgr, encode_param)
        
        left_msg = CompressedImage()
        left_msg.header.stamp = sync_stamp
        left_msg.header.frame_id = left_header.frame_id
        left_msg.format = 'jpeg'
        left_msg.data = left_data.tobytes()
        self.left_compressed_pub.publish(left_msg)
        
        # Right
        if len(right_cv.shape) == 3 and right_cv.shape[2] == 3:
            right_bgr = cv2.cvtColor(right_cv, cv2.COLOR_RGB2BGR)
        else:
            right_bgr = right_cv
        _, right_data = cv2.imencode('.jpg', right_bgr, encode_param)
        
        right_msg = CompressedImage()
        right_msg.header.stamp = sync_stamp
        right_msg.header.frame_id = right_header.frame_id
        right_msg.format = 'jpeg'
        right_msg.data = right_data.tobytes()
        self.right_compressed_pub.publish(right_msg)
    
    def _log_stats(self):
        """5초마다 통계 로깅"""
        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 5e9:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                in_fps = self.input_count / elapsed
                out_fps = self.output_count / elapsed
                drop_rate = (self.dropped_count / max(self.input_count, 1)) * 100
                
                self.get_logger().info(
                    f'[ADJUST] In: {in_fps:.1f} Hz → Out: {out_fps:.1f} Hz '
                    f'(target: {self.target_fps:.0f}), '
                    f'dropped: {self.dropped_count} ({drop_rate:.0f}%), '
                    f'total: {self.output_count} frames'
                )
            
            # 카운터 리셋
            self.input_count = 0
            self.output_count = 0
            self.dropped_count = 0
            self.start_time = time.time()
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = StereoAdjustNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
