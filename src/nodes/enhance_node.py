#!/usr/bin/env python3
"""
Enhance Node (DimCam Stereo Enhancement)
DimCam 모델을 사용하여 스테레오 이미지를 실시간으로 향상시킵니다.

Subscribed Topics:
    /stereo/left/rgb_noisy/compressed (sensor_msgs/CompressedImage): Left camera RGB image
    /stereo/right/rgb_noisy/compressed (sensor_msgs/CompressedImage): Right camera RGB image

Published Topics:
    /stereo/left/enhanced/compressed (sensor_msgs/CompressedImage): Enhanced left image
    /stereo/right/enhanced/compressed (sensor_msgs/CompressedImage): Enhanced right image
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
import time

# 모델 경로 추가
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
LPROJECT_CAM_DIR = os.path.join(PROJECT_ROOT, 'Lproject_cam')
sys.path.append(SCRIPT_DIR)
sys.path.append(LPROJECT_CAM_DIR)  # DimCam 모델 임포트용


class DimCamEnhancerNode(Node):
    """DimCam Stereo Image Enhancement ROS2 Node (Original Model)"""
    
    def __init__(self):
        super().__init__('dimcam_enhancer_node')
        
        # --- 파라미터 선언 ---
        self.declare_parameter('model_weights', 
            os.path.join(LPROJECT_CAM_DIR, 'dimcam_enhancer_epoch_30.pth'))
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('img_size', 512)
        self.declare_parameter('embed_dim', 64)  # ★ 체크포인트와 일치
        self.declare_parameter('num_blocks', 5)
        self.declare_parameter('use_fp16', True)
        self.declare_parameter('max_rate', 15.0)  # 목표 처리 Hz
        self.declare_parameter('input_mode', 'noisy')  # 'noisy', 'denoised', or 'raw'
        self.declare_parameter('left_topic', '')  # 비어있으면 기본값 사용
        self.declare_parameter('right_topic', '')  # 비어있으면 기본값 사용
        
        # 파라미터 로드
        self.model_weights = self.get_parameter('model_weights').value
        self.device_name = self.get_parameter('device').value
        self.img_size = self.get_parameter('img_size').value
        self.embed_dim = self.get_parameter('embed_dim').value
        self.num_blocks = self.get_parameter('num_blocks').value
        self.use_fp16 = self.get_parameter('use_fp16').value
        self.max_rate = self.get_parameter('max_rate').value
        self.input_mode = self.get_parameter('input_mode').value  # 'noisy', 'denoised', 'raw'
        self.left_topic = self.get_parameter('left_topic').value
        self.right_topic = self.get_parameter('right_topic').value
        
        # --- 디바이스 설정 ---
        if self.device_name == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.get_logger().info(f'Using GPU: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device('cpu')
            self.get_logger().info('Using CPU')
        
        # --- 모델 로드 ---
        self.model = None
        self._load_model()
        
        # --- CV Bridge ---
        self.bridge = CvBridge()
        
        # --- 처리 상태 ---
        self.process_count = 0
        self.start_time = time.time()
        self.last_process_time = 0.0
        self.is_processing = False
        
        # --- CUDA 최적화 ---
        self._postprocess_stream = None
        if self.device.type == 'cuda':
            self._postprocess_stream = torch.cuda.Stream()
        
        # --- QoS 설정 ---
        # Raw Input: RELIABLE (원본 토픽은 RELIABLE)
        # Compressed Input: BEST_EFFORT (속도 향상)
        if self.input_mode == 'raw':
            sub_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )
            # 토픽 결정
            left_topic = self.left_topic if self.left_topic else '/stereo/left/rgb'
            right_topic = self.right_topic if self.right_topic else '/stereo/right/rgb'
            msg_type = Image
            callback = self.synced_callback_raw
        elif self.input_mode == 'denoised':
            sub_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )
            # 토픽 결정 (denoised compressed)
            left_topic = self.left_topic if self.left_topic else '/stereo/left/rgb_denoised/compressed'
            right_topic = self.right_topic if self.right_topic else '/stereo/right/rgb_denoised/compressed'
            msg_type = CompressedImage
            callback = self.synced_callback_compressed
        else:  # 'noisy' (default)
            sub_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )
            # 토픽 결정
            left_topic = self.left_topic if self.left_topic else '/stereo/left/rgb_noisy/compressed'
            right_topic = self.right_topic if self.right_topic else '/stereo/right/rgb_noisy/compressed'
            msg_type = CompressedImage
            callback = self.synced_callback_compressed
        
        # --- Synchronized Subscribers (message_filters) ---
        self.left_sub = Subscriber(self, msg_type, left_topic, qos_profile=sub_qos)
        self.right_sub = Subscriber(self, msg_type, right_topic, qos_profile=sub_qos)
        
        # ApproximateTimeSynchronizer: 타임스탬프가 비슷한 Left/Right 이미지를 매칭
        self.sync = ApproximateTimeSynchronizer(
            [self.left_sub, self.right_sub],
            queue_size=10,
            slop=0.01  # 10ms 이내의 타임스탬프 차이 허용
        )
        self.sync.registerCallback(callback)
        
        self.get_logger().info(f'  - Input mode: {self.input_mode}')
        self.get_logger().info(f'  - Left topic: {left_topic}')
        self.get_logger().info(f'  - Right topic: {right_topic}')
        
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
        self.left_enhanced_pub = self.create_publisher(CompressedImage, '/stereo/left/enhanced/compressed', compressed_qos)
        self.right_enhanced_pub = self.create_publisher(CompressedImage, '/stereo/right/enhanced/compressed', compressed_qos)
        # Raw Image publishers (for RViz - RELIABLE)
        self.left_raw_pub = self.create_publisher(Image, '/stereo/left/enhanced', raw_qos)
        self.right_raw_pub = self.create_publisher(Image, '/stereo/right/enhanced', raw_qos)
        
        self.get_logger().info('DimCam Enhancer Node initialized (Original Model)')
        self.get_logger().info(f'  - Model weights: {self.model_weights}')
        self.get_logger().info(f'  - Image size: {self.img_size}x{self.img_size}')
        self.get_logger().info(f'  - Embed dim: {self.embed_dim}, Num blocks: {self.num_blocks}')
        self.get_logger().info(f'  - Target rate: {self.max_rate} Hz')
        self.get_logger().info(f'  - FP16 inference: {self.use_fp16}')
    
    def _load_model(self):
        """기존 DimCam 모델 로드"""
        try:
            # 기존 model.py 임포트
            import model as dimcam_model
            
            self.get_logger().info('Loading DimCam model (Original)...')
            
            # 기존 모델 초기화 (lambda_depth=0으로 depth 비활성화)
            self.model = dimcam_model.DimCamEnhancer(
                img_size=self.img_size,
                embed_dim=self.embed_dim,
                num_blocks=self.num_blocks,
                lambda_depth=0.0,  # Depth 비활성화
                use_tiled_inference=False,
            ).to(self.device)
            
            # 가중치 로드
            if os.path.exists(self.model_weights):
                state_dict = torch.load(self.model_weights, map_location=self.device)
                
                # depth_net 관련 가중치 제거
                filtered_state_dict = {
                    k: v for k, v in state_dict.items() 
                    if not k.startswith('depth_net')
                }
                self.model.load_state_dict(filtered_state_dict, strict=False)
                self.get_logger().info(f'Loaded model weights from {self.model_weights}')
            else:
                self.get_logger().warn(f'Model weights not found: {self.model_weights}')
                self.get_logger().warn('Using randomly initialized weights!')
            
            self.model.eval()
            
            # FP16 변환
            if self.use_fp16 and self.device.type == 'cuda':
                self.model = self.model.half()
                self.get_logger().info('Model converted to FP16')
            
            self.get_logger().info('Model loaded successfully!')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
    
    def synced_callback_compressed(self, left_msg: CompressedImage, right_msg: CompressedImage):
        """동기화된 Left/Right Compressed 이미지 처리 콜백"""
        # Rate-limit and processing guard BEFORE expensive decode+GPU transfer
        current_time = time.time()
        if self.is_processing or (current_time - self.last_process_time) < (1.0 / self.max_rate):
            return
        
        self.is_processing = True
        self.last_process_time = current_time
        
        try:
            left_tensor, left_size = self._compressed_msg_to_tensor(left_msg)
            right_tensor, right_size = self._compressed_msg_to_tensor(right_msg)
            self._process_and_publish(left_tensor, right_tensor, left_size, right_size, 
                                     left_msg.header, right_msg.header)
        except Exception as e:
            self.get_logger().error(f'Error processing images: {e}')
        finally:
            self.is_processing = False
    
    def synced_callback_raw(self, left_msg: Image, right_msg: Image):
        """동기화된 Left/Right Raw 이미지 처리 콜백"""
        # Rate-limit and processing guard BEFORE expensive decode+GPU transfer
        current_time = time.time()
        if self.is_processing or (current_time - self.last_process_time) < (1.0 / self.max_rate):
            return
        
        self.is_processing = True
        self.last_process_time = current_time
        
        try:
            left_tensor, left_size = self._raw_msg_to_tensor(left_msg)
            right_tensor, right_size = self._raw_msg_to_tensor(right_msg)
            self._process_and_publish(left_tensor, right_tensor, left_size, right_size, 
                                     left_msg.header, right_msg.header)
        except Exception as e:
            self.get_logger().error(f'Error processing images: {e}')
        finally:
            self.is_processing = False
    
    def _process_and_publish(self, left_tensor, right_tensor, left_size, right_size, left_header, right_header):
        """공통 처리 및 발행 로직
        
        Note: Rate-limiting and is_processing guard are handled by the caller
        (synced_callback_compressed / synced_callback_raw).
        """
        start_time = time.time()
        
        inference_start = time.time()
        
        # 모델 추론 (single-threaded spin → Lock 불필요)
        with torch.no_grad():
            if self.use_fp16 and self.device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = self.model(left_tensor, right_tensor)
            else:
                outputs = self.model(left_tensor, right_tensor)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # outputs: (enhanced_l, enhanced_r, depth, dpce_only_l, dpce_only_r, gamma_l, gamma_r)
            enhanced_left = outputs[0].clamp(0, 1)
            enhanced_right = outputs[1].clamp(0, 1)
        
        inference_time = time.time() - inference_start
        
        # 통일된 타임스탬프 (noise_node과 동일한 방식 - 스테레오 동기화 보장)
        sync_stamp = self.get_clock().now().to_msg()
        
        # GPU→CPU 변환을 한 번만 수행하여 Raw + Compressed 모두에 재사용
        left_np = self._tensor_to_numpy(enhanced_left, left_size)
        right_np = self._tensor_to_numpy(enhanced_right, right_size)
        
        # Compressed 발행
        left_enhanced_msg = self._numpy_to_compressed_msg(left_np, left_header, sync_stamp)
        right_enhanced_msg = self._numpy_to_compressed_msg(right_np, right_header, sync_stamp)
        self.left_enhanced_pub.publish(left_enhanced_msg)
        self.right_enhanced_pub.publish(right_enhanced_msg)
        
        # Raw Image 발행 (RViz용) - 동일 numpy 재사용
        left_raw_msg = self._numpy_to_raw_msg(left_np, left_header, sync_stamp)
        right_raw_msg = self._numpy_to_raw_msg(right_np, right_header, sync_stamp)
        self.left_raw_pub.publish(left_raw_msg)
        self.right_raw_pub.publish(right_raw_msg)
        
        self.process_count += 1
        
        # 성능 로깅 (5초마다)
        process_time = time.time() - start_time
        current_time = time.time()
        elapsed = current_time - self.start_time
        if elapsed > 5.0 and self.process_count > 0:
            avg_fps = self.process_count / elapsed
            self.get_logger().info(
                f'Inference: {inference_time*1000:.1f}ms, Total: {process_time*1000:.1f}ms, '
                f'Output: {avg_fps:.1f} FPS'
            )
            self.start_time = current_time
            self.process_count = 0
    
    def _compressed_msg_to_tensor(self, msg: CompressedImage) -> torch.Tensor:
        """ROS CompressedImage 메시지를 PyTorch 텐서로 변환"""
        # Compressed image 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return self._image_to_tensor(cv_image)
    
    def _raw_msg_to_tensor(self, msg: Image) -> torch.Tensor:
        """ROS Raw Image 메시지를 PyTorch 텐서로 변환"""
        # cv_bridge로 디코딩
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        return self._image_to_tensor(cv_image)
    
    def _image_to_tensor(self, cv_image: np.ndarray) -> tuple:
        """OpenCV 이미지를 PyTorch 텐서로 변환 (공통 로직)"""
        original_h, original_w = cv_image.shape[:2]
        
        cv_image = cv2.resize(cv_image, (self.img_size, self.img_size))
        image = cv_image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)
        
        if self.use_fp16 and self.device.type == 'cuda':
            tensor = tensor.half()
        
        return tensor, (original_h, original_w)
    
    def _tensor_to_numpy(self, tensor: torch.Tensor, original_size: tuple) -> np.ndarray:
        """PyTorch 텐서를 numpy uint8 RGB 이미지로 변환 (한 번만 수행)
        
        이 결과를 _numpy_to_compressed_msg / _numpy_to_raw_msg에서 재사용하여
        중복 GPU→CPU 전송 및 후처리를 제거합니다.
        """
        image = tensor.squeeze(0).float().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        
        original_h, original_w = original_size
        if image.shape[0] != original_h or image.shape[1] != original_w:
            image = cv2.resize(image, (original_w, original_h))
        
        return image
    
    def _numpy_to_compressed_msg(self, image_rgb: np.ndarray, header, 
                                  sync_stamp) -> CompressedImage:
        """numpy RGB 이미지를 CompressedImage로 변환 (통일 타임스탬프 사용)"""
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, compressed_data = cv2.imencode('.jpg', image_bgr, encode_param)
        
        msg = CompressedImage()
        msg.header.stamp = sync_stamp
        msg.header.frame_id = header.frame_id
        msg.format = 'jpeg'
        msg.data = compressed_data.tobytes()
        
        return msg
    
    def _numpy_to_raw_msg(self, image_rgb: np.ndarray, header, 
                          sync_stamp) -> Image:
        """numpy RGB 이미지를 Raw Image로 변환 (통일 타임스탬프 사용, RViz용)"""
        msg = self.bridge.cv2_to_imgmsg(image_rgb, encoding='rgb8')
        msg.header.stamp = sync_stamp
        msg.header.frame_id = header.frame_id
        return msg


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = DimCamEnhancerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
