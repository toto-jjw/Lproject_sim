#!/usr/bin/env python3
"""
Denoise Node (NAFNet)

NAFNet 모델을 사용하여 노이즈가 있는 이미지를 디노이징하는 ROS2 노드입니다.

구독 토픽:
- /stereo/left/rgb_noisy/compressed
- /stereo/right/rgb_noisy/compressed

발행 토픽:
- /stereo/left/rgb_denoised (sensor_msgs/Image)
- /stereo/left/rgb_denoised/compressed (sensor_msgs/CompressedImage)
- /stereo/right/rgb_denoised (sensor_msgs/Image)
- /stereo/right/rgb_denoised/compressed (sensor_msgs/CompressedImage)

Usage:
    ./run_noiser.sh
    
    또는:
    python3 denoise_node.py --ros-args -p use_raw_input:=false -p max_rate:=15.0
    
참고: 
    사전학습된 NAFNet-SIDD-width64 모델 다운로드 필요:
    https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view
    
    다운로드 후: NAFNet/experiments/pretrained_models/NAFNet-SIDD-width64.pth
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import Optional
import time


# 센서 데이터에 적합한 QoS 프로파일
SENSOR_SUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
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


# ========== Color Transfer 후처리 ==========

def color_transfer_lab(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    LAB 색공간에서 Color Transfer 수행
    
    NAFNet 출력(source)의 색상을 원본(reference)에 맞춤.
    밝기(L)는 NAFNet 결과 유지, 색상(a,b)만 보정.
    
    Args:
        source: NAFNet 디노이징 결과 (RGB, uint8)
        reference: 원본 노이즈 이미지 (RGB, uint8)
        
    Returns:
        색상 보정된 이미지 (RGB, uint8)
    """
    # RGB -> LAB 변환
    src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # L(밝기)는 NAFNet 결과 유지, a,b(색상)만 보정
    for ch in [1, 2]:  # a, b 채널만
        src_mean = src_lab[:, :, ch].mean()
        src_std = src_lab[:, :, ch].std()
        ref_mean = ref_lab[:, :, ch].mean()
        ref_std = ref_lab[:, :, ch].std()
        
        # 표준편차가 0에 가까우면 스킵 (단색 이미지)
        if src_std < 1e-6:
            continue
            
        # 선형 변환: (x - mean_src) * (std_ref / std_src) + mean_ref
        src_lab[:, :, ch] = (src_lab[:, :, ch] - src_mean) * (ref_std / src_std) + ref_mean
    
    # LAB -> RGB 변환
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)
    
    return result


# ========== NAFNet 모델 정의 (NAFNet repository에서 가져옴) ==========

class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D inputs"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SimpleGate(nn.Module):
    """Simple gating mechanism"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Nonlinear Activation Free Block"""
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet: Nonlinear Activation Free Network for Image Restoration
    
    ECCV 2022: "Simple Baselines for Image Restoration"
    https://arxiv.org/abs/2204.04676
    
    SIDD Denoising: 40.30 dB PSNR (SOTA)
    """
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class CameraDenoisingNode(Node):
    """
    NAFNet 모델을 사용한 카메라 이미지 디노이징 ROS2 노드
    
    동기화된 스테레오 처리:
    - ApproximateTimeSynchronizer로 Left/Right 이미지 매칭
    - 동일한 타임스탬프로 출력하여 일관성 보장
    
    사전학습 모델 다운로드:
    1. Google Drive: https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view
    
    2. 다운로드 후 다음 경로에 배치:
       NAFNet/experiments/pretrained_models/NAFNet-SIDD-width64.pth
    """
    
    # 기본 모델 경로 (Lproject_sim/NAFNet 기준)
    NAFNET_BASE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'NAFNet'
    )
    DEFAULT_MODEL_PATH = os.path.join(NAFNET_BASE_DIR, 'experiments', 'pretrained_models', 'NAFNet-SIDD-width64.pth')
    
    # NAFNet-SIDD 모델 설정 (width에 따라 달라짐)
    # width64: 67.89M params, 40.30dB PSNR (SIDD)
    # width32: 17.11M params, 39.97dB PSNR (SIDD) - 더 가벼움
    MODEL_CONFIGS = {
        32: {
            'width': 32,
            'enc_blk_nums': [2, 2, 4, 8],
            'middle_blk_num': 12,
            'dec_blk_nums': [2, 2, 2, 2],
        },
        64: {
            'width': 64,
            'enc_blk_nums': [2, 2, 4, 8],
            'middle_blk_num': 12,
            'dec_blk_nums': [2, 2, 2, 2],
        },
    }
    
    def __init__(self):
        super().__init__('camera_denoising_node')
        
        # CV Bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()
        
        # 성능 측정용
        self._process_times: list = []
        self._frame_count: int = 0
        self._last_log_time = self.get_clock().now()
        self._last_process_time: float = 0.0
        self._is_processing: bool = False
        
        # Declare parameters
        self.declare_parameter('use_raw_input', False)
        self.declare_parameter('max_rate', 15.0)
        self.declare_parameter('model_path', '')  # 빈 문자열이면 width에 따라 자동 결정
        self.declare_parameter('model_width', 64)  # 32 또는 64
        self.declare_parameter('jpeg_quality', 90)
        self.declare_parameter('color_transfer', False)  # NAFNet 색상 드리프트 보정
        
        # 설정 로드
        self.use_raw_input = self.get_parameter('use_raw_input').value
        self.max_rate = self.get_parameter('max_rate').value
        self.model_width = self.get_parameter('model_width').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.color_transfer = self.get_parameter('color_transfer').value
        self.min_interval = 1.0 / self.max_rate if self.max_rate > 0 else 0.0
        
        # model_width 검증
        if self.model_width not in self.MODEL_CONFIGS:
            self.get_logger().warn(f"Invalid model_width {self.model_width}, using 64")
            self.model_width = 64
        
        # 모델 경로 결정 (비어있으면 width에 따라 자동 설정)
        model_path_param = self.get_parameter('model_path').value
        if model_path_param:
            self.model_path = model_path_param
        else:
            self.model_path = f'{self.NAFNET_BASE_DIR}/experiments/pretrained_models/NAFNet-SIDD-width{self.model_width}.pth'
        
        # 디바이스 설정 (GPU 우선)
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.get_logger().info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            self.get_logger().info("Using CPU")
        
        # 모델 초기화
        self.model: Optional[nn.Module] = None
        
        # Publishers (synchronized output)
        self._left_pub_raw = None
        self._left_pub_compressed = None
        self._right_pub_raw = None
        self._right_pub_compressed = None
        
        # Synchronized Subscribers
        self._left_sub = None
        self._right_sub = None
        self._sync = None
        
        self._setup_model()
        self._setup_topics()
        ct_status = "enabled" if self.color_transfer else "disabled"
        self.get_logger().info(f"Denoise Node (NAFNet) initialized with synchronized stereo processing")
        self.get_logger().info(f"Color Transfer: {ct_status} (LAB space, a/b channels only)")
    
    def _setup_model(self):
        """NAFNet 모델 로드 및 초기화"""
        try:
            # 선택된 width에 맞는 설정 사용
            config = self.MODEL_CONFIGS[self.model_width]
            self.get_logger().info(f"Creating NAFNet model with width={self.model_width}")
            
            # NAFNet 모델 생성
            self.model = NAFNet(
                img_channel=3,
                width=config['width'],
                middle_blk_num=config['middle_blk_num'],
                enc_blk_nums=config['enc_blk_nums'],
                dec_blk_nums=config['dec_blk_nums'],
            ).to(self.device)
            
            # 가중치 로드
            if os.path.exists(self.model_path):
                self.get_logger().info(f"Loading NAFNet model from: {self.model_path}")
                
                # 체크포인트 로드
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # BasicSR 형식 처리 (params_ema 또는 params 키)
                if 'params_ema' in checkpoint:
                    state_dict = checkpoint['params_ema']
                    self.get_logger().info("Loaded params_ema from checkpoint")
                elif 'params' in checkpoint:
                    state_dict = checkpoint['params']
                    self.get_logger().info("Loaded params from checkpoint")
                else:
                    state_dict = checkpoint
                
                self.model.load_state_dict(state_dict, strict=True)
                self.model.eval()
                self.get_logger().info("NAFNet model loaded successfully!")
                
                # 모델 정보 출력
                total_params = sum(p.numel() for p in self.model.parameters())
                self.get_logger().info(f"Model parameters: {total_params / 1e6:.2f}M")
                
            else:
                self.get_logger().warn(f"Model file not found: {self.model_path}")
                self.get_logger().warn(
                    "=== NAFNet 사전학습 모델 다운로드 방법 ===\n"
                    "1. Google Drive에서 다운로드:\n"
                    "   NAFNet-SIDD-width64 (40.30dB, 67.89M params):\n"
                    "     https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view\n"
                    "   NAFNet-SIDD-width32 (39.97dB, 17.11M params, 경량):\n"
                    "     https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view\n"
                    "2. 다운로드 후 다음 폴더에 배치:\n"
                    f"   {self.model_path}"
                )
                # 랜덤 초기화 모델 사용 (테스트용)
                self.get_logger().warn("Using randomly initialized model (for testing only)")
                self.model.eval()
                
        except Exception as e:
            self.get_logger().error(f"Failed to load NAFNet model: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.model = None
    
    def _setup_topics(self):
        """토픽 구독자/발행자 설정 (synchronized stereo)"""
        # --- Publishers ---
        # Left Raw
        self._left_pub_raw = self.create_publisher(
            Image, '/stereo/left/rgb_denoised', RAW_PUB_QOS)
        # Left Compressed
        self._left_pub_compressed = self.create_publisher(
            CompressedImage, '/stereo/left/rgb_denoised/compressed', COMPRESSED_PUB_QOS)
        # Right Raw
        self._right_pub_raw = self.create_publisher(
            Image, '/stereo/right/rgb_denoised', RAW_PUB_QOS)
        # Right Compressed
        self._right_pub_compressed = self.create_publisher(
            CompressedImage, '/stereo/right/rgb_denoised/compressed', COMPRESSED_PUB_QOS)
        
        # --- Synchronized Subscribers (message_filters) ---
        if self.use_raw_input:
            left_topic = '/stereo/left/rgb_noisy'
            right_topic = '/stereo/right/rgb_noisy'
            msg_type = Image
            callback = self._synced_raw_callback
            sub_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )
        else:
            left_topic = '/stereo/left/rgb_noisy/compressed'
            right_topic = '/stereo/right/rgb_noisy/compressed'
            msg_type = CompressedImage
            callback = self._synced_compressed_callback
            sub_qos = SENSOR_SUB_QOS
        
        self._left_sub = Subscriber(self, msg_type, left_topic, qos_profile=sub_qos)
        self._right_sub = Subscriber(self, msg_type, right_topic, qos_profile=sub_qos)
        
        # ApproximateTimeSynchronizer: 타임스탬프가 비슷한 Left/Right 이미지를 매칭
        self._sync = ApproximateTimeSynchronizer(
            [self._left_sub, self._right_sub],
            queue_size=10,
            slop=0.1  # 100ms 이내의 타임스탬프 차이 허용
        )
        self._sync.registerCallback(callback)
        
        self.get_logger().info(f"  Left input:  {left_topic}")
        self.get_logger().info(f"  Right input: {right_topic}")
        self.get_logger().info(f"  Left output:  /stereo/left/rgb_denoised[/compressed]")
        self.get_logger().info(f"  Right output: /stereo/right/rgb_denoised[/compressed]")
        self.get_logger().info(f"  Sync slop: 100ms (ApproximateTimeSynchronizer)")
    
    def _synced_compressed_callback(self, left_msg: CompressedImage, right_msg: CompressedImage):
        """동기화된 Left/Right Compressed 이미지 처리 콜백"""
        if self.model is None:
            return
        
        # Rate limiting
        current_time = time.time()
        if (current_time - self._last_process_time) < self.min_interval:
            return
        
        # 이미 처리 중이면 스킵
        if self._is_processing:
            return
        
        self._is_processing = True
        self._last_process_time = current_time
        
        try:
            # Decode left image
            left_arr = np.frombuffer(left_msg.data, np.uint8)
            left_cv = cv2.imdecode(left_arr, cv2.IMREAD_COLOR)
            if left_cv is None:
                self.get_logger().error("Failed to decode left compressed image")
                return
            left_rgb = cv2.cvtColor(left_cv, cv2.COLOR_BGR2RGB)
            
            # Decode right image
            right_arr = np.frombuffer(right_msg.data, np.uint8)
            right_cv = cv2.imdecode(right_arr, cv2.IMREAD_COLOR)
            if right_cv is None:
                self.get_logger().error("Failed to decode right compressed image")
                return
            right_rgb = cv2.cvtColor(right_cv, cv2.COLOR_BGR2RGB)
            
            # Process and publish
            self._process_and_publish_stereo(
                left_rgb, right_rgb, 
                left_msg.header, right_msg.header
            )
            
        except Exception as e:
            self.get_logger().error(f"Error processing compressed images: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self._is_processing = False
    
    def _synced_raw_callback(self, left_msg: Image, right_msg: Image):
        """동기화된 Left/Right Raw 이미지 처리 콜백"""
        if self.model is None:
            return
        
        # Rate limiting
        current_time = time.time()
        if (current_time - self._last_process_time) < self.min_interval:
            return
        
        # 이미 처리 중이면 스킵
        if self._is_processing:
            return
        
        self._is_processing = True
        self._last_process_time = current_time
        
        try:
            # Decode left image
            left_cv = self.bridge.imgmsg_to_cv2(left_msg, desired_encoding='rgb8')
            
            # Decode right image  
            right_cv = self.bridge.imgmsg_to_cv2(right_msg, desired_encoding='rgb8')
            
            # Process and publish
            self._process_and_publish_stereo(
                left_cv, right_cv,
                left_msg.header, right_msg.header
            )
            
        except Exception as e:
            self.get_logger().error(f"Error processing raw images: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            self._is_processing = False
    
    def _process_and_publish_stereo(self, left_rgb: np.ndarray, right_rgb: np.ndarray,
                                     left_header, right_header):
        """동기화된 스테레오 이미지 처리 및 발행"""
        start_time = time.time()
        
        # 디노이징 수행 (left, right 순차적으로)
        denoised_left = self._denoise(left_rgb)
        denoised_right = self._denoise(right_rgb)
        
        # === Left Raw Image 발행 ===
        left_raw_msg = self.bridge.cv2_to_imgmsg(denoised_left, encoding='rgb8')
        left_raw_msg.header = left_header
        self._left_pub_raw.publish(left_raw_msg)
        
        # === Left Compressed Image 발행 ===
        left_bgr = cv2.cvtColor(denoised_left, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, left_compressed_data = cv2.imencode('.jpg', left_bgr, encode_param)
        
        left_compressed_msg = CompressedImage()
        left_compressed_msg.header = left_header
        left_compressed_msg.format = 'jpeg'
        left_compressed_msg.data = left_compressed_data.tobytes()
        self._left_pub_compressed.publish(left_compressed_msg)
        
        # === Right Raw Image 발행 ===
        right_raw_msg = self.bridge.cv2_to_imgmsg(denoised_right, encoding='rgb8')
        right_raw_msg.header = right_header
        self._right_pub_raw.publish(right_raw_msg)
        
        # === Right Compressed Image 발행 ===
        right_bgr = cv2.cvtColor(denoised_right, cv2.COLOR_RGB2BGR)
        _, right_compressed_data = cv2.imencode('.jpg', right_bgr, encode_param)
        
        right_compressed_msg = CompressedImage()
        right_compressed_msg.header = right_header
        right_compressed_msg.format = 'jpeg'
        right_compressed_msg.data = right_compressed_data.tobytes()
        self._right_pub_compressed.publish(right_compressed_msg)
        
        # 성능 측정
        process_time = time.time() - start_time
        self._process_times.append(process_time)
        self._frame_count += 1
        
        # 5초마다 로깅
        now = self.get_clock().now()
        if (now - self._last_log_time).nanoseconds > 5e9:
            if self._process_times:
                avg_ms = sum(self._process_times) / len(self._process_times) * 1000
                fps = self._frame_count / 5.0
                self.get_logger().info(
                    f'[Stereo Denoise] Avg: {avg_ms:.1f}ms, Output: {fps:.1f} FPS (synced pairs)'
                )
            self._process_times = []
            self._frame_count = 0
            self._last_log_time = now
    
    def _denoise(self, image: np.ndarray, original: np.ndarray = None) -> np.ndarray:
        """
        NAFNet을 사용한 이미지 디노이징 + Color Transfer 보정
        
        Args:
            image: RGB 이미지 (H, W, 3), uint8
            original: 원본 노이즈 이미지 (color transfer용, None이면 image 사용)
            
        Returns:
            denoised: RGB 이미지 (H, W, 3), uint8
        """
        # Color transfer용 원본 저장
        if original is None:
            original = image
        
        with torch.no_grad():
            # Normalize to [0, 1] and convert to tensor
            img_tensor = torch.from_numpy(image).float() / 255.0
            
            # (H, W, C) -> (1, C, H, W)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # NAFNet 추론 (padding은 모델 내부에서 처리)
            output = self.model(img_tensor)
            
            # Tensor to numpy
            output = output.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            output = output.cpu().numpy()
            
            # Clip and convert to uint8
            output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
            
            # Color Transfer: NAFNet 출력의 색상을 원본에 맞춤
            if self.color_transfer:
                output = color_transfer_lab(output, original)
            
            return output


def main(args=None):
    rclpy.init(args=args)
    
    node = CameraDenoisingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
