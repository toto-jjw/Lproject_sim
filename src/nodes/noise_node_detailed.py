#!/usr/bin/env python3
# Lproject_sim/src/nodes/noise_node_detailed.py
"""
Noise Node (Detailed Version)

ROS2 이미지 토픽에 센서 노이즈를 추가하는 노드입니다.
원본 토픽을 구독하고 노이즈가 적용된 이미지를 새 토픽으로 발행합니다.

지원하는 노이즈 유형:
- Gaussian noise (가우시안 노이즈)
- Salt & Pepper noise (점 노이즈)
- Exposure variation (노출 변화)
- Motion blur (모션 블러) - TODO

Usage:
    ros2 run <package> noise_node --ros-args -p config_file:=/path/to/config.yaml
    
    또는 launch 파일에서:
    ros2 launch <package> camera_noise.launch.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import numpy as np
import cv2
import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# 센서 데이터에 적합한 QoS 프로파일
# Subscriber: BEST_EFFORT (원본 센서 데이터 수신)
SENSOR_SUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,  # 최신 프레임만 유지
    durability=DurabilityPolicy.VOLATILE
)

# Publisher (Raw Image): RELIABLE (메시지 손실 방지)
RAW_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE
)

# Publisher (Compressed Image): BEST_EFFORT (실시간 스트리밍, 프레임 드롭 허용)
COMPRESSED_PUB_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,  # 최신 프레임만 유지
    durability=DurabilityPolicy.VOLATILE
)


@dataclass
class NoiseConfig:
    """노이즈 설정 데이터 클래스"""
    enabled: bool = True
    
    # ========== 기존 단순 노이즈 모델 ==========
    # Gaussian Noise
    gaussian_enabled: bool = True
    gaussian_mean: float = 0.0
    gaussian_std: float = 5.0  # 픽셀 값 기준 (0-255 스케일)
    
    # Salt & Pepper Noise
    salt_pepper_enabled: bool = False
    salt_pepper_prob: float = 0.001  # 각 픽셀이 노이즈가 될 확률
    
    # Exposure Variation
    exposure_enabled: bool = False
    exposure_variation: float = 0.1  # 밝기 변화 비율 (±10%)
    
    # Depth Noise (depth 이미지 전용)
    depth_gaussian_std: float = 0.01  # 미터 단위
    depth_dropout_prob: float = 0.001  # 깊이 값 손실 확률
    
    # ========== 물리 기반 노이즈 모델 (Moseley et al. CVPR 2021) ==========
    # 달 극지방 영구 그림자 영역(PSR) 저조도 환경용
    # 참조: "Extreme Low-Light Environment-Driven Image Denoising Over 
    #       Permanently Shadowed Lunar Regions"
    physical_noise_enabled: bool = False
    
    # 센서 파라미터 (카메라 스펙에 따라 조정)
    quantum_efficiency: float = 0.7      # 양자 효율 (0-1)
    analog_gain: float = 1.0             # 아날로그 게인
    full_well_capacity: int = 10000      # 풀웰 용량 (electrons)
    bit_depth: int = 8                   # ADC 비트 깊이
    
    # Shot Noise (포아송 분포 - 광자 노이즈)
    shot_noise_enabled: bool = True
    
    # Read Noise (가우시안 분포 - 전자 회로 노이즈)
    read_noise_std: float = 5.0          # electrons (RMS)
    
    # Dark Current (온도 의존적)
    dark_current_enabled: bool = True
    dark_current_rate: float = 0.1       # electrons/pixel/second
    exposure_time: float = 0.033         # 노출 시간 (초) - 30fps 기준
    sensor_temperature: float = 293.0    # 센서 온도 (K) - 20°C
    
    # Fixed Pattern Noise (FPN) - 픽셀별 고정 오프셋
    fpn_enabled: bool = True
    fpn_strength: float = 0.02           # 전체 범위 대비 비율 (2%)
    
    # Photo Response Non-Uniformity (PRNU) - 픽셀 감도 편차
    prnu_enabled: bool = True
    prnu_strength: float = 0.01          # 감도 편차 비율 (1%)


@dataclass  
class TopicConfig:
    """토픽별 설정"""
    input_topic: str
    output_topic: str
    image_type: str = "rgb"  # "rgb" or "depth"
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)


class CameraNoiseNode(Node):
    """
    카메라 이미지에 센서 노이즈를 추가하는 ROS2 노드
    """
    
    # 기본 설정 파일 경로
    DEFAULT_CONFIG_FILE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'config', 'simulation_config.yaml'
    )
    
    def __init__(self):
        super().__init__('camera_noise_node')
        
        # CV Bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()
        
        # 성능 측정용
        self._process_times: Dict[str, list] = {}
        self._frame_counts: Dict[str, int] = {}
        self._last_log_time = self.get_clock().now()
        
        # Declare parameters
        self.declare_parameter('config_file', self.DEFAULT_CONFIG_FILE)
        self.declare_parameter('enabled', True)
        
        # 개별 노이즈 파라미터 (config 파일 없이 사용 가능)
        self.declare_parameter('gaussian_enabled', True)
        self.declare_parameter('gaussian_mean', 0.0)
        self.declare_parameter('gaussian_std', 5.0)
        self.declare_parameter('salt_pepper_enabled', False)
        self.declare_parameter('salt_pepper_prob', 0.001)
        self.declare_parameter('exposure_enabled', False)
        self.declare_parameter('exposure_variation', 0.1)
        self.declare_parameter('depth_gaussian_std', 0.01)
        self.declare_parameter('depth_dropout_prob', 0.001)
        
        # 토픽 설정 파라미터
        self.declare_parameter('rgb_topics', [
            '/stereo/left/rgb',
            '/stereo/right/rgb'
        ])
        self.declare_parameter('depth_topics', [
            '/front_camera/depth/depth'
        ])
        self.declare_parameter('output_suffix', '_noisy')
        
        # 설정 로드
        self.load_config()
        
        # Publishers & Subscribers 초기화
        self._subs: Dict[str, Any] = {}
        self._pubs: Dict[str, Any] = {}
        
        # 고정 노이즈 마스크 (토픽별로 저장)
        # Salt & Pepper 노이즈의 위치를 고정하기 위해 사용
        self._salt_masks: Dict[str, np.ndarray] = {}
        self._pepper_masks: Dict[str, np.ndarray] = {}
        self._depth_dropout_masks: Dict[str, np.ndarray] = {}
        self._noise_buffers: Dict[str, Any] = {}  # 노이즈 생성 최적화용 버퍼
        
        # 실시간 센서 온도 (토픽에서 업데이트됨)
        self._current_sensor_temp: float = 293.0  # 기본값 20°C (K)
        self._temp_topic: str = '/rover/sensor_temperature'  # 온도 토픽
        
        if self.enabled:
            self.setup_topics()
            self.setup_temperature_subscriber()
            self.get_logger().info(f"Camera Noise Node initialized with {len(self.topic_configs)} topics")
        else:
            self.get_logger().info("Camera Noise Node disabled")
    
    def setup_temperature_subscriber(self):
        """
        로버 센서 온도 토픽 구독 설정
        
        온도는 Dark Current에 영향을 줍니다:
        - 온도가 10°C 상승할 때마다 암전류가 약 2배 증가 (Arrhenius 법칙)
        - 달 표면: 낮 ~127°C, 밤 ~-173°C, PSR ~-230°C
        """
        # 온도 토픽 파라미터
        self.declare_parameter('temperature_topic', self._temp_topic)
        self._temp_topic = self.get_parameter('temperature_topic').value
        
        # 초기 온도 설정 (config에서 로드된 값 사용)
        self._current_sensor_temp = self.default_noise_config.sensor_temperature
        
        # 온도 토픽 구독 (있으면 실시간 업데이트)
        self._temp_sub = self.create_subscription(
            Float64,
            self._temp_topic,
            self._temperature_callback,
            10
        )
        self.get_logger().info(
            f"Temperature subscriber: {self._temp_topic} "
            f"(initial: {self._current_sensor_temp:.1f}K = {self._current_sensor_temp - 273.15:.1f}°C)"
        )
    
    def _temperature_callback(self, msg: Float64):
        """
        온도 토픽 콜백
        
        온도 단위 자동 감지:
        - 값이 200 이상이면 Kelvin으로 간주
        - 값이 200 미만이면 Celsius로 간주하여 Kelvin으로 변환
        """
        temp = msg.data
        
        # 단위 자동 감지 및 변환
        if temp < 200:  # Celsius로 간주
            temp_k = temp + 273.15
        else:  # Kelvin으로 간주
            temp_k = temp
        
        # 유효 범위 체크 (달 표면: 40K ~ 400K)
        if 40.0 <= temp_k <= 400.0:
            old_temp = self._current_sensor_temp
            self._current_sensor_temp = temp_k
            
            # 온도 변화가 클 때만 로깅 (10K 이상)
            if abs(temp_k - old_temp) > 10.0:
                dc_factor = 2.0 ** ((temp_k - 293.0) / 10.0)
                self.get_logger().info(
                    f"Sensor temp updated: {temp_k:.1f}K ({temp_k - 273.15:.1f}°C), "
                    f"Dark current factor: {dc_factor:.2f}x"
                )
    
    def load_config(self):
        """설정 파일 또는 파라미터에서 설정 로드"""
        config_file = self.get_parameter('config_file').value
        
        self.enabled = self.get_parameter('enabled').value
        
        # 기본 노이즈 설정
        self.default_noise_config = NoiseConfig(
            enabled=self.enabled,
            gaussian_enabled=self.get_parameter('gaussian_enabled').value,
            gaussian_mean=self.get_parameter('gaussian_mean').value,
            gaussian_std=self.get_parameter('gaussian_std').value,
            salt_pepper_enabled=self.get_parameter('salt_pepper_enabled').value,
            salt_pepper_prob=self.get_parameter('salt_pepper_prob').value,
            exposure_enabled=self.get_parameter('exposure_enabled').value,
            exposure_variation=self.get_parameter('exposure_variation').value,
            depth_gaussian_std=self.get_parameter('depth_gaussian_std').value,
            depth_dropout_prob=self.get_parameter('depth_dropout_prob').value,
        )
        
        self.topic_configs: list[TopicConfig] = []
        
        # Config 파일에서 로드 시도
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    
                env_config = yaml_config.get('environment', {})
                noise_config = env_config.get('camera_noise', {})
                
                self.enabled = noise_config.get('enabled', self.enabled)
                
                # YAML에서 노이즈 설정 오버라이드
                if 'gaussian' in noise_config:
                    g = noise_config['gaussian']
                    self.default_noise_config.gaussian_enabled = g.get('enabled', True)
                    self.default_noise_config.gaussian_mean = g.get('mean', 0.0)
                    self.default_noise_config.gaussian_std = g.get('std', 5.0)
                    
                if 'salt_pepper' in noise_config:
                    sp = noise_config['salt_pepper']
                    self.default_noise_config.salt_pepper_enabled = sp.get('enabled', False)
                    self.default_noise_config.salt_pepper_prob = sp.get('prob', 0.001)
                    
                if 'exposure' in noise_config:
                    exp = noise_config['exposure']
                    self.default_noise_config.exposure_enabled = exp.get('enabled', False)
                    self.default_noise_config.exposure_variation = exp.get('variation', 0.1)
                    
                if 'depth' in noise_config:
                    d = noise_config['depth']
                    self.default_noise_config.depth_gaussian_std = d.get('gaussian_std', 0.01)
                    self.default_noise_config.depth_dropout_prob = d.get('dropout_prob', 0.001)
                
                # 물리 기반 노이즈 설정 로드
                if 'physical' in noise_config:
                    p = noise_config['physical']
                    self.default_noise_config.physical_noise_enabled = p.get('enabled', False)
                    self.default_noise_config.quantum_efficiency = p.get('quantum_efficiency', 0.7)
                    self.default_noise_config.analog_gain = p.get('analog_gain', 1.0)
                    self.default_noise_config.full_well_capacity = p.get('full_well_capacity', 10000)
                    self.default_noise_config.bit_depth = p.get('bit_depth', 8)
                    
                    if 'shot_noise' in p:
                        self.default_noise_config.shot_noise_enabled = p['shot_noise'].get('enabled', True)
                    
                    if 'read_noise' in p:
                        self.default_noise_config.read_noise_std = p['read_noise'].get('std', 5.0)
                    
                    if 'dark_current' in p:
                        dc = p['dark_current']
                        self.default_noise_config.dark_current_enabled = dc.get('enabled', True)
                        self.default_noise_config.dark_current_rate = dc.get('rate', 0.1)
                        self.default_noise_config.exposure_time = dc.get('exposure_time', 0.033)
                        self.default_noise_config.sensor_temperature = dc.get('sensor_temperature', 293.0)
                    
                    if 'fpn' in p:
                        self.default_noise_config.fpn_enabled = p['fpn'].get('enabled', True)
                        self.default_noise_config.fpn_strength = p['fpn'].get('strength', 0.02)
                    
                    if 'prnu' in p:
                        self.default_noise_config.prnu_enabled = p['prnu'].get('enabled', True)
                        self.default_noise_config.prnu_strength = p['prnu'].get('strength', 0.01)
                    
                    if self.default_noise_config.physical_noise_enabled:
                        self.get_logger().info(
                            f"Physical noise model enabled: "
                            f"QE={self.default_noise_config.quantum_efficiency}, "
                            f"FWC={self.default_noise_config.full_well_capacity}, "
                            f"read_noise={self.default_noise_config.read_noise_std}e-"
                        )
                
                # 토픽 설정
                if 'topics' in noise_config:
                    topics = noise_config['topics']
                    for t in topics.get('rgb', []):
                        self.topic_configs.append(TopicConfig(
                            input_topic=t,
                            output_topic=t + noise_config.get('output_suffix', '_noisy'),
                            image_type='rgb',
                            noise_config=self.default_noise_config
                        ))
                    for t in topics.get('depth', []):
                        self.topic_configs.append(TopicConfig(
                            input_topic=t,
                            output_topic=t + noise_config.get('output_suffix', '_noisy'),
                            image_type='depth',
                            noise_config=self.default_noise_config
                        ))
                        
                self.get_logger().info(f"Loaded config from: {config_file}")
                
            except Exception as e:
                self.get_logger().warn(f"Failed to load config file: {e}, using parameters")
        
        # 파라미터에서 토픽 설정 (config 파일 없거나 토픽 설정이 없을 때)
        if not self.topic_configs:
            output_suffix = self.get_parameter('output_suffix').value
            
            rgb_topics = self.get_parameter('rgb_topics').value
            for t in rgb_topics:
                self.topic_configs.append(TopicConfig(
                    input_topic=t,
                    output_topic=t + output_suffix,
                    image_type='rgb',
                    noise_config=self.default_noise_config
                ))
                
            depth_topics = self.get_parameter('depth_topics').value
            for t in depth_topics:
                self.topic_configs.append(TopicConfig(
                    input_topic=t,
                    output_topic=t + output_suffix,
                    image_type='depth',
                    noise_config=self.default_noise_config
                ))
    
    def setup_topics(self):
        """토픽 구독자/발행자 설정"""
        for config in self.topic_configs:
            # Publisher 생성 - Raw Image (RELIABLE QoS - 메시지 손실 방지)
            self._pubs[config.input_topic] = self.create_publisher(
                Image, 
                config.output_topic, 
                RAW_PUB_QOS
            )
            
            # Publisher 생성 - Compressed Image (RGB만, BEST_EFFORT - 실시간 스트리밍)
            if config.image_type == 'rgb':
                compressed_topic = config.output_topic + '/compressed'
                self._pubs[config.input_topic + '_compressed'] = self.create_publisher(
                    CompressedImage,
                    compressed_topic,
                    COMPRESSED_PUB_QOS
                )
            
            # Subscriber 생성 (BEST_EFFORT QoS - 센서 데이터 수신)
            self._subs[config.input_topic] = self.create_subscription(
                Image,
                config.input_topic,
                lambda msg, cfg=config: self.image_callback(msg, cfg),
                SENSOR_SUB_QOS
            )
            
            if config.image_type == 'rgb':
                self.get_logger().info(
                    f"  {config.input_topic} -> {config.output_topic} + /compressed ({config.image_type})"
                )
            else:
                self.get_logger().info(
                    f"  {config.input_topic} -> {config.output_topic} ({config.image_type})"
                )
    
    def image_callback(self, msg: Image, config: TopicConfig):
        """이미지 콜백 - 노이즈 적용 후 발행"""
        import time
        start_time = time.time()
        
        if not config.noise_config.enabled:
            # 노이즈 비활성화 시 원본 그대로 발행
            self._pubs[config.input_topic].publish(msg)
            return
        
        try:
            # ROS Image -> NumPy array (zero-copy when possible)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            if config.image_type == 'rgb':
                noisy_image = self.apply_rgb_noise_fast(cv_image, config.noise_config, config.input_topic)
            else:  # depth
                noisy_image = self.apply_depth_noise_fast(cv_image, config.noise_config, config.input_topic)
            
            noisy_msg = self.bridge.cv2_to_imgmsg(noisy_image, encoding=msg.encoding)
            
            # 타임스탬프 유지
            noisy_msg.header = msg.header
            
            # Raw Image 발행
            self._pubs[config.input_topic].publish(noisy_msg)
            
            # Compressed Image 발행 (RGB만)
            if config.image_type == 'rgb':
                compressed_key = config.input_topic + '_compressed'
                if compressed_key in self._pubs:
                    # BGR로 변환 (cv2.imencode 용)
                    if len(noisy_image.shape) == 3:
                        image_for_encode = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR) if noisy_image.shape[2] == 3 else noisy_image
                    else:
                        image_for_encode = noisy_image
                    
                    # JPEG 압축
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    _, compressed_data = cv2.imencode('.jpg', image_for_encode, encode_param)
                    
                    compressed_msg = CompressedImage()
                    compressed_msg.header = msg.header
                    compressed_msg.format = 'jpeg'
                    compressed_msg.data = compressed_data.tobytes()
                    self._pubs[compressed_key].publish(compressed_msg)
            
            # 성능 측정
            process_time = time.time() - start_time
            topic = config.input_topic
            if topic not in self._process_times:
                self._process_times[topic] = []
                self._frame_counts[topic] = 0
            self._process_times[topic].append(process_time)
            self._frame_counts[topic] += 1
            
            # 5초마다 로깅
            now = self.get_clock().now()
            if (now - self._last_log_time).nanoseconds > 5e9:
                for t, times in self._process_times.items():
                    if times:
                        avg_ms = sum(times) / len(times) * 1000
                        fps = self._frame_counts[t] / 5.0
                        self.get_logger().info(
                            f'[{t}] Process: {avg_ms:.1f}ms, Output: {fps:.1f} FPS'
                        )
                self._process_times = {t: [] for t in self._process_times}
                self._frame_counts = {t: 0 for t in self._frame_counts}
                self._last_log_time = now
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
    
    def apply_rgb_noise_fast(self, image: np.ndarray, config: NoiseConfig, topic: str) -> np.ndarray:
        """RGB 이미지에 노이즈 적용 (최적화 버전)"""
        
        # 물리 기반 노이즈 모델 사용 시
        if config.physical_noise_enabled:
            return self.apply_physical_noise(image, config, topic)
        
        # 기존 단순 노이즈 모델
        # int16으로 작업 (float32보다 2배 빠름, overflow 방지)
        noisy = image.astype(np.int16)
        
        # 1. Gaussian Noise (매 프레임 랜덤) - 정수 노이즈로 최적화
        if config.gaussian_enabled and config.gaussian_std > 0:
            # 미리 계산된 노이즈 버퍼 재사용 (첫 프레임에서 shape 캐시)
            noise_key = f"{topic}_gaussian"
            if noise_key not in self._noise_buffers:
                self._noise_buffers[noise_key] = image.shape
            
            # int16 노이즈 생성 (float보다 빠름)
            gaussian = np.random.randint(
                int(-config.gaussian_std * 3),
                int(config.gaussian_std * 3) + 1,
                size=image.shape,
                dtype=np.int16
            )
            noisy += gaussian
        
        # 2. Salt & Pepper Noise (고정 위치)
        if config.salt_pepper_enabled and config.salt_pepper_prob > 0:
            if topic not in self._salt_masks:
                self._salt_masks[topic] = np.random.random(image.shape[:2]) < config.salt_pepper_prob / 2
                self._pepper_masks[topic] = np.random.random(image.shape[:2]) < config.salt_pepper_prob / 2
                self.get_logger().info(
                    f"[{topic}] Created fixed salt/pepper masks: "
                    f"salt={np.sum(self._salt_masks[topic])}, pepper={np.sum(self._pepper_masks[topic])}"
                )
            noisy[self._salt_masks[topic]] = 255
            noisy[self._pepper_masks[topic]] = 0
        
        # 3. Exposure Variation (매 프레임 랜덤)
        if config.exposure_enabled and config.exposure_variation > 0:
            factor = 1.0 + np.random.uniform(-config.exposure_variation, config.exposure_variation)
            noisy = (noisy * factor).astype(np.int16)
        
        # Clip and convert back (in-place where possible)
        np.clip(noisy, 0, 255, out=noisy)
        return noisy.astype(np.uint8)
    
    def apply_physical_noise(self, image: np.ndarray, config: NoiseConfig, topic: str) -> np.ndarray:
        """
        물리 기반 센서 노이즈 모델 (Moseley et al. CVPR 2021)
        
        달 극지방 영구 그림자 영역(PSR)의 극저조도 환경을 시뮬레이션합니다.
        
        노이즈 모델 구성 (물리적 순서대로):
        
        1. PRNU (Photo Response Non-Uniformity) - **곱셈적** 노이즈
           - 원인: 포토다이오드 감도 차이 (제조 공정 편차)
           - 수식: y = x × (1 + δ_prnu), δ_prnu ~ N(0, σ_prnu)
           - 특성: 밝은 픽셀에서 더 큰 절대 오차 발생
           
        2. Shot Noise - 포아송 분포 (광자 노이즈)
           - 원인: 광자 도착의 양자역학적 불확실성
           - 수식: n ~ Poisson(λ), σ = √λ
           - 특성: 어두운 픽셀에서 상대적 노이즈 큼 (SNR = √signal)
        
        3. Dark Current - 온도 의존적 암전류
           - 원인: 열적 전자 생성
           - 수식: I_d = r × t × 2^((T-293)/10), Arrhenius 법칙
           
        4. Read Noise - 전자 회로 가우시안 노이즈
           - 원인: ADC, 앰프 등 전자회로 노이즈
           - 수식: n ~ N(0, σ_read)
           
        5. FPN (Fixed Pattern Noise) - **덧셈적** 노이즈
           - 원인: Dark current의 픽셀별 차이 + Readout 회로 오프셋
           - 수식: y = x + δ_fpn, δ_fpn ~ N(0, σ_fpn)
           - 특성: 어두운 영역에서 더 눈에 띔 (신호와 무관한 고정 오프셋)
           
        6. ADC Quantization - 비트 깊이에 따른 양자화
        
        ※ PRNU vs FPN: 물리적으로 다른 노이즈!
           - PRNU: 곱셈적 (밝을수록 영향 큼)
           - FPN: 덧셈적 (신호와 무관, 어두울수록 상대적 영향 큼)
        """
        h, w = image.shape[:2]
        is_color = len(image.shape) == 3
        
        # 정규화된 이미지 (0-1 범위로 변환)
        img_normalized = image.astype(np.float32) / 255.0
        
        # === 1. PRNU (Photo Response Non-Uniformity) - 픽셀 감도 편차 ===
        # 첫 프레임에서 생성 후 고정 (센서 제조 결함 시뮬레이션)
        prnu_key = f"{topic}_prnu"
        if config.prnu_enabled:
            if prnu_key not in self._noise_buffers:
                # 각 픽셀의 상대 감도 (평균 1.0, 표준편차 = prnu_strength)
                if is_color:
                    self._noise_buffers[prnu_key] = np.random.normal(
                        1.0, config.prnu_strength, (h, w, 1)
                    ).astype(np.float32)
                else:
                    self._noise_buffers[prnu_key] = np.random.normal(
                        1.0, config.prnu_strength, (h, w)
                    ).astype(np.float32)
                self.get_logger().info(f"[{topic}] Created PRNU map (σ={config.prnu_strength})")
            
            img_normalized = img_normalized * self._noise_buffers[prnu_key]
        
        # === 2. 광자 수로 변환 (전자 단위) ===
        # 신호 전자 수 = 정규화된 신호 * 풀웰 용량 * 양자 효율
        signal_electrons = img_normalized * config.full_well_capacity * config.quantum_efficiency
        
        # === 3. Shot Noise (포아송 분포) ===
        # 광자 도착의 통계적 변동 - 포아송 분포의 특성:
        # - 노이즈 표준편차 σ = √(signal)
        # - SNR = signal / σ = √(signal)
        # - 따라서 어두운 픽셀(signal 작음)일수록 상대적 노이즈가 커짐!
        # - 예: signal=100 → SNR=10 (10% 노이즈), signal=4 → SNR=2 (50% 노이즈)
        if config.shot_noise_enabled:
            # 포아송 분포에서 샘플링 (신호가 큰 경우 가우시안 근사)
            # np.random.poisson은 음수를 처리하지 못하므로 클램핑
            signal_electrons_clamped = np.maximum(signal_electrons, 0)
            
            # 최적화: 큰 값에서는 가우시안 근사 사용 (더 빠름)
            # 포아송(λ) ≈ N(λ, √λ) when λ > 100
            shot_noise = np.where(
                signal_electrons_clamped > 100,
                np.random.normal(0, np.sqrt(np.maximum(signal_electrons_clamped, 1))),
                np.random.poisson(signal_electrons_clamped).astype(np.float32) - signal_electrons_clamped
            )
            signal_electrons = signal_electrons + shot_noise
        
        # === 4. Dark Current (암전류) - 실시간 온도 반영 ===
        # 온도에 따른 열적 전자 생성 (Arrhenius 법칙)
        # - 온도가 10°C 상승할 때마다 암전류가 약 2배 증가
        # - 달 표면: 낮 ~127°C, 밤 ~-173°C, PSR ~-230°C
        if config.dark_current_enabled:
            # 실시간 온도 사용 (토픽에서 업데이트됨)
            current_temp = self._current_sensor_temp
            
            # 암전류 = 기본율 * 노출시간 * 온도 계수
            # 기준 온도: 293K (20°C)
            temp_factor = 2.0 ** ((current_temp - 293.0) / 10.0)
            dark_electrons = config.dark_current_rate * config.exposure_time * temp_factor
            
            # 암전류도 포아송 분포를 따름
            if dark_electrons > 0:
                dark_noise = np.random.poisson(dark_electrons, image.shape).astype(np.float32)
                signal_electrons = signal_electrons + dark_noise
        
        # === 5. Read Noise (읽기 노이즈) ===
        # 전자 회로에서 발생하는 가우시안 노이즈
        read_noise = np.random.normal(0, config.read_noise_std, image.shape).astype(np.float32)
        signal_electrons = signal_electrons + read_noise
        
        # === 6. Fixed Pattern Noise (FPN) - 고정 패턴 노이즈 ===
        # 픽셀별 오프셋 (제조 결함)
        fpn_key = f"{topic}_fpn"
        if config.fpn_enabled:
            if fpn_key not in self._noise_buffers:
                # 고정 오프셋 (전자 단위)
                fpn_amplitude = config.fpn_strength * config.full_well_capacity
                if is_color:
                    self._noise_buffers[fpn_key] = np.random.normal(
                        0, fpn_amplitude, (h, w, 1)
                    ).astype(np.float32)
                else:
                    self._noise_buffers[fpn_key] = np.random.normal(
                        0, fpn_amplitude, (h, w)
                    ).astype(np.float32)
                self.get_logger().info(f"[{topic}] Created FPN map (σ={config.fpn_strength*100:.1f}%)")
            
            signal_electrons = signal_electrons + self._noise_buffers[fpn_key]
        
        # === 7. 아날로그 게인 적용 ===
        signal_electrons = signal_electrons * config.analog_gain
        
        # === 8. ADC 변환 (양자화) ===
        # 전자 → 디지털 값 변환
        max_digital = (2 ** config.bit_depth) - 1
        digital_signal = signal_electrons / config.full_well_capacity * max_digital
        
        # 양자화 (정수로 반올림)
        digital_signal = np.round(digital_signal)
        
        # === 9. 클리핑 및 출력 ===
        digital_signal = np.clip(digital_signal, 0, max_digital)
        
        # 8비트 출력으로 변환
        if config.bit_depth != 8:
            digital_signal = digital_signal / max_digital * 255
        
        return digital_signal.astype(np.uint8)
    
    def apply_depth_noise_fast(self, image: np.ndarray, config: NoiseConfig, topic: str) -> np.ndarray:
        """Depth 이미지에 노이즈 적용 (최적화 버전)"""
        is_uint16 = image.dtype == np.uint16
        
        if is_uint16:
            noisy = image.astype(np.int32)  # overflow 방지
            scale = 1000.0
        else:
            noisy = image.copy()
            scale = 1.0
        
        # 1. Gaussian Noise (단순화 - 균일 노이즈로 대체, 더 빠름)
        if config.depth_gaussian_std > 0:
            noise_mm = int(config.depth_gaussian_std * scale * 3)  # 3-sigma
            if noise_mm > 0:
                gaussian = np.random.randint(-noise_mm, noise_mm + 1, size=image.shape, dtype=np.int32)
                noisy += gaussian
        
        # 2. Depth Dropout (고정 위치)
        if config.depth_dropout_prob > 0:
            if topic not in self._depth_dropout_masks:
                self._depth_dropout_masks[topic] = np.random.random(image.shape) < config.depth_dropout_prob
                self.get_logger().info(f"[{topic}] Created fixed depth dropout mask")
            noisy[self._depth_dropout_masks[topic]] = 0
        
        # Clip and convert back
        if is_uint16:
            np.clip(noisy, 0, 65535, out=noisy)
            return noisy.astype(np.uint16)
        else:
            np.clip(noisy, 0, None, out=noisy)
            return noisy.astype(np.float32)
    
    # Legacy methods (kept for compatibility)
    def apply_rgb_noise(self, image: np.ndarray, config: NoiseConfig, topic: str) -> np.ndarray:
        """RGB 이미지에 노이즈 적용 (레거시 - apply_rgb_noise_fast 사용 권장)"""
        return self.apply_rgb_noise_fast(image, config, topic)
    
    def apply_depth_noise(self, image: np.ndarray, config: NoiseConfig, topic: str) -> np.ndarray:
        """Depth 이미지에 노이즈 적용 (레거시 - apply_depth_noise_fast 사용 권장)"""
        return self.apply_depth_noise_fast(image, config, topic)


def main(args=None):
    rclpy.init(args=args)
    
    node = CameraNoiseNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
