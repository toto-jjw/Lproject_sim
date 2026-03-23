# Lproject_sim — Technical Details

> 전체 시스템의 상세 설계 문서. 각 모듈의 내부 구조, 파라미터, 데이터 흐름을 기술합니다.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Main Execution System](#main-execution-system)
- [Core Systems](#core-systems)
  - [SimulationManager](#1-simulationmanager)
  - [SceneManager](#21-scenemanager)
  - [StellarManager](#22-stellarmanager)
  - [EnergyManager](#3-energymanager)
  - [ROSManager](#4-rosmanager)
  - [ThermalManager](#5-thermalmanager)
  - [RobotContext](#6-robotcontext)
- [Configuration](#configuration)
  - [simulation_config.yaml](#1-simulation_configyaml)
  - [physics_config.py](#2-physics_configpy)
- [Terrain Generation](#terrain-generation)
- [Material: LunarRegolith8k.mdl](#material-lunarregolith8kmdl)
- [Rendering Effects](#rendering-effects)
- [Advanced Physics](#advanced-physics)
- [Dust System](#dust-system)
- [SDG System](#sdg-system)
- [Rover System](#rover-system)
  - [USD & OmniGraph](#1-usd--omnigraph)
  - [rover.py](#2-roverpy)
  - [solar_panel.py](#3-solar_panelpy)
- [Sun Sensor](#sun-sensor)
- [HUD System](#hud-system)
- [ROS 2 Custom Nodes](#ros-2-custom-nodes)
  - [solar_control_node.py](#1-solar_control_nodepy)
  - [noise_node.py](#2-noise_nodepy)
  - [denoise_node.py (NAFNet)](#3-denoise_nodepy--nafnet)
  - [enhance_node.py (DimCam)](#4-enhance_nodepy--dimcam)
- [Navigation: LiDAR + Nav2](#navigation-lidar--nav2)
- [Navigation: Stereo SLAM Stack](#navigation-stereo-slam-stack)
- [References & Roadmap](#references--roadmap)

---

## Project Structure

```
Lproject_sim/
├── assets/                  # 3D 모델, 텍스처, 지형 등 시뮬레이션 환경 구성을 위한 에셋
├── config/                  # Nav2, Nvblox 및 시뮬레이션 구동을 위한 YAML 설정 파일
├── data/                    # 시뮬레이션 결과 데이터, DEM 출력물 및 분석용 Jupyter Notebook
├── isaac_ros_nvblox/        # 3D 환경 매핑(Nvblox) 및 V-SLAM용 NVIDIA Isaac ROS 패키지
├── launch/                  # Nav2 + 시뮬레이션 노드 통합 실행용 ROS 2 Launch 파일
├── Lproject_cam/            # 카메라 이미지 품질 향상용 딥러닝 모델 및 가중치
├── NAFNet/                  # 이미지 노이즈 제거 및 복원용 NAFNet 비전 모델
├── navigation2/             # 로봇 자율 주행 및 경로 계획 담당 ROS 2 Nav2 프레임워크
├── scripts/                 # 시뮬레이션 운영을 돕는 유틸리티 스크립트
└── src/                     # 시뮬레이션 핵심 로직 소스 코드
    ├── config/              #   YAML 설정 로드 및 파라미터 관리
    ├── core/                #   메인 루프, 씬 관리 등 핵심 로직
    ├── environment/         #   조명, 하늘, 먼지 효과 등 배경 환경
    ├── nodes/               #   ROS 2 통신 및 이미지 처리 노드
    ├── physics/             #   토양역학, 지형 변형 등 커스텀 물리 로직
    ├── rendering/           #   카메라 렌더링 파이프라인 및 후처리 품질 관리
    ├── robots/              #   로봇 모델링, 기구학, 배터리/열 모델 및 제어기
    ├── sdg/                 #   AI 학습용 합성 데이터 생성 및 자동 라벨링
    ├── sensors/             #   센서 모델링 및 ROS 2 데이터 퍼블리싱
    ├── terrain/             #   DEM 로드, 절차적 지형 생성, 크레이터 모델링
    └── ui/                  #   배터리, 속도 등 상태 모니터링 HUD
```

---

## Main Execution System

### main.py — 시뮬레이션 시작점

Isaac Sim 엔진을 올바른 설정으로 구동하는 진입점.

| 부분 | 역할 |
|------|------|
| `CONFIG` | Isaac Sim 앱의 핵심 동작 정의 (헤드리스, 렌더러, 확장 기능) |
| `SimulationApp(CONFIG)` | CONFIG에 따라 Isaac Sim 엔진 초기화 및 시작 |
| 지연된 임포트 | SimulationApp 시작 후 `simulation_runner` 등 프로젝트 모듈 로드 |
| `main()` | `rclpy` 초기화 후 `SimulationRunner` 객체 생성 및 `run()` 호출 |
| `finally` | `simulation_app.close()` + `rclpy.shutdown()` 으로 모든 리소스 정리 |

---

### simulation_runner.py — 시뮬레이션 실행 및 관리

설정 파일을 기반으로 모든 매니저 객체를 생성하고 업데이트 루프를 실행.

#### `__init__()` — 초기화 단계

| 순서 | 작업 |
|------|------|
| 1 | **설정 로드**: `ConfigLoader`로 `config.yaml` 읽기 |
| 2 | **환경 생성**: `SimulationManager`(월드), `SceneManager`(조명) |
| 3 | **렌더링 설정**: `RenderingManager`로 렌즈 플레어 등 시각 효과 적용 |
| 4 | **지형 생성**: `TerrainManager`로 지형 메시 및 바위 생성 |
| 5 | **물리 준비**: `PhysicsManager`로 지반 역학 및 먼지 효과 준비 |
| 6 | **로봇 생성**: `robots` 설정에 따라 `RobotContext` 객체 생성 |
| 7 | **SDG 준비**: `SDGRandomizer`, `Annotator` 준비 |
| 8 | **UI 생성**: HUD 생성 |
| 9 | **물리 안정화**: `step(render=False)` 60회 반복으로 물리 객체 안착 |

#### `run()` — 실행 루프

```python
while self.sim.is_running():
    if self.sim.is_playing():
        # 1. 입력 처리
        input_manager.update()

        # 2. 로봇 업데이트 (센서, 에너지, ROS 발행)
        for robot in self.robots:
            robot.update(...)

        # 3. Isaac Sim 물리 / 렌더링 스텝
        self.sim.step()

    if self.sim.is_playing():
        # 4. 물리 효과 업데이트 (바퀴 자국, 먼지)
        self.physics_manager.update(...)

        # 5. UI 업데이트
        self.hud.update(...)
```

#### `_cleanup()` — 정리 단계

시뮬레이션 루프 종료 후 HUD, InputManager, ROSManager 등 모든 객체의 리소스를 안전하게 해제.

---

## Core Systems

`src/core` 디렉토리는 시뮬레이션의 기본 환경, 로봇 객체, 외부 시스템 상호작용을 관리하는 핵심 클래스들을 포함. `simulation_runner.py`에 의해 조립되어 전체 시뮬레이션의 흐름을 구성.

### 1. SimulationManager

**역할**: Isaac Sim의 `World` 객체를 제어하여 시뮬레이션 시간 흐름을 관리.

| 메서드 | 기능 |
|--------|------|
| `step()` | 물리 계산과 렌더링을 한 프레임 진행 |
| `is_playing()` | 사용자가 'Play' 버튼을 눌렀는지 확인하여 Pause 기능 구현 |

---

### 2.1. SceneManager

**역할**: 시뮬레이션 월드의 조명 등 정적 환경 요소 설정.

| 메서드 | 기능 |
|--------|------|
| `setup_lighting()` | `config.yaml`의 `scene.sun_light` 설정을 읽어 태양광(`DistantLight`) 생성 |

- `intensity`, `color_temperature`로 조명 품질 설정
- `elevation`, `azimuth` 값으로 태양 방향을 정밀 제어하여 그림자 생성

---

### 2.2. StellarManager

NASA의 JPL Ephemeris 데이터를 기반으로 달 표면 특정 좌표에서의 태양·지구 위치(고도, 방위각)를 계산하고 Isaac Sim의 조명에 실시간 반영.

#### StellarManager — 천문 계산 엔진

- **Ephemeris 데이터 활용**: `de421.bsp` + 달 좌표계 데이터(`moon_*.tf/tpc`) 로드
- **관측자 설정**: 달 표면 위도/경도에 관측자를 배치하여 겉보기 위치(Apparent Position) 계산
- **시간 관리**: UTC 기준, `time_scale`로 시간 가속(예: 1초 = 1시간) 가능

#### StellarSceneUpdater — USD 씬 동기화

천문 데이터를 Isaac Sim USD 스테이지에 시각적으로 반영하는 연결 모듈.

| 기능 | 내용 |
|------|------|
| 태양 조명 제어 | `/World/Sun` 경로의 `DistantLight`를 찾아 제어 |
| 좌표 변환 | 고도(Altitude), 방위각(Azimuth) → USD Euler Angles 변환 (`RotateZ`: 방위각, `RotateX`: 90 - 고도) |
| 최적화 | `update_interval`에 따라 주기적으로 갱신하여 연산 부하 최소화 |

#### 데이터 흐름

```
Update  →  시뮬레이션 dt × time_scale 만큼 시간 전진
Calculate →  현재 시간/위치 기준 태양의 고도/방위각 계산 (Skyfield)
Apply   →  계산된 각도를 USD Sun Light의 Transform에 적용
Result  →  그림자 길이/방향, 낮/밤 변화가 실제 달 환경과 동일하게 구현
```

#### 주요 파라미터 (StellarConfig)

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `latitude` / `longitude` | 달 표면 관측자의 위도/경도 | -26.3°, 46.8° |
| `start_time` | 시뮬레이션 시작 시간 (UTC) | 2024-05-01 12:00:00 |
| `time_scale` | 시간 배속 (1.0 = 실시간, 3600.0 = 1시간/초) | 1.0 |
| `update_interval` | 천체 위치 재계산 간격 (초) | 1.0 |
| `auto_update` | 매 프레임 자동 갱신 여부 | `True` |

---

### 3. EnergyManager

로버의 배터리 충/방전, 태양광 발전, 전력 소비를 물리적 수식 기반으로 시뮬레이션. **ThermalModel**과 연동하여 극한의 달 환경 온도에 따른 배터리 효율 저하를 반영.

#### 태양광 발전 모델

```
P_solar = I × A × η × cos(θ) × dust_factor
```

| 변수 | 설명 | 값 |
|------|------|-----|
| `I` | 태양 상수 | 1361 W/m² |
| `A` | 패널 면적 | 0.25 m² |
| `η` | 패널 효율 | 20% |
| `cos(θ)` | 태양 입사각 | 실시간 계산 |
| `dust_factor` | 먼지에 의한 효율 저하 | 기본 1.0 |

> `SunSensor`로부터 태양 벡터를 받아 입사각을 실시간 계산. 그림자 진입 시 발전량 = 0.

#### 전력 소비 모델

```
P_load = P_base + k × v
```

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `P_base` | 기본 소비 전력 (통신, 컴퓨터 등) | 10 W |
| `k` | 모터 부하 계수 | 5 W per m/s |
| `v` | 로버 이동 속도 | — |

#### 온도에 따른 배터리 효율

```
η_batt = max(0.1, 1 - 0.01 × |T - T_opt|)
```

저온(-173°C)이나 고온(+127°C) 환경에서는 배터리 효율이 급격히 저하. 온도 데이터는 `ThermalModel`에서 `set_temperature()` 메서드를 통해 주입.

#### 주요 설정 (EnergyConfig)

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `capacity_wh` | 배터리 총 용량 | 100.0 Wh |
| `initial_charge_wh` | 초기 충전량 | 50.0 Wh |
| `solar_efficiency` | 태양광 패널 효율 | 0.20 (20%) |
| `base_load_w` | 대기 전력 소비량 | 10.0 W |
| `motor_load_factor` | 속도에 따른 추가 전력 소비 계수 | 5.0 |

---

### 4. ROSManager

Isaac Sim 내부 Python 환경에서 직접 실행되는 **네이티브 ROS 2 노드**. 커스텀 시뮬레이션 로직 데이터(배터리 상태, 태양 벡터, 센서 온도 등)를 ROS 2 네트워크와 연결하는 브리지.

#### 발행 토픽 (Publishers)

| 토픽 | 메시지 타입 | 설명 |
|------|------------|------|
| `/{robot}/sun_vector` | `geometry_msgs/Vector3` | 태양 센서가 감지한 태양 방향 벡터 (패널 제어에 사용) |
| `/{robot}/battery_state` | `sensor_msgs/BatteryState` | EnergyManager가 계산한 배터리 전압, 잔량(%), 전류 상태 |
| `/rover/sensor_temperature` | `std_msgs/Float64` | 로버 센서 온도 (°C → K 변환). `camera_noise_node`가 구독하여 Dark Current 계산 |
| `/tf_gt` | `tf2_msgs/TFMessage` | Ground Truth 절대 위치 (`map → base_link`). SLAM 정확도 평가용으로 별도 토픽 분리 |

#### 수신 토픽 (Subscribers)

| 토픽 | 설명 |
|------|------|
| `/{robot}/solar_panel/cmd_angle` (`Float32`) | 태양광 패널 회전 각도 명령 수신 |

> **참고**: 로버의 주행 명령 `cmd_vel`은 Isaac Sim의 OmniGraph 노드에서 직접 처리하여 지연 시간을 최소화.

#### 좌표계 변환

| 항목 | Isaac Sim (Core) | ROS 2 |
|------|-----------------|--------|
| 쿼터니언 순서 | `(w, x, y, z)` | `(x, y, z, w)` |

`publish_map_to_base_tf` 메서드에서 순서를 수동 재배열하여 호환성 보장.

#### TF 충돌 방지

일반적인 `/tf` 대신 `/tf_gt`를 사용하는 이유: SLAM 노드가 추정하여 발행하는 `/tf`와 시뮬레이터의 정답 `/tf`가 충돌하여 로봇 위치가 떨리는 현상 방지.

---

### 5. ThermalManager

물리 기반 **6면체 박스 모델(Six-faced box model)**로 로버의 열 역학을 시뮬레이션. 태양광 입사각과 로버 자세에 따른 각 면의 온도 변화를 계산하고 평균하여 내부 온도 도출.

#### 주요 특징

- **달 환경 온도 범위**: 주간 적도 +127°C ~ 야간/그림자 -173°C
- **열 관성 (Thermal Inertia)**: 시정수(Time Constant)와 시그모이드 곡선을 따라 목표 온도에 서서히 도달
- **노이즈 시뮬레이션**: 센서 오차를 위한 가우시안 노이즈 추가 가능

#### 주요 계산 단계

| 단계 | 내용 |
|------|------|
| 태양 위치 및 자세 업데이트 | 로버 위치/Yaw, 태양 방향 벡터로 상대 위치 계산 |
| 뷰 팩터(View Factor) 계산 | 각 면 법선 벡터와 태양 벡터의 내적(dot product)으로 태양광 수신 비율(0~1) 계산 |
| 목표 온도 설정 | 뷰 팩터를 시그모이드 함수에 대입하여 목표 온도 설정 (그림자 시 모든 면 -173°C 수렴) |
| 온도 전이 | 현재 온도와 목표 온도 차이에 시정수 적용하여 스텝별 변화량 계산 |
| 내부 온도 도출 | 6개 외부 면의 온도 단순 평균으로 `interior` 온도 갱신 |

#### 설정 모드

| 모드 | 설명 |
|------|------|
| 동적 (`enabled=True`) | 태양 위치와 로버 움직임에 따라 실시간 온도 변화 |
| 정적 (`enabled=False`) | `static_temperature`로 설정된 고정값 유지 (테스트용) |

#### 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `min_temp` | 달 표면 최저 온도 (야간/그림자) | -173.0°C |
| `max_temp` | 달 표면 최고 온도 (주간 직사광선) | 127.0°C |
| `time_constant` | 목표 온도 도달 속도 (열 관성 계수) | 600.0초 |
| `sigmoid_gain` | 태양광 노출에 따른 온도 변화 민감도 | 8.0 |

---

### 6. RobotContext

단일 로봇 인스턴스를 관리하는 최상위 컨테이너. 물리적 로봇(`Rover`), 센서(`SunSensor`), 논리 컴포넌트(`EnergyManager`, `ThermalModel`), 통신 인터페이스(`ROSManager`)를 하나로 묶어 상호작용.

#### 초기화 단계

| 항목 | 내용 |
|------|------|
| 지형 스냅 (Terrain Snap) | `terrain_manager`로 초기 위치(x, y)의 지형 높이(z) 계산 → 로봇을 지면에 정확히 안착 |
| 물리 파라미터 로드 | 마찰력, 질량 등 물리 속성 설정 |
| 센서 초기화 | `SunSensor` 등 커스텀 센서 생성 |
| 컴포넌트 초기화 | `SolarPanel`, `EnergyManager`, `ThermalModel` 생성 |
| ROS 2 초기화 | `ROSManager` 생성으로 외부 통신 채널 개방 |
| Latency 설정 | 통신 지연 시뮬레이션을 위한 `LatencyManager` 설정 |

#### 업데이트 루프 데이터 파이프라인

```
1. ROS 통신 처리  → ros2.spin_once(): 외부 명령(패널 회전 등) 수신
2. 환경 감지      → SunSensor: 태양 벡터 획득 + 레이캐스팅으로 그림자 판별
3. 열 모델링      → ThermalModel: 1Hz 간격으로 내부 온도 계산
4. 에너지 관리    → EnergyManager: 온도/그림자 계수/먼지 효율 주입 후 배터리 갱신
5. 데이터 발행    → ROSManager: 위치/태양 벡터/배터리/온도를 ROS 2 토픽으로 발행
```

#### 서브시스템 간 상호작용

| 데이터 소스 | → 데이터 목적지 | 목적 |
|------------|----------------|------|
| `SunSensor` | → `ThermalModel` | 그림자 여부에 따른 급격한 온도 변화 계산 |
| `SunSensor` | → `EnergyManager` | 태양 입사각 및 그림자에 따른 발전량 계산 |
| `ThermalModel` | → `EnergyManager` | 온도 변화에 따른 배터리 효율 저하 반영 |
| `ROSManager` | → `SolarPanel` | 사용자 명령에 따라 패널 각도 제어 |
| `EnergyManager` | → `ROSManager` | 배터리 상태(전압, %)를 HUD 및 외부로 송출 |

---

## Configuration

### 1. simulation_config.yaml

코드를 직접 수정하지 않고 시뮬레이션 환경, 로봇 동작, 물리 현상을 변경하는 중앙 설정 파일.

#### 1. Simulation (기본 환경)

| 파라미터 경로 | 타입 | 기본값 | 설명 |
|-------------|------|--------|------|
| `simulation.headless` | Bool | `false` | GUI 표시 여부 (True: 렌더링 없음, 서버용) |
| `simulation.dt` | Float | `0.033` | 물리 타임스텝 (30Hz) |
| `simulation.ros_bridge_type` | String | `"omnigraph"` | ROS 2 브리지 방식 (`omnigraph` / `native`) |
| `simulation.renderer.lens_flare.enabled` | Bool | `false` | 렌즈 플레어 효과 |
| `simulation.renderer.lens_flare.scale` | Float | `2.0` | 렌즈 플레어 크기 배율 |
| `simulation.renderer.lens_flare.blades` | Int | `5` | 조리개 날개 수 |
| `simulation.renderer.motion_blur.enabled` | Bool | `false` | 모션 블러 효과 |
| `simulation.renderer.dlss.enabled` | Bool | `false` | NVIDIA DLSS 업스케일링 |
| `simulation.physics_scene.gravity` | List | `[0,0,-1.62]` | 중력 가속도 벡터 (달 중력) |

#### 2. Terrain (지형 생성)

| 파라미터 경로 | 타입 | 기본값 | 설명 |
|-------------|------|--------|------|
| `terrain.type` | String | `"procedural"` | 생성 모드 (`procedural` / `real_data` / `hybrid`) |
| `terrain.x_size`, `y_size` | Float | `100` | 메인 지형 크기 (m) |
| `terrain.resolution` | Float | `0.05` | 지형 해상도 (m/pixel) |
| `terrain.z_scale` | Float | `4.0` | 높이 배율 |
| `terrain.seed` | Int | `952` | 지형 생성 난수 시드 |
| `terrain.outer_terrain.enabled` | Bool | `true` | 외곽 지형 생성 활성화 |
| `terrain.outer_terrain.size_multiplier` | Float | `5.0` | 전체 크기 배율 (메인 대비) |
| `terrain.outer_terrain.rim_height` | Float | `20.0` | 외곽 산맥 높이 (m) |
| `terrain.outer_terrain.num_craters` | Int | `10` | 외곽 지역 대형 크레이터 수 |
| `terrain.num_rocks` | Int | `150` | 배치할 바위 개수 |

#### 3. Environment (환경 효과 및 상호작용)

| 파라미터 경로 | 타입 | 기본값 | 설명 |
|-------------|------|--------|------|
| `environment.terramechanics.enabled` | Bool | `false` | 테라메카닉스 물리 활성화 |
| `environment.dust.enabled` | Bool | `false` | 먼지 시뮬레이션 활성화 |
| `environment.hud.enabled` | Bool | `true` | HUD 표시 |
| `environment.nav2.enabled` | Bool | `true` | ROS 2 Nav2 연동 |
| `environment.deformation.enabled` | Bool | `false` | 바퀴 자국 생성 |
| `environment.deformation.use_track_renderer` | Bool | `true` | `true`: 리본 메시(빠름), `false`: GPU 변형(느림) |
| `environment.thermal_model.enabled` | Bool | `true` | 열 모델 활성화 |
| `environment.thermal_model.time_constant` | Float | `600.0` | 열적 시정수 |
| `environment.latency.enabled` | Bool | `false` | 통신 지연 활성화 |
| `environment.latency.delay_seconds` | Float | `1.3` | 지연 시간 (Earth-Moon 약 1.3초) |
| `environment.latency.dropout_rate` | Float | `0.05` | 패킷 손실률 |
| `environment.sdg.enabled` | Bool | `false` | 합성 데이터 생성 활성화 |

#### 4. Camera Noise Model

| 파라미터 경로 | 타입 | 기본값 | 설명 |
|-------------|------|--------|------|
| `camera_noise.enabled` | Bool | `true` | 노이즈 노드 활성화 |
| `camera_noise.output_resolution` | List | `[512, 512]` | 출력 해상도 |
| `camera_noise.physical.enabled` | Bool | `true` | 물리 기반 노이즈 활성화 |
| `camera_noise.physical.quantum_efficiency` | Float | `0.8` | 양자 효율 |
| `camera_noise.physical.full_well_capacity` | Int | `50000` | 풀웰 용량 |
| `camera_noise.physical.bit_depth` | Int | `12` | 비트 깊이 |
| `camera_noise.physical.read_noise.std` | Float | `5.0` | 읽기 노이즈 표준편차 |
| `camera_noise.physical.dark_current.rate` | Float | `0.01` | 암전류 발생률 |
| `camera_noise.physical.fpn.strength` | Float | `0.1` | FPN 강도 |
| `camera_noise.physical.prnu.strength` | Float | `0.05` | PRNU 강도 |

#### 5. Stellar (천체 엔진)

| 파라미터 경로 | 타입 | 기본값 | 설명 |
|-------------|------|--------|------|
| `stellar.enabled` | Bool | `false` | 천체 엔진 활성화 |
| `stellar.latitude` | Float | `-80.0` | 관측자 위도 |
| `stellar.longitude` | Float | `-45.0` | 관측자 경도 |
| `stellar.time_scale` | Float | `1.0` | 시간 배속 |
| `stellar.update_interval` | Float | `1.0` | 업데이트 간격 (초) |

#### 6. Scene (조명 및 배경)

| 파라미터 경로 | 타입 | 기본값 | 설명 |
|-------------|------|--------|------|
| `scene.sun_light.intensity_scale` | Float | `40.0` | 태양광 강도 |
| `scene.sun_light.color_temperature` | Float | `5778.0` | 태양 색온도 (K) |
| `scene.sun_light.elevation` | Float | `10.0` | 태양 고도 (Stellar 비활성 시) |
| `scene.sun_light.azimuth` | Float | `90.0` | 태양 방위각 |
| `scene.dome_light.enabled` | Bool | `true` | 돔 라이트(산란광) 활성화 |
| `scene.stars.enabled` | Bool | `true` | 별 배경 활성화 |
| `scene.stars.count` | Int | `1000` | 별 개수 |

#### 7. Robots

| 파라미터 경로 | 타입 | 기본값 | 설명 |
|-------------|------|--------|------|
| `robots[0].name` | String | `"Husky_1"` | 로봇 이름 |
| `robots[0].position` | List | `[0,0,2]` | 초기 위치 |
| `robots[0].terrain_snap` | Bool | `false` | 지형 높이 자동 스냅 |
| `robots[0].sensors.sun_sensor.enabled` | Bool | `true` | 태양 센서 활성화 |
| `robots[0].components.solar_panel.enabled` | Bool | `true` | 태양광 패널 활성화 |
| `robots[0].components.energy_manager.enabled` | Bool | `true` | 에너지 매니저 활성화 |

---

### 2. physics_config.py

`dataclasses`를 사용하여 로봇과 지형의 물리적 특성을 구조화된 데이터 형태로 정의. 코드 내 매직 넘버 사용 방지.

#### 로봇 파라미터 (RobotParameter)

| 파라미터 | 단위 | 기본값 | 설명 |
|----------|------|--------|------|
| `num_wheels` | — | 4 | 바퀴 개수 |
| `wheel_radius` | m | 0.165 | 바퀴 반경 (Husky 기준) |
| `wheel_width` | m | 0.1 | 바퀴 폭 |
| `wheel_base` | m | 0.5 | 축간 거리 |
| `mass` | kg | 50.0 | 로봇 총 질량 |
| `wheel_stiffness` | N/m | 100000.0 | 바퀴 강성 |
| `wheel_damping` | N·s/m | 10000.0 | 바퀴 감쇠 |
| `max_effort` | N·m | 100.0 | 최대 토크 |
| `max_velocity` | rad/s | 10.0 | 최대 각속도 |

#### 지형 역학 파라미터 (TerrainMechanicalParameter) — Bekker 모델

| 파라미터 | 단위 | 기본값 | 설명 |
|----------|------|--------|------|
| `k_c` | N/m^(n+1) | 1400.0 | 점착 계수 |
| `k_phi` | N/m^(n+2) | 820000.0 | 마찰 계수 |
| `n` | — | 1.0 | 침하 지수 |
| `c` | Pa | 170.0 | 점착력 (달 레골리스) |
| `phi` | rad | 0.5 | 내부 마찰각 (~29°) |
| `K` | m | 0.015 | 전단 변형 계수 |
| `rho` | kg/m³ | 1600.0 | 토양 밀도 (달 레골리스) |

---

## Terrain Generation

### 지형 생성 모드 (terrain_generator.py)

#### A. `real_data` 모드 — 실제 지형 재현

NASA LRO(Lunar Reconnaissance Orbiter) 등에서 획득한 실제 달 표면 고도 데이터(`.npy`)를 로드.

- **메모리 최적화 (Crop-then-Resize)**: 필요한 영역만 먼저 잘라낸(Crop) 후 목표 해상도로 리사이징하여 OOM 방지
- **좌표 오프셋**: `crop_center_meters` 설정으로 거대 지도 내 탐사 구역을 미터 단위로 지정 가능

#### B. `procedural` 모드 — 무작위 지형 생성

수학적 노이즈 알고리즘으로 자연스러운 달 지형 생성.

| 기법 | 설명 |
|------|------|
| **Perlin-like Noise** | 여러 옥타브 노이즈를 중첩하여 큰 언덕과 작은 요철 동시 표현 |
| **Smoothing** | `cv2.GaussianBlur`로 로버가 주행 가능한 부드러운 경사면 생성 |
| **Crater Injection** | `CraterGenerator`로 생성된 크레이터 높이맵을 지형에 합성 |

#### C. `hybrid` 모드

실제 지형의 거시적 형태(Low Frequency) + 절차적 노이즈의 미세한 디테일(High Frequency) 결합. 저해상도 DEM의 밋밋한 표면에 인위적인 거칠기를 추가하여 사실감 향상.

---

### 크레이터 생성기 (CraterGenerator)

| 기법 | 설명 |
|------|------|
| `CubicSpline` 변형 | 크레이터 가장자리를 불규칙하게 변형 |
| Marks & Noise | 크레이터 내/외부에 방사형 자국과 노이즈 추가로 오래된 지형 표현 |
| Profile | 사전 정의된 단면 프로파일(`profiles.pkl`)로 깊이와 경사 결정 |

---

### 지형 관리 (terrain_manager.py)

#### 지형 메쉬 생성

- **버텍스 생성**: 그리드 정점(Vertex)을 생성하고 Z축 높이를 DEM 값으로 설정
- **인덱싱**: 정점들을 연결하여 삼각형(Triangle) 면 구성
- **최적화**: `current_vertices_np` 변수에 전체 정점 데이터 캐싱, NumPy 벡터 연산으로 고속 업데이트
- **Y-Flip 보정**: 이미지 좌표계(Y-down) ↔ 3D 좌표계(Y-up) 차이 보정

#### 외곽 지형 (Outer Terrain)

- **구조**: 메인 지형(고해상도) 중심으로 4배 넓은 저해상도 지형이 둘러싸는 형태
- **블렌딩**: `_blend_with_main_terrain` 메서드로 경계를 부드럽게 연결
- **Mountain Rim**: 시야 차단과 고립된 달 환경 연출을 위해 가장자리에 높은 산맥 배치

#### 바위 분산 배치 (Rock Scattering)

- **USDZ 에셋 활용**: 지정 디렉토리의 바위 3D 모델 로드 후 인스턴싱
- **지형 적응**: `sample_height_at_xy`로 정확한 지형 높이 계산 → 공중 부양/파묻힘 방지
- **랜덤화**: 위치, 회전(Yaw), 크기(Scale) 무작위 설정
- **Semantic Labeling**: 각 바위에 `"rock"` 시맨틱 라벨 자동 적용 (SDG 세그멘테이션 활용)

#### 물리 속성

| 속성 | 값 | 설명 |
|------|-----|------|
| Static Friction | 0.7 | 달 레골리스 특성 모사 |
| Dynamic Friction | 0.6 | — |
| Contact Offset | 2cm | 고속 주행 시 지형 관통 방지 |
| Rest Offset | 5mm | — |
| Collision API | `approximation="none"` | 메쉬 형태 그대로의 정밀 충돌 (레이캐스팅 정확도 향상) |

---

## Material: LunarRegolith8k.mdl

MDL(Material Definition Language)로 달 표면 재질을 물리 법칙 기반으로 정의. NVIDIA 표준 셰이더 `OmniPBR` 기반.

### 주요 MDL 파라미터

| 카테고리 | 파라미터 | 값 | 설명 |
|---------|----------|-----|------|
| **기본 색상** | `diffuse_texture` | `*_diff.png` | 실제 색상과 패턴 핵심 텍스처 |
| **기본 색상** | `diffuse_color_constant` | `color(0.1, 0.1, 0.1)` | 텍스처 없을 때 기본 색상 (어두운 회색) |
| **거칠기** | `reflection_roughness_constant` | `1.0` | 극도로 거친 표면 (완전 무광, 흙/먼지 특성) |
| **거칠기** | `metallic_constant` | `0.0` | 금속 재질 아님 |
| **입체감** | `normalmap_texture` | `*_nor_gl.png` | 표면 미세 요철 표현 핵심 텍스처 |
| **입체감** | `bump_factor` | `0.8` | 노멀맵 효과 강도 (80%) |
| **음영** | `ao_texture` | `*_ao.png` | 빛이 덜 닿는 틈새 정보 |
| **음영** | `ao_to_diffuse` | `0.35` | AO 맵을 35% 강도로 기본 색상에 곱해 입체감 향상 |
| **타일링** | `texture_scale` | `float2(0.5)` | 텍스처를 2배 크게 표시하여 반복 느낌 감소 |

### 빠른 튜닝 가이드

| 원하는 효과 | 조정할 파라미터 | 변경 방향 |
|------------|---------------|-----------|
| 지형 패턴 크기 조절 | `Texture Scale` | 작게 ↓ / 크게 ↑ |
| 젖거나 얼어붙은 느낌 | `Reflection Roughness Constant` | 값 낮춤 ↓ |
| 표면 요철 강조 | `Bump Factor` | 값 높임 ↑ |

---

## Rendering Effects

`RenderingManager` 클래스는 Isaac Sim의 실시간 렌더링 품질과 관련된 시각 효과(Post-Processing)를 코드 레벨에서 제어. `carb.settings`를 사용하여 복잡한 경로를 간단한 함수로 추상화.

### 주요 기능

| 효과 | 메서드 | 설명 |
|------|--------|------|
| 렌즈 플레어 | `enable_lens_flare()`, `set_lens_flare_params()` | 밝은 광원의 렌즈 내부 빛 산란 시뮬레이션 |
| 모션 블러 | `enable_motion_blur()` | 빠른 움직임에 잔상 효과 추가 |
| DLSS | `enable_dlss()` | AI 기반 업스케일링으로 FPS 향상 (RTX GPU 전용) |

### 효과 적용 범위

| 효과 | 메인 뷰포트 | 로버 카메라(센서 데이터) | 비고 |
|------|:-----------:|:----------------------:|------|
| Lens Flare | ✅ | ✅ | 후처리 효과, 모든 렌더링 결과물 적용 |
| Motion Blur | ✅ | ✅ | 후처리 효과, 모든 렌더링 결과물 적용 |
| DLSS | ✅ | ❌ | 메인 뷰포트 성능 향상 기술, 센서 데이터에는 영향 없음 |

---

## Advanced Physics

세 모듈이 협력하여 월면 주행을 시뮬레이션.

| 모듈 | 역할 |
|------|------|
| `PhysicsManager` | 전체 물리 계산 조율, 모듈 간 데이터 전달 및 로봇/지형에 결과 적용 |
| `TerramechanicsSolver` | 바퀴-흙 상호작용(미끄러짐, 저항력)을 Wong & Bekker 모델로 수학적 해석 |
| `DeformationEngine` | NVIDIA Warp을 이용한 GPU 기반 실시간 바퀴 자국(지형 변형) 시뮬레이션 |

### PhysicsManager — 중앙 흐름 제어

`_process_robot` 메서드 내 처리 순서:

```
1. 상태 수집      → 로봇 좌표, 선속도, 각 바퀴의 회전 속도
2. 바퀴 상태 계산  → 로봇 중심 좌표/방향 기반 각 바퀴의 월드 좌표 계산
3. 침하량 계산    → sinkage = wheel_radius - 지형높이 공식
4. Terramechanics → 바퀴별 상태를 TerramechanicsSolver로 전달
5. 저항력 적용    → 저항 토크(My) 기반 주행 저항력(Drag Force) 계산 후 로봇에 적용
6. 지형 변형      → 수직항력(Fz)과 바퀴 위치를 DeformationEngine으로 전달
7. 먼지 발생      → 바퀴 속도/침하량이 임계값 초과 시 DustManager 호출
```

### TerramechanicsSolver — 지반 역학 해석

| 계산 단계 | 수식 |
|----------|------|
| **슬립률(Slip Ratio)** | `s = 1 - (v_actual / v_theoretical)` |
| **수직 응력 σ(θ)** | `(k_c/b + k_φ) × z(θ)^n` (Bekker 모델) |
| **전단 응력 τ(θ)** | `(c + σ(θ) × tan(φ)) × (1 - e^(-j(θ)/K))` (Janosi & Hanamoto 모델) |
| **최종 힘/토크** | 분포된 응력을 접촉 면적에 대해 수치 적분 (`np.trapz`) |

#### 출력값

| 출력 | 설명 |
|------|------|
| `Fx` (수평력) | 로버의 구동력/제동력 |
| `Fz` (수직력) | 지면이 바퀴를 떠받치는 힘 (지형 변형의 원인) |
| `My` (저항 토크) | 바퀴 회전을 방해하는 토크 (주행 저항의 근원) |

### DeformationEngine — GPU 기반 실시간 지형 변형

NVIDIA Warp을 사용하여 지형 변형 계산을 GPU에서 대규모 병렬 처리.

`deform_terrain_kernel` — 지형 그리드의 모든 픽셀에 대해 동시에 수행:

```
1. 자신의 픽셀 좌표 → 월드 좌표 변환
2. 모든 바퀴와의 거리 계산 → 영향 반경(footprint_radius) 확인
3. 영향권 내: depth = amplitude_slope × Fz + amplitude_intercept
4. New_Height = Current_Height - depth
```

> **주의**: 지형 업데이트 시 전체 DEM을 재로드하므로 실시간 렌더링은 불가. 실시간 동작을 위해서는 지형을 작은 패치로 나누어 관리하는 최적화 필요.

### 파라미터 튜닝 가이드

| 조정 목적 | 파라미터 | 방향 | 결과 |
|----------|---------|------|------|
| 더 단단한 지면 | `c` (점착력), `phi` (마찰각), `k_c`, `k_phi` | ↑ | 침하 감소, 슬립 감소, 주행 저항 감소 |
| 더 푹신한 지면 | `c`, `phi` | ↓ | 침하 증가, 슬립 증가 |
| 바퀴 자국 깊이 증가 | `amplitude_slope` | ↑ | 같은 힘에 더 깊은 변형 |
| 바퀴 자국 유지 시간 | `deform_decay_ratio` | ↓ | 자국이 더 오래 유지 |

#### 시나리오별 파라미터 예시

| 시나리오 | `c` | `phi` | `k_c` |
|---------|-----|-------|-------|
| 지구 흙 | 1000 | 0.6 | 5000 |
| **달 레골리스 (기본값)** | 170 | 0.5 | 1400 |

---

## Dust System

아폴로 미션에서 관찰된 **Rooster Tail** 현상을 NVIDIA Warp GPU 가속 파티클 시스템으로 물리적으로 모사.

### 물리적 특성

| 특성 | 설명 |
|------|------|
| **진공 환경** | 공기 저항 없음 → 먼지 입자가 완벽한 포물선(탄도 궤적)으로 비행 |
| **저중력** | 1.62 m/s² 적용 → 먼지가 더 높이, 더 멀리 날아감 |
| **Rooster Tail 패턴** | 바퀴 회전력에 의해 뒤쪽 위 방향(20~50°), 좌우 ±30° 부채꼴로 퍼짐 |

### GPU 파티클 시스템

| 커널 | 기능 |
|------|------|
| `integrate_particles` | 매 프레임 모든 활성 입자의 위치/속도 갱신 (`v_new = v_old + g×dt`, `p_new = p_old + v_new×dt`) |
| `init_particles` | 새 먼지 생성 시 초기 위치/속도/수명을 GPU 메모리에 할당 |
| `PointInstancer` | 수만 개의 입자를 단일 드로우 콜로 렌더링하여 성능 극대화 |

### 주요 파라미터

| 파라미터 | 값 |
|----------|-----|
| `LUNAR_GRAVITY` | 1.62 m/s² |
| `EJECT_ANGLE` | 0° ~ 50° (수직 기준) |
| `EJECT_SPEED` | 0.5 ~ 1.5 m/s |
| `PARTICLE_LIFETIME` | 0.8 ~ 2.5초 |
| `max_particles` | 100,000개 |

> 실제 달 먼지(20-100μm)는 육안으로 보이지 않으므로, 시각적 효과를 위해 약 1mm~3cm 크기로 확대 렌더링. 지형과 동일한 `LunarRegolith8k` MDL 재질 적용.

---

## SDG System

Isaac Sim의 **Omni Replicator** 프레임워크 기반으로 AI 모델 학습용 합성 데이터셋(이미지 + 레이블)을 자동 생성.

| 모듈 | 역할 |
|------|------|
| `SDGRandomizer` | 학습 데이터 다양성 확보를 위한 시뮬레이션 환경 주기적 변경 |
| `Annotator` | 변경된 장면을 로봇 카메라 시점에서 캡처하고 정답 데이터 생성 |

### 1. SDGRandomizer — 환경 무작위화

`rep.trigger.on_frame`을 통해 설정된 프레임 간격마다 실행.

| 항목 | 기능 |
|------|------|
| 태양광 | `/World/Sun`의 **회전(rotation)** + **광도(intensity)**를 지정 범위 내에서 무작위 변경 |
| 바위 배치 | `rep.randomizer.scatter_2d`로 지형 표면에 무작위 재배치 + `rep.modify.pose`로 회전/크기 추가 무작위화 |

### 2. Annotator — 데이터 주석 및 저장

#### 지원 데이터 유형

| 데이터 유형 | 설명 | 파일 형식 |
|------------|------|----------|
| RGB | 일반 컬러 이미지 | PNG/JPG |
| Depth | 카메라로부터의 거리 정보 | NPY/EXR |
| Semantic Segmentation | 픽셀별 클래스 분류 (지형, 바위, 로버 등) | PNG (Colorized) |
| Instance Segmentation | 개별 객체 단위 분류 (바위1, 바위2...) | PNG (Colorized) |
| Bounding Box 2D | 객체를 감싸는 2D 사각형 좌표 (Tight/Loose) | JSON |
| Bounding Box 3D | 객체의 3차원 공간 위치 및 크기 | JSON |
| Normals | 표면 법선 벡터 이미지 | PNG |

#### 시맨틱 라벨링

```
/World/Terrain  →  "terrain"
/World/Rocks/*  →  "rock"
/World/Husky_*  →  "rover"
```

#### 출력 구조

```
data/sdg_output/
└── 20240501_120000/        # 세션 ID (날짜_시간)
    ├── rgb/                #   rgb_0001.png, rgb_0002.png, ...
    ├── semantic_segmentation/
    ├── bounding_box_2d_tight/
    └── metadata.json       # 카메라 설정, 시뮬레이션 파라미터 등
```

---

## Rover System

로봇 동작은 **USD 파일**, **rover.py**, **solar_panel.py** 세 요소의 조합으로 이루어짐.

| 컴포넌트 | 역할 |
|---------|------|
| `ros2_husky_all.usd` | 3D 모델, 물리 관절, 센서 위치, ROS 2 연동 OmniGraph가 모두 정의된 핵심 설계도 |
| `src/robots/rover.py` | 헤드라이트 등 추가 요소 관리 Python 클래스 |
| `src/robots/solar_panel.py` | 동적으로 생성되는 태양광 패널 제어 클래스 |

---

### 1. USD & OmniGraph

모든 ROS 2 연동 기능이 USD 파일 내의 OmniGraph를 통해 구현.

| OmniGraph 경로 | 목적 | 주요 토픽 / 파라미터 |
|---------------|------|-------------------|
| `/husky_robot/Graphs/differential_controller` | 로봇 제어 구독 | `/cmd_vel` (Twist) / `wheelRadius: 0.165`, `wheelDistance: 0.545` |
| `/husky_robot/Graphs/ROS_Odometry` | 주행 정보 발행 | `/odom` (Odometry), `/tf` (TFMessage) |
| `/husky_robot/Graphs/ROS_LiDAR` | LiDAR 데이터 발행 | `/pointcloud` (PointCloud2) / `frameId: vlp16` |
| `/husky_robot/Graphs/ROS_Camera` | 전면 카메라 + IMU 발행 | `/front_camera/depth`, `/front_camera/mono/rgb`, IMU 데이터 |
| `/husky_robot/Graphs/ROS_Stereo` | 스테레오 카메라 발행 | `/stereo/left/rgb`, `/stereo/right/rgb` |
| `/husky_robot/Graphs/ROS_Joint_state` | 관절 상태 발행 | `/joint_states` (JointState) |
| `/husky_robot/Graphs/ROS_Clock` | 시뮬레이션 시간 발행 | `/clock` (Clock) |

> `rover.py` 클래스는 ROS 2 메시지를 직접 처리할 필요가 없음. 모든 ROS 2 통신은 USD에 내장된 그래프가 자동으로 처리.

#### 스테레오 카메라 설정

| 항목 | 값 | 단위 |
|------|-----|------|
| `focal_length` | 9 | mm |
| `horizontal_aperture` | 12.8 | mm |
| `vertical_aperture` | 9.6 | mm |
| `focus_distance` | 8 | m |
| `width` × `height` | 2288 × 1712 | pixels (4MP 4:3) |

---

### 2. rover.py

#### 휠 조인트 파라미터 튜닝

기본 USD의 설정은 험지 주행 로버에 적합하지 않아 `_fix_wheel_drive_params` 메서드로 실시간 수정.

| 파라미터 | 설정값 | 이유 |
|---------|--------|------|
| `Stiffness` | `0.0` | 기존 높은 강성(10,000,000)은 지형 충돌 시 심한 진동 유발. 0으로 설정하여 순수 속도 제어(Velocity Control) 모드 전환 |
| `Damping` | `5000.0` | 적절한 저항력으로 급격한 가속/튀는 현상 억제 |
| `Max Force` | `500.0` | 비정상적인 물리 연산으로 인한 무한대 힘 방지 |

#### 차체 진동 억제 (Rigid Body Damping)

| 파라미터 | 값 | 효과 |
|---------|-----|------|
| `Linear Damping` | 0.1 | 정지 상태 미끄러짐 억제 |
| `Angular Damping` | 0.05 | 주행 중 차체 불필요한 흔들림 억제 |

#### LED 헤드라이트

야간 주행 및 영구 그림자 영역(PSR) 탐사용.

- 광원: `UsdLux.SphereLight`, 밝기: `2000.0`
- 위치: 로버 전면 상단, 지면을 향해 60° 아래로 기울임
- Shaping: 45° 원뿔(Cone) 형태로 빛을 모아 전방 시야 확보

---

### 3. solar_panel.py

회전 가능한 태양광 패널 시뮬레이션 (시각적 모델링 + 물리적 회전 제어 + 먼지 축적 효율 저하).

#### 3D 모델 구조

| 부품 | 재질 | 설명 |
|------|------|------|
| `FrontFace` | 파란색 | 태양광 흡수 셀 면 (먼지 쌓이면 갈색으로 변색) |
| `BackFace` | 회색 | 패널 뒷면 구조물 |
| `Frame` | 은색 | 패널 금속 프레임 |
| `Support Pole` | — | 로버 `base_link`와 패널 연결 원통형 기둥 |

#### 먼지 축적 및 효율 저하

```
efficiency_factor = 1.0 - dust_accumulation   (0.0 = 깨끗함, 1.0 = 완전 덮임)
```

- `accumulate_dust()`: `dust_accumulation` 수치가 0→1 증가
- 시각적 피드백: 먼지 증가에 따라 FrontFace 색상이 파란색 → 갈색(`0.5, 0.4, 0.3`)으로 점진적 변화

---

## Sun Sensor

로버에 장착된 가상 센서. 태양의 상대적 위치를 추적하고 물리 엔진(PhysX) 기반 레이캐스팅으로 그림자 진입 여부를 판별.

### 태양 벡터 계산

| 메서드 | 설명 |
|--------|------|
| `get_sun_vector()` | 센서(로버) 기준 로컬 좌표계에서 태양 방향 벡터. 패널 입사각 계산에 사용 |
| `get_sun_direction_world()` | 월드 좌표계에서 태양빛 방향 벡터. 패널 법선 벡터와의 내적으로 전력 계산 |

### 물리 기반 그림자 감지 (PhysX Raycast)

```
센서 위치에서 태양 방향으로 Ray 발사
  ↓
1000m 내 Rigid Body 충돌 감지
  ├── 충돌 → in_shadow = True, shadow_factor = 0.0 (그림자)
  └── 미충돌 → in_shadow = False, shadow_factor = 1.0 (직사광선)
```

> **Self-Intersection 방지**: 레이캐스트 시작점을 태양 방향으로 0.1m 띄워서 로버 자신의 차체에 광선이 부딪히는 오작동 방지.

### 시스템 연동

| 연동 시스템 | 효과 |
|------------|------|
| 전력 시스템 | 그림자 진입 시 태양광 발전량을 0W로 차단 |
| 열 시스템 | 그림자 진입 시 로버 표면 온도 급격 하강 (달 야간 온도 모사) |
| HUD | "Sunlight" / "SHADOW" 상태 시각적 경고 표시 |

---

## HUD System

`omni.ui`를 통해 로버의 물리적 상태, 전력 시스템, 환경 변수를 실시간으로 시각화.

### 주요 UI 섹션

| 섹션 | 기능 | 시각화 특징 |
|------|------|------------|
| **Battery** | 배터리 잔량 및 상태 | 잔량에 따라 색상 동적 변경 (초록 → 노랑 → 주황 → 빨강) |
| **Solar Power** | 태양광 발전 효율 모니터링 | 발전량 게이지 바, 그림자(Shadow) 진입 경고, Net Power 표시 |
| **Temperature** | 로버 외부 온도 | -150°C ~ 150°C 범위 그라데이션 바, 적정 범위 이탈 시 색상 경고 |
| **Rover State** | 물리적 움직임 정보 | 속도, 3D 위치 좌표, 회전 각도(Yaw/Roll/Pitch) 수치 표시 |
| **Simulation** | 시뮬레이션 및 천체 정보 | Stellar Engine 연동 태양 고도/방위각, 달 현지 시간 표시 |

### HUDData 주요 필드

| 카테고리 | 데이터 |
|---------|--------|
| 전력 시스템 | 배터리 잔량(%, Wh), 충/방전 상태, 태양광 발전량(W), 패널 효율, Net Power |
| 환경 정보 | 외부 온도(°C), 태양 입사각, 그림자 진입 여부(`in_shadow`) |
| 로버 상태 | 속도(m/s), 위치(x, y, z), 자세(Yaw, Roll, Pitch) |
| Stellar Engine | 달 현지 시간, 태양 고도(Altitude), 방위각(Azimuth) |

---

## ROS 2 Custom Nodes

### 1. solar_control_node.py

독립적인 ROS 2 노드. Isaac Sim 시뮬레이션과 토픽을 통해 통신하며 태양광 패널 자동 추적.

#### 핵심 제어 로직 (`control_loop`, 10Hz)

```python
# 태양 방위각 계산
sun_angle = math.atan2(s_y, s_x)

# 목표 각도 계산
# solar_panel.py에서 0도 명령 = 로봇 뒤쪽(180도) 방향
# → sun_angle = cmd + π → cmd = sun_angle - π
target_cmd = sun_angle - math.pi

# 각도 정규화: [-π, +π] 범위로
if target_cmd < -math.pi:
    target_cmd += 2 * math.pi
elif target_cmd > math.pi:
    target_cmd -= 2 * math.pi
```

#### 토픽 구성

| 방향 | 토픽 | 타입 |
|------|------|------|
| Subscribe | `/husky_1/sun_vector` | `geometry_msgs/Vector3` |
| Publish | `/husky_1/solar_panel/cmd_angle` | `std_msgs/Float32` |

---

### 2. noise_node.py

Isaac Sim의 이상적인(Clean) 이미지에 실제 카메라 센서의 물리적 특성과 결함을 후처리로 주입. 달 극지방의 **영구 그림자 영역(PSR)**과 같은 극저조도 환경에서의 센서 한계를 정밀 모사.

#### 노이즈 모델 비교

| 구분 | 단순 모델 (Basic Noise) | 물리 기반 모델 (Moseley et al. CVPR 2021) |
|------|----------------------|----------------------------------------|
| 종류 | Gaussian, Salt & Pepper, Exposure Variation | Shot Noise, Dark Current, Read Noise, FPN, PRNU |
| 목적 | 일반 컴퓨터 비전 테스트 | 달 극한 환경 센서 한계 정밀 모사 |

#### 물리 기반 노이즈 파이프라인

| 단계 | 노이즈 유형 | 설명 |
|------|------------|------|
| 1. 광자 입사 | PRNU (감도 편차) | 픽셀마다 빛 감도가 미세하게 다른 제조 공정상의 결함 (곱셈적 노이즈) |
| 2. 전하 변환 | Shot Noise | 광자 도달의 양자역학적 불확실성. 신호가 약할수록 상대적 잡음 증가 (Poisson 분포) |
| 3. 열전자 생성 | Dark Current | 빛이 없어도 열에 의해 전자가 생성. 센서 온도에 따라 **지수적** 증가 |
| 4. 회로 판독 | Read Noise | 아날로그 회로(앰프, ADC) 통과 시 발생하는 전자적 잡음 |
| 5. 오프셋 | FPN (고정 패턴) | 센서 픽셀별 고정된 오프셋 값 차이 (덧셈적 노이즈) |

#### 환경 연동 — 온도에 따른 동적 노이즈

- `/rover/sensor_temperature` 토픽으로 로버의 현재 센서 온도를 실시간 수신
- **Arrhenius 법칙**: 온도 10°C 상승마다 Dark Current 약 2배 증가
- 로버가 뜨거워지면 카메라 이미지에 열잡음(Thermal Noise) 자동 증가

#### 성능 최적화

| 기법 | 설명 |
|------|------|
| Pre-calculated Buffers | PRNU, FPN 등 고정 패턴은 미리 계산하여 메모리 캐싱 |
| Vectorization | NumPy 벡터 연산으로 대량 픽셀 병렬 처리 |
| In-place Operation | 메모리 할당/복사 최소화 |
| Gaussian Approximation | 광자 수 충분 시 Poisson 대신 Gaussian 근사로 속도 향상 |

---

### 3. denoise_node.py — NAFNet

물리적 노이즈가 주입된 이미지를 **NAFNet** 딥러닝 모델로 실시간 복원. 실제 로버의 ISP(Image Signal Processor) 후처리 과정 시뮬레이션.

#### NAFNet (Nonlinear Activation Free Network, ECCV 2022)

| 항목 | 내용 |
|------|------|
| 핵심 특징 | ReLU/GELU 제거, 행렬 곱셈 + **SimpleGate** 메커니즘만으로 고성능 달성 |
| 성능 | SIDD 벤치마크 PSNR **40.30dB** (SOTA급) |
| 구조 | NAFBlock(Layer Norm + Conv + SimpleGate + SCA) + U-Net 인코더-디코더 |

#### 데이터 흐름

```
입력: /stereo/{side}/rgb_noisy/compressed  (대역폭 절약 압축 이미지)
  ↓  PyTorch GPU 추론 (CUDA)
출력: /stereo/{side}/rgb_denoised          (복원된 Raw 이미지)
      /stereo/{side}/rgb_denoised/compressed (모니터링용)
```

#### 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `model_width` | 채널 폭 (64: PSNR 40.30dB, 67.89M params / 32: PSNR 39.97dB, 17.11M params) | 64 |
| `max_rate` | 최대 처리 프레임 속도 (Hz). GPU 부하 조절 | 15.0 |
| `use_raw_input` | `True`: Raw 이미지, `False`: Compressed 이미지 구독 | `False` |

---

### 4. enhance_node.py — DimCam

자체 개발된 **Lproject_cam** 딥러닝 모델로 달의 PSR과 같은 극한 저조도 환경에서 획득한 스테레오 이미지를 사람이 식별 가능한 수준으로 향상.

- **추론 시간**: 60~100ms

#### Lproject_cam — 스테레오 특화 개선 모델

| 특징 | 설명 |
|------|------|
| **Stereo-Aware** | 좌/우 이미지를 하나의 네트워크에서 동시 연산 → 밝기 균형 + 특징점 매칭 확률 향상 |
| **Transformer 기반** | `embed_dim` + `num_blocks` 파라미터로 제어하는 트랜스포머 블록 → 전역 문맥(Global Context) 파악 |
| **유연한 입력** | `noisy` (원본) 또는 `denoised` (NAFNet 출력) 입력 선택 가능 |

#### 데이터 동기화

`ApproximateTimeSynchronizer`: 좌/우 카메라 토픽의 타임스탬프 비교, **100ms 이내** 오차를 가진 프레임 쌍(Pair)을 찾아 동시 처리.

#### 성능 최적화

| 기법 | 설명 |
|------|------|
| FP16 (Half Precision) | GPU 메모리 절약 + 추론 속도 향상 (`use_fp16=True`) |
| Resizing | 입력을 모델 입력 크기(512×512)로 축소 후 추론, 원본 해상도로 복원하여 발행 |
| Rate Limiting | `max_rate`로 초당 처리 횟수 제한 |

#### 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `model_weights` | 학습된 DimCam 모델 가중치 경로 (.pth) | `dimcam_enhancer_epoch_30.pth` |
| `input_mode` | 입력 소스 선택 (`noisy` / `denoised` / `raw`) | `noisy` |
| `img_size` | 모델 추론 해상도 (정사각형) | 512 |
| `use_fp16` | FP16 가속 사용 여부 (GPU 필요) | `True` |
| `max_rate` | 목표 처리 속도 (Hz) | 15.0 |

---

## Navigation: LiDAR + Nav2

### nav2.launch.py — 주요 구성 요소

| 노드 | 역할 |
|------|------|
| `pointcloud_to_laserscan_node` | 3D PointCloud2 → 2D LaserScan 변환. `min_height` ~ `max_height` 범위 외 노이즈 제거 |
| `nav2_bringup_group` | Nav2 전체 핵심 노드 실행 (Controller, Planner, BT Navigator 등) |

### nav2_params.yaml — 주요 컴포넌트

| 컴포넌트 | 역할 | 핵심 파라미터 |
|---------|------|-------------|
| `slam_toolbox` | SLAM: 지도 작성 + 위치 추정 | `use_odometry: false` + `publish_odom_tf: true` → 바퀴 오도메트리 대신 LiDAR 스캔 매칭만으로 위치 추정 (연약 지반 대응) |
| `planner_server` | 전역 경로 계획 (현재 위치 → 목표) | `plugin: nav2_navfn_planner::NavfnPlanner` (A* 알고리즘) |
| `controller_server` | 지역 경로 계획 + 주행 명령 생성 | `plugin: dwb_core::DWBLocalPlanner` (DWB 기반), critics: 장애물 거리, 경로 추종성 등 |
| `local_costmap` | 로봇 주변 소영역 실시간 업데이트 | `controller_server` 사용 |
| `global_costmap` | 전체 지도 기반 비용 지도 | `planner_server` 사용 |
| `bt_navigator` | 자율 주행 흐름 조율 (경로 계획→추종→도착), 실패 시 복구 행동 지시 | — |
| `collision_monitor` | 충돌 위험 감지 시 즉시 정지 | — |

### 알려진 한계

- **평지**: LiDAR 데이터 노이즈 없음 → 장애물 회피 및 자율주행 원활
- **울퉁불퉁한 지형**: 로버가 기울어지며 LiDAR가 바닥면을 장애물로 잘못 인식 → 자율주행 불안정

---

## Navigation: Stereo SLAM Stack

`ORB-SLAM3`, `FoundationStereo`, `Nvblox`, `Nav2`의 통합 파이프라인으로 스테레오 이미지 기반 자율 주행 지원.

---

## References & Roadmap

### 목표

OmniLRS & jaops-sim 벤치마킹을 통한 달 환경 자율주행 시뮬레이션 구축.

**핵심 3요소**:
1. 사실적인 달 지형(Terrain) & 조명(Lighting)
2. 로버의 물리적 거동(Physics)
3. ROS 2를 통한 데이터 통신(Integration)

**기반 환경**: Ubuntu 24.04 · ROS 2 Jazzy · Isaac Sim 5.0.0

### 참고 오픈소스

| 프로젝트 | 링크 |
|---------|------|
| OmniLRS | `https://github.com/OmniLRS/OmniLRS.git` |
| OmniLRS ROS 2 Demo | `https://github.com/OmniLRS/omnilrs_ros2_demo.git` |
| JAOPS Sim | `https://github.com/jaops-space/jaops-sim.git` |

### 개발 로드맵

| Step | 단계 | 목표 |
|------|------|------|
| 0-1 | 로봇 설정 학습 | Robot Setup Tutorials Series (Isaac Sim), 센서, OmniGraph |
| 0-2 | 랜덤 씬 생성 학습 | Replicator 이용 SDG (randomization, sensor simulation, data collection) |
| 0-3 | ROS 2 + OmniGraph 학습 | Isaac Sim ROS 2 Bridge 및 통합 튜토리얼 |
| 1-1 | 달 환경 구축 | 달 지형 + 태양광 조명 설정, GUI 조작 학습 |
| 1-2 | 지형 생성 | 기본 지형 생성 & 에셋 배치 (DEM import) |
| 1-3 | 조명 설정 | 시간/위도/경도별 태양광 조절 |
| 2-1 | 로버 모델링 | 로봇 관절(Joint) & 구동부(Drive) 설정 |
| 2-2 | USD/URDF Import | 기존 ROS 로버 모델 임포트 |
| 2-3 | 물리 엔진 | 중력, 마찰력, 바퀴 회전 설정 |
| 3-1 | 센서 부착 | 스테레오 카메라 & IMU 센서 부착 |
| 3-2 | IMU 센서 | 가속도/자이로 데이터 수집 |
| 3-3 | 센서 데이터 출력 | 이미지 저장 & 학습 데이터 추출 |
| 4-1 | ROS 2 연동 | Isaac Sim ↔ ROS 2 연결 |
| 4-2 | 주행 제어 | `cmd_vel` 명령으로 로버 제어 |
| 4-3 | 센서 통신 | 카메라/IMU 데이터를 ROS 2 토픽으로 발행 |

### USD vs URDF 비교

| 항목 | USD | URDF |
|------|-----|------|
| 주요 목적 | **전체 3D 씬(Scene)**의 구성 및 렌더링 | **단일 로봇**의 기구학적/동역학적 구조 정의 |
| 범위 | 모델, 환경, 조명, 카메라, 애니메이션 등 | 로봇의 링크(Links)와 조인트(Joints) |
| 핵심 생태계 | NVIDIA Omniverse, Isaac Sim, VFX, AR/VR | ROS (Robot Operating System) |
| 시각적 표현 | 매우 뛰어남 (고품질 재질, 텍스처, 렌더링 정보) | 제한적 (외부 3D 모델 파일에 의존, 재질 표현 빈약) |
| 구조 | 레이어 기반의 합성(Compositional) 구조 | 링크-조인트 기반의 트리(Tree) 구조 |
| Isaac Sim에서 | 네이티브(Native) 포맷. 모든 것의 기반 | 임포트(Import) 대상. 내부적으로 USD로 변환됨 |
