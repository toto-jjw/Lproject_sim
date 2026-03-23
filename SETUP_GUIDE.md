# 환경 세팅 및 전체 기능 실행 방법

## 목차

- [1. 컴퓨터 환경](#1-컴퓨터-환경)
- [2. 사전 요구사항 설치](#2-사전-요구사항-설치)
  - [Lproject_sim 설치](#lproject_sim-설치)
  - [Isaac Sim 설치](#isaac-sim-설치)
  - [ROS 2 Jazzy 설치](#ros-2-jazzy-설치)
  - [Isaac Sim ROS2 빌드](#isaac-sim-ros2-빌드)
- [3. 시뮬레이션 실행](#3-시뮬레이션-실행)
  - [터미널 1 — Isaac Sim 시뮬레이션 시작](#1-터미널-1--isaac-sim-시뮬레이션-시작)
  - [시뮬레이션 기능 사용](#2-시뮬레이션-기능-사용)
  - [스테레오 이미지 기반 자율주행 파이프라인 구동](#3-스테레오-이미지-기반-자율주행-파이프라인-구동)

---

## 1. 컴퓨터 환경

| 항목 | 버전 |
|------|------|
| Kernel | Linux 6.14.0-33-generic |
| OS | Ubuntu 24.04.3 LTS (Noble) |
| GPU Driver | 580.119.02 |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Python | 3.12.3 |

---

## 2. 사전 요구사항 설치

### Lproject_sim 설치

```bash
mkdir ~/Lproject_sim
cd ~/Downloads
unzip "Lproject_sim.zip" -d ~/Lproject_sim
```

### Isaac Sim 설치

Isaac Sim 5.0.0 설치: [Workstation Installation — Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html)

> Isaac Sim 5.0.0 standalone zip 다운로드:
> https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip

```bash
mkdir ~/isaacsim
cd ~/Downloads
unzip "isaac-sim-standalone-5.0.0-linux-x86_64.zip" -d ~/isaacsim
cd ~/isaacsim
./post_install.sh

python.sh -m pip install skyfield   # 천체 계산 라이브러리 설치
./isaac-sim.selector.sh             # Isaac Sim 버전 선택기 실행
./isaac-sim.sh                      # Isaac Sim 실행
```

### ROS 2 Jazzy 설치

참고: [ROS 2 Installation — Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_ros.html)

#### System setup — Set locale

```bash
locale  # UTF-8 확인

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # 설정 확인
```

#### Enable required repositories

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
```

#### Install development tools (optional)

```bash
sudo apt update && sudo apt install ros-dev-tools
```

#### Install ROS 2

```bash
sudo apt update
sudo apt upgrade

# Desktop Install (권장) — ROS, RViz, demos, tutorials
sudo apt install ros-jazzy-desktop

# ROS-Base Install (최소) — 통신 라이브러리 및 CLI 도구만 설치
sudo apt install ros-jazzy-ros-base
```

#### Setup environment

```bash
source /opt/ros/jazzy/setup.bash
```

### Isaac Sim ROS2 빌드

Isaac Sim과 연동할 ROS 2 라이브러리를 빌드합니다 (이미 빌드했다면 건너뛰기).

> **주의:** Isaac Sim 5.x는 Python 3.11, ROS 2 Jazzy는 Python 3.12를 사용하므로 `rclpy` 등 Python 라이브러리의 직접 호환이 불가합니다. NVIDIA에서 제공하는 Dockerfile 기반 빌드를 사용해야 합니다.
>
> 현재 GitHub(`https://github.com/isaac-sim/IsaacSim-ros_workspaces.git`)에 배포된 파일은 Isaac Sim 6.0.0 / Python 3.12 버전으로 변경되어 **Isaac Sim 5.x.x에는 적용 불가**합니다.
> 반드시 아래 파일을 다운로드하여 진행하세요: **IsaacSim-ros_workspaces.zip**

```bash
mkdir ~/IsaacSim-ros_workspaces
cd ~/Downloads
unzip "IsaacSim-ros_workspaces.zip" -d ~/IsaacSim-ros_workspaces
cd ~/IsaacSim-ros_workspaces

# ROS 2 Jazzy, Ubuntu 24.04 빌드
./build_ros.sh -d jazzy -v 24.04
```

빌드가 완료되면 `~/IsaacSim-ros_workspaces/build_ws/jazzy` 폴더에 Isaac Sim 전용 라이브러리가 생성됩니다.

---

## 3. 시뮬레이션 실행

### 1. [터미널 1] — Isaac Sim 시뮬레이션 시작

```bash
cd ~/Lproject_sim

# 도움말 확인
./scripts/start_simulation.sh -h

# 시뮬레이션 시작
./scripts/start_simulation.sh
```

### 2. 시뮬레이션 기능 사용

```bash
cd ~/Lproject_sim

# 태양광 패널 시뮬레이션 실행
./scripts/solar_panel.sh

# 이미지 노이즈 추가
./scripts/run_noiser.sh

# 이미지 디노이즈 (NAFNet)
./scripts/run_denoiser.sh

# 저조도 이미지 향상 (LprojectCam)
./scripts/run_enhancer.sh

# 이미지 퍼블리시 hz 조정
./scripts/run_adjust.sh
```

#### ROS 2 기본 명령어

```bash
source /opt/ros/jazzy/setup.bash

ros2 -h
ros2 topic -h
rviz2

# 키보드로 로봇 제어 (uiojklm,.)
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

#### 로봇 위치 제어

```bash
# 특정 위치로 로봇 이동
ros2 topic pub --once /husky_1/reset_pose_target geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'world'}, pose: {position: {x: 5.0, y: 3.0, z: 0.5}, orientation: {w: 1.0}}}"

# 리셋 목표 위치 설정 (R/Trigger 버튼으로 이동)
ros2 topic pub --once /husky_1/set_reset_target geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'world'}, pose: {position: {x: 10.0, y: 0.0, z: 0.5}, orientation: {w: 1.0}}}"

# 로봇 위치 리셋 실행 (Isaac Sim에서 'R' 키로도 가능)
ros2 topic pub --once /husky_1/reset_pose_cmd std_msgs/msg/Empty
```

### 3. 스테레오 이미지 기반 자율주행 파이프라인 구동

```bash
# 도움말 확인
./scripts/run_pipeline.sh -h

# 전체 파이프라인 실행 (sim 모드)
./scripts/run_pipeline.sh --all --sim
```
