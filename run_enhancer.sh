#!/bin/bash
# DimCam Enhancer Node 실행 스크립트
# 다중 사용자 ROS 2 통신을 위한 환경 설정 포함

set -e

source /opt/ros/jazzy/setup.bash


# --- ROS 2 환경 설정 (Isaac Sim과 동일하게) ---
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

# 다중 사용자 통신: Shared Memory 비활성화, UDP만 사용
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

echo "=== DimCam Enhancer Node ==="
echo "  ROS Distro: $ROS_DISTRO"
echo "  RMW: $RMW_IMPLEMENTATION"
echo "  Domain ID: $ROS_DOMAIN_ID"
echo "  Discovery: $ROS_AUTOMATIC_DISCOVERY_RANGE"
echo "  Transport: $FASTDDS_BUILTIN_TRANSPORTS (No Shared Memory)"
echo ""

# --- 실행 옵션 ---
# 기본값
INPUT_MODE="noisy"
MAX_RATE=15.0

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --raw)
            INPUT_MODE="raw"
            shift
            ;;
        --denoised)
            INPUT_MODE="denoised"
            shift
            ;;
        --noisy)
            INPUT_MODE="noisy"
            shift
            ;;
        --rate)
            MAX_RATE=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --noisy       Use noisy compressed input (default) - /stereo/*/rgb_noisy/compressed"
            echo "  --denoised    Use denoised compressed input - /stereo/*/rgb_denoised/compressed"
            echo "  --raw         Use raw image input - /stereo/*/rgb"
            echo "  --rate HZ     Target processing rate (default: 15.0)"
            echo "  -h, --help    Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                    # Noisy compressed input (default)"
            echo "  $0 --denoised         # Denoised compressed input"
            echo "  $0 --raw              # Raw input (~15-20fps)"
            echo "  $0 --denoised --rate 30"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- 스크립트 디렉토리로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 모델 경로 ---
MODEL_WEIGHTS="${SCRIPT_DIR}/Lproject_cam/dimcam_enhancer_epoch_30.pth"

# 모델 파일 확인
if [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "========================================"
    echo "⚠️  DimCam 모델을 찾을 수 없습니다"
    echo "========================================"
    echo "  경로: ${MODEL_WEIGHTS}"
    echo ""
fi

# --- 실행 ---
echo "Starting enhancer node..."
echo "  Input mode: $INPUT_MODE"
echo "  Max Rate: $MAX_RATE Hz"
echo "  Model: $MODEL_WEIGHTS"
echo ""

# src/nodes 폴더에서 실행
python3 "${SCRIPT_DIR}/src/nodes/enhance_node.py" --ros-args \
    -p input_mode:=$INPUT_MODE \
    -p max_rate:=$MAX_RATE \
    -p model_weights:="${MODEL_WEIGHTS}"
