#!/bin/bash
# Camera Denoising Node (NAFNet) 실행 스크립트
# 다중 사용자 ROS 2 통신을 위한 환경 설정 포함
#
# 사전학습 모델 다운로드 필요:
# https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view
#
# 다운로드 후 다음 경로에 배치:
# NAFNet/experiments/pretrained_models/NAFNet-SIDD-width64.pth

set -e


source /opt/ros/jazzy/setup.bash

# --- ROS 2 환경 설정 (Isaac Sim과 동일하게) ---
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

# 다중 사용자 통신: Shared Memory 비활성화, UDP만 사용
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

echo "=== Camera Denoising Node (NAFNet) ==="
echo "  ROS Distro: $ROS_DISTRO"
echo "  RMW: $RMW_IMPLEMENTATION"
echo "  Domain ID: $ROS_DOMAIN_ID"
echo "  Discovery: $ROS_AUTOMATIC_DISCOVERY_RANGE"
echo "  Transport: $FASTDDS_BUILTIN_TRANSPORTS (No Shared Memory)"
echo ""

# --- 실행 옵션 ---
# 기본값
USE_RAW=false
MAX_RATE=15.0
MODEL_WIDTH=64

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --raw)
            USE_RAW=true
            shift
            ;;
        --rate)
            MAX_RATE=$2
            shift 2
            ;;
        --width)
            MODEL_WIDTH=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --raw           Use raw image input (/stereo/left/rgb_noisy) instead of compressed"
            echo "  --rate HZ       Target processing rate (default: 15.0)"
            echo "  --width WIDTH   Model width: 32 or 64 (default: 64)"
            echo "                    - width64: 67.89M params, 40.30dB PSNR (higher quality)"
            echo "                    - width32: 17.11M params, 39.97dB PSNR (lighter, faster)"
            echo "  -h, --help      Show this help"
            echo ""
            echo "Input topics (compressed mode - default):"
            echo "  /stereo/left/rgb_noisy/compressed"
            echo "  /stereo/right/rgb_noisy/compressed"
            echo ""
            echo "Output topics:"
            echo "  /stereo/left/rgb_denoised"
            echo "  /stereo/left/rgb_denoised/compressed"
            echo "  /stereo/right/rgb_denoised"
            echo "  /stereo/right/rgb_denoised/compressed"
            echo ""
            echo "Examples:"
            echo "  $0                    # Compressed input, width64 model"
            echo "  $0 --width 32         # Use lighter width32 model"
            echo "  $0 --raw              # Raw input"
            echo "  $0 --raw --rate 30    # Raw input, max 30Hz"
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

# --- 모델 경로 설정 ---
MODEL_PATH="${SCRIPT_DIR}/NAFNet/experiments/pretrained_models/NAFNet-SIDD-width${MODEL_WIDTH}.pth"

# 모델 파일 확인
if [ ! -f "$MODEL_PATH" ]; then
    echo "========================================"
    echo "⚠️  NAFNet 사전학습 모델을 찾을 수 없습니다"
    echo "========================================"
    echo ""
    echo "다운로드 링크:"
    echo "  NAFNet-SIDD-width64 (40.30dB PSNR, 67.89M params):"
    echo "  https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view"
    echo ""
    echo "  NAFNet-SIDD-width32 (39.97dB PSNR, 17.11M params, 경량):"
    echo "  https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view"
    echo ""
    echo "다운로드 후 다음 경로에 배치하세요:"
    echo "  ${MODEL_PATH}"
    echo ""
    echo "모델 없이 계속 실행합니다 (테스트용, 디노이징 효과 없음)..."
    echo ""
fi

# --- 실행 ---
echo "Starting NAFNet denoising node..."
echo "  Input: $([ "$USE_RAW" = true ] && echo "Raw" || echo "Compressed")"
echo "  Max Rate: $MAX_RATE Hz"
echo "  Model Width: $MODEL_WIDTH"
echo "  Model: $MODEL_PATH"
echo ""

# src/nodes 폴더에서 실행
python3 "${SCRIPT_DIR}/src/nodes/denoise_node.py" --ros-args \
    -p use_raw_input:=$USE_RAW \
    -p max_rate:=$MAX_RATE \
    -p model_width:=$MODEL_WIDTH \
    -p model_path:="${MODEL_PATH}"
