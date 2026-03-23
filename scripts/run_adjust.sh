#!/bin/bash
# Stereo Adjust Node 실행 스크립트
# 스테레오 이미지 동기화 + FPS 조절

set -e

# --- ROS 2 Environment Setup ---
source /opt/ros/jazzy/setup.bash

# --- Multi-user ROS 2 Discovery Settings ---
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

echo "=== Stereo Adjust Node ==="
echo "  ROS Distro: $ROS_DISTRO"
echo "  RMW: $RMW_IMPLEMENTATION"
echo "  Domain ID: $ROS_DOMAIN_ID"
echo "  Discovery: $ROS_AUTOMATIC_DISCOVERY_RANGE"
echo "  Transport: $FASTDDS_BUILTIN_TRANSPORTS (No Shared Memory)"
echo ""

# --- 기본값 ---
TARGET_FPS=5.0
INPUT_TYPE="enhanced"
SYNC_SLOP=0.05
OUTPUT_SUFFIX="adjusted"

# --- 인자 파싱 ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --fps)
            TARGET_FPS=$2
            shift 2
            ;;
        --input)
            INPUT_TYPE=$2
            shift 2
            ;;
        --slop)
            SYNC_SLOP=$2
            shift 2
            ;;
        --output-suffix)
            OUTPUT_SUFFIX=$2
            shift 2
            ;;
        --raw)
            INPUT_TYPE="raw"
            shift
            ;;
        --noisy)
            INPUT_TYPE="noisy"
            shift
            ;;
        --enhanced)
            INPUT_TYPE="enhanced"
            shift
            ;;
        --noisy-compressed)
            INPUT_TYPE="noisy_compressed"
            shift
            ;;
        --enhanced-compressed)
            INPUT_TYPE="enhanced_compressed"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fps HZ               Target output FPS (default: 5.0)"
            echo "  --input TYPE            Input type (default: enhanced)"
            echo "  --slop SEC              Sync time tolerance in seconds (default: 0.05)"
            echo "  --output-suffix NAME    Output topic suffix (default: adjusted)"
            echo ""
            echo "Input type shortcuts:"
            echo "  --raw                   /stereo/*/rgb (Raw Image, RELIABLE)"
            echo "  --noisy                 /stereo/*/rgb_noisy (Raw Image, RELIABLE)"
            echo "  --noisy-compressed      /stereo/*/rgb_noisy/compressed (CompressedImage, BEST_EFFORT)"
            echo "  --enhanced              /stereo/*/enhanced (Raw Image, RELIABLE)"
            echo "  --enhanced-compressed   /stereo/*/enhanced/compressed (CompressedImage, BEST_EFFORT)"
            echo ""
            echo "Examples:"
            echo "  $0                              # enhanced input, 5 FPS"
            echo "  $0 --fps 10 --enhanced          # enhanced input, 10 FPS"
            echo "  $0 --fps 3 --noisy              # noisy raw input, 3 FPS"
            echo "  $0 --enhanced-compressed --fps 5 # compressed enhanced, 5 FPS"
            exit 0
            ;;
        *)
            echo "Unknown option: $1 (use -h for help)"
            exit 1
            ;;
    esac
done

# --- 스크립트 디렉토리로 이동 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 실행 ---
echo "Starting Stereo Adjust Node..."
echo "  Input type: $INPUT_TYPE"
echo "  Target FPS: $TARGET_FPS Hz"
echo "  Sync slop: ${SYNC_SLOP}s"
echo "  Output suffix: $OUTPUT_SUFFIX"
echo ""

python3 -m src.nodes.adjust_node --ros-args \
    -p target_fps:=$TARGET_FPS \
    -p input_type:=$INPUT_TYPE \
    -p sync_slop:=$SYNC_SLOP \
    -p output_suffix:=$OUTPUT_SUFFIX
