#!/bin/bash
# Camera Noise Node 실행 스크립트
# 멀티유저 ROS2 통신을 위한 환경 설정 포함

set -e

# --- ROS 2 Environment Setup ---
# Source ROS 2 (adjust path if needed)
source /opt/ros/jazzy/setup.bash

# --- Multi-user ROS 2 Discovery Settings ---
# CRITICAL: These settings allow cross-user communication
# Shared Memory (SHM) only works within same user, so we disable it
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# --- Show help if requested ---
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Camera Noise Node - Adds physical sensor noise to camera images"
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Description:"
    echo "  Subscribes to stereo camera topics and publishes noisy versions."
    echo "  Implements physical noise model (shot noise, dark current, read noise, etc.)"
    echo ""
    echo "ROS 2 Topics:"
    echo "  Subscribed:  /stereo/left/rgb, /stereo/right/rgb"
    echo "  Published:   /stereo/left/rgb_noisy, /stereo/right/rgb_noisy"
    echo "               /stereo/left/rgb_noisy/compressed, /stereo/right/rgb_noisy/compressed"
    echo ""
    echo "Configuration:"
    echo "  Edit config/simulation_config.yaml to adjust noise parameters"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start noise node"
    echo ""
    exit 0
fi

echo "=== Camera Noise Node ==="
echo "  ROS Distro: $ROS_DISTRO"
echo "  RMW: $RMW_IMPLEMENTATION"
echo "  Domain ID: $ROS_DOMAIN_ID"
echo "  Discovery: $ROS_AUTOMATIC_DISCOVERY_RANGE"
echo "  Transport: $FASTDDS_BUILTIN_TRANSPORTS (Shared Memory DISABLED for multi-user)"
echo ""

# --- Run the node ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

cd "$PROJECT_DIR"

# Use system Python with ROS 2
python3 -m src.nodes.noise_node "$@"
