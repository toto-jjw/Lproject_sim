#!/bin/bash
# Solar Panel Control Node 실행 스크립트
# 멀티유저 ROS2 통신을 위한 환경 설정 포함

set -e

# --- ROS 2 Environment Setup ---
source /opt/ros/jazzy/setup.bash

# --- Multi-user ROS 2 Discovery Settings ---
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# --- Show help if requested ---
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Solar Panel Control Node - Tracks sun direction and adjusts panel angle"
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Description:"
    echo "  Subscribes to sun vector topic and publishes optimal solar panel angle."
    echo "  Gradually rotates the panel to track the sun (0.3 rad/s rotation speed)."
    echo ""
    echo "ROS 2 Topics:"
    echo "  Subscribed:  /husky_1/sun_vector (geometry_msgs/Vector3)"
    echo "  Published:   /husky_1/solar_panel/cmd_angle (std_msgs/Float32)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start solar panel tracking"
    echo ""
    exit 0
fi

echo "=== Solar Panel Control Node ==="
echo "  ROS Distro: $ROS_DISTRO"
echo "  RMW: $RMW_IMPLEMENTATION"
echo "  Domain ID: $ROS_DOMAIN_ID"
echo "  Discovery: $ROS_AUTOMATIC_DISCOVERY_RANGE"
echo "  Transport: $FASTDDS_BUILTIN_TRANSPORTS (Shared Memory DISABLED for multi-user)"
echo ""

# --- Run the node ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m src.nodes.solar_control_node "$@"
