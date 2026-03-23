#!/bin/bash
set -e

# --- Project Root (auto-detect from script location) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LPROJECT_SIM_DIR="$SCRIPT_DIR"

# --- Isaac Sim Path ---
# Set via environment variable, or edit this default for your system
export ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-$HOME/isaacsim}"

# 2. Source Custom ROS Build
source ~/IsaacSim-ros_workspaces/build_ws/jazzy/jazzy_ws/install/local_setup.bash
source ~/IsaacSim-ros_workspaces/build_ws/jazzy/isaac_sim_ros_ws/install/local_setup.bash



# --- Configuration ---
PYTHON_EXE="$ISAAC_SIM_PATH/python.sh"
MAIN_SCRIPT="main.py"

# --- Usage Function ---
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --sample-scene    Load pre-saved sample scene for faster startup"
    echo "  --scene-path PATH     Custom path to saved USD scene file"
    echo "  --save-scene          Save current scene after initialization"
    echo "  --save-scene-all      Save 12 scene variants (3 rocks × 2 lights × 2 terrain)"
    echo ""
    echo "  Scene Variants (with outer terrain):"
    echo "  --s1    Full rocks + Bright"
    echo "  --s2    Full rocks + Dim"
    echo "  --s3    Half rocks + Bright"
    echo "  --s4    Half rocks + Dim"
    echo "  --s5    No rocks + Bright"
    echo "  --s6    No rocks + Dim"
    echo ""
    echo "  Scene Variants (without outer terrain):"
    echo "  --s7    Full rocks + Bright (no outer)"
    echo "  --s8    Full rocks + Dim (no outer)"
    echo "  --s9    Half rocks + Bright (no outer)"
    echo "  --s10   Half rocks + Dim (no outer)"
    echo "  --s11   No rocks + Bright (no outer)"
    echo "  --s12   No rocks + Dim (no outer)"
    echo ""
    echo "  --headless            Run in headless mode (no GUI)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Normal startup (generate terrain from scratch)"
    echo "  $0 --sample-scene     # Fast startup using saved scene"
    echo "  $0 --save-scene-all   # Save 12 variants (s1~s12)"
    echo "  $0 --s1               # Full rocks + bright (with outer terrain)"
    echo "  $0 --s7               # Full rocks + bright (no outer terrain)"
    echo "  $0 --s1 --headless    # Fast startup in headless mode"
    exit 0
}

# --- Parse Arguments ---
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--sample-scene)
            EXTRA_ARGS="$EXTRA_ARGS --sample-scene"
            shift
            ;;
        --scene-path)
            EXTRA_ARGS="$EXTRA_ARGS --scene-path $2"
            shift 2
            ;;
        --save-scene)
            EXTRA_ARGS="$EXTRA_ARGS --save-scene"
            shift
            ;;
        --save-scene-all)
            EXTRA_ARGS="$EXTRA_ARGS --save-scene-all"
            shift
            ;;
        --s1)
            EXTRA_ARGS="$EXTRA_ARGS --s1"
            shift
            ;;
        --s2)
            EXTRA_ARGS="$EXTRA_ARGS --s2"
            shift
            ;;
        --s3)
            EXTRA_ARGS="$EXTRA_ARGS --s3"
            shift
            ;;
        --s4)
            EXTRA_ARGS="$EXTRA_ARGS --s4"
            shift
            ;;
        --s5)
            EXTRA_ARGS="$EXTRA_ARGS --s5"
            shift
            ;;
        --s6)
            EXTRA_ARGS="$EXTRA_ARGS --s6"
            shift
            ;;
        --s7)
            EXTRA_ARGS="$EXTRA_ARGS --s7"
            shift
            ;;
        --s8)
            EXTRA_ARGS="$EXTRA_ARGS --s8"
            shift
            ;;
        --s9)
            EXTRA_ARGS="$EXTRA_ARGS --s9"
            shift
            ;;
        --s10)
            EXTRA_ARGS="$EXTRA_ARGS --s10"
            shift
            ;;
        --s11)
            EXTRA_ARGS="$EXTRA_ARGS --s11"
            shift
            ;;
        --s12)
            EXTRA_ARGS="$EXTRA_ARGS --s12"
            shift
            ;;
        --headless)
            EXTRA_ARGS="$EXTRA_ARGS --headless"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# --- Checks ---
if [ ! -d "$ISAAC_SIM_PATH" ]; then
    echo "Error: Isaac Sim not found at $ISAAC_SIM_PATH"
    echo "Please set ISAAC_SIM_PATH to your Isaac Sim installation directory."
    exit 1
fi

if [ -n "$CONDA_PREFIX" ]; then
    echo "CRITICAL WARNING: Running in a Conda environment ($CONDA_PREFIX)."
    echo "This often causes Python version conflicts with Isaac Sim."
    echo "Please run 'conda deactivate' before starting."
    # We don't exit here, just warn, in case the user knows what they are doing.
fi

# --- Environment Setup ---
echo "Setting up environment for Isaac Sim 5.0.0..."

# 1. Clean PYTHONPATH to prevent system ROS 2 (Python 3.12) from crashing Isaac Sim (Python 3.11)
# We save the old one just in case, but for execution we want it clean or Isaac-specific.
export OLD_PYTHONPATH=$PYTHONPATH
# export PYTHONPATH="" 
# echo "Cleared PYTHONPATH to ensure isolation from system Python libraries."

# 2. Detect Internal ROS 2 Bridge Version
BRIDGE_PATH="$ISAAC_SIM_PATH/exts/isaacsim.ros2.bridge"
if [ -d "$BRIDGE_PATH/jazzy" ]; then
    echo "Detected internal ROS 2 Jazzy bridge."
    TARGET_DISTRO="jazzy"
elif [ -d "$BRIDGE_PATH/humble" ]; then
    echo "Detected internal ROS 2 Humble bridge."
    TARGET_DISTRO="humble"
else
    echo "Warning: Could not detect specific ROS 2 bridge version in $BRIDGE_PATH. Defaulting to 'humble'."
    TARGET_DISTRO="humble"
fi

# 3. Set ROS Environment Variables
export ROS_DISTRO=$TARGET_DISTRO
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0

# 4. Multi-user ROS 2 Discovery Settings (Disable Shared Memory for cross-user communication)
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# 5. Update LD_LIBRARY_PATH
# We append the bridge libraries to ensure the internal bridge works correctly.
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$BRIDGE_PATH/$TARGET_DISTRO/lib"
echo "Configured LD_LIBRARY_PATH for $TARGET_DISTRO bridge."

# --- Execution ---
echo "Starting Simulation..."
echo "  > Isaac Sim Path: $ISAAC_SIM_PATH"
echo "  > ROS Distro: $ROS_DISTRO"
echo "  > RMW: $RMW_IMPLEMENTATION"
echo "  > Domain ID: $ROS_DOMAIN_ID"
echo "  > Discovery Range: $ROS_AUTOMATIC_DISCOVERY_RANGE"
echo "  > DDS Transport: $FASTDDS_BUILTIN_TRANSPORTS"
if [ -n "$EXTRA_ARGS" ]; then
    echo "  > Extra Args: $EXTRA_ARGS"
fi

"$PYTHON_EXE" "$MAIN_SCRIPT" $EXTRA_ARGS













