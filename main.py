# Lproject_sim/main.py

import sys
import os
import traceback
import argparse

# src 폴더를 Python 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# 명령줄 인자 파싱 (SimulationApp 초기화 전에 수행)
def parse_args():
    parser = argparse.ArgumentParser(description="Lunar Rover Simulation")
    parser.add_argument(
        "--sample-scene", "-s",
        action="store_true",
        help="Load pre-saved sample scene for faster startup"
    )
    parser.add_argument(
        "--scene-path",
        type=str,
        default=None,
        help="Custom path to saved USD scene file"
    )
    parser.add_argument(
        "--save-scene",
        action="store_true",
        help="Save current scene after initialization (for creating sample scenes)"
    )
    parser.add_argument(
        "--save-scene-all",
        action="store_true",
        help="Save 12 scene variants: 3 rock levels x 2 lights x 2 terrain modes (s1~s12)"
    )
    parser.add_argument(
        "--s1",
        action="store_true",
        help="Load scene variant 1: full rocks, bright"
    )
    parser.add_argument(
        "--s2",
        action="store_true",
        help="Load scene variant 2: full rocks, dim"
    )
    parser.add_argument(
        "--s3",
        action="store_true",
        help="Load scene variant 3: half rocks, bright"
    )
    parser.add_argument(
        "--s4",
        action="store_true",
        help="Load scene variant 4: half rocks, dim"
    )
    parser.add_argument(
        "--s5",
        action="store_true",
        help="Load scene variant 5: no rocks, bright"
    )
    parser.add_argument(
        "--s6",
        action="store_true",
        help="Load scene variant 6: no rocks, dim"
    )
    parser.add_argument(
        "--s7",
        action="store_true",
        help="Load scene variant 7: full rocks, bright, no outer terrain"
    )
    parser.add_argument(
        "--s8",
        action="store_true",
        help="Load scene variant 8: full rocks, dim, no outer terrain"
    )
    parser.add_argument(
        "--s9",
        action="store_true",
        help="Load scene variant 9: half rocks, bright, no outer terrain"
    )
    parser.add_argument(
        "--s10",
        action="store_true",
        help="Load scene variant 10: half rocks, dim, no outer terrain"
    )
    parser.add_argument(
        "--s11",
        action="store_true",
        help="Load scene variant 11: no rocks, bright, no outer terrain"
    )
    parser.add_argument(
        "--s12",
        action="store_true",
        help="Load scene variant 12: no rocks, dim, no outer terrain"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)"
    )
    return parser.parse_args()

args = parse_args()

from isaacsim import SimulationApp

# [핵심 1] 로드할 필수 확장 기능 목록을 명확히 정의
REQUIRED_EXTENSIONS = [
    "isaacsim.ros2.bridge",
    "omni.replicator.core",
    "omni.graph.window.action", # [Debug] Disable UI ext temporarily
]

# [핵심 2] SimulationApp 설정을 딕셔너리로 완벽하게 구성
CONFIG = {
    "headless": args.headless,
    "renderer": "RayTracedLighting",
    
    
    "enable_ros_bridge": True,
    "enabled_extensions": REQUIRED_EXTENSIONS
}

# [핵심 4] 위에서 정의한 설정 딕셔너리를 사용하여 SimulationApp을 초기화합니다.
simulation_app = SimulationApp(CONFIG)

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.core.nodes")
# 이제 Isaac Sim이 완전히 초기화된 후 나머지 모듈을 임포트합니다.
from src.core.simulation_runner import SimulationRunner
from src.core.scene_io import SceneIO
import rclpy

def main():
    """시뮬레이션 실행기(Runner)를 생성하고 실행합니다."""
    # ROS 2 Init
    rclpy.init()
    
    try:
        # Sample Scene 모드 확인
        use_sample_scene = (args.sample_scene or 
                           args.s1 or args.s2 or args.s3 or args.s4 or args.s5 or args.s6 or
                           args.s7 or args.s8 or args.s9 or args.s10 or args.s11 or args.s12)
        scene_path = args.scene_path
        save_scene_after_init = args.save_scene
        save_all_variants = args.save_scene_all
        
        # --s1 ~ --s12 옵션에 따라 씬 경로 결정
        for i in range(1, 13):
            if getattr(args, f's{i}', False):
                scene_path = SceneIO.get_scene_variant_path(f's{i}')
                break
        
        if use_sample_scene:
            # 저장된 씬 경로 확인
            if scene_path is None:
                scene_path = SceneIO.get_default_scene_path()
            
            if SceneIO.scene_exists(scene_path):
                print(f"\n{'='*60}")
                print(f"[FAST MODE] Loading pre-saved sample scene...")
                print(f"  Scene path: {scene_path}")
                print(f"{'='*60}\n")
                
                # Sample Scene 모드로 Runner 생성
                runner = SimulationRunner(
                    config_path="config/simulation_config.yaml",
                    sample_scene_path=scene_path
                )
            else:
                print(f"\n{'='*60}")
                print(f"[WARNING] Sample scene not found: {scene_path}")
                print(f"  Falling back to normal initialization...")
                print(f"  To create a sample scene, run with --save-scene flag")
                print(f"{'='*60}\n")
                runner = SimulationRunner(config_path="config/simulation_config.yaml")
        else:
            # 일반 모드
            runner = SimulationRunner(config_path="config/simulation_config.yaml")
        
        # 씬 저장 모드
        if save_scene_after_init:
            print(f"\n{'='*60}")
            print(f"[SAVE MODE] Saving current scene...")
            output_path = scene_path if scene_path else SceneIO.get_default_scene_path()
            runner.save_scene_to_file(output_path)
            print(f"  Scene saved to: {output_path}")
            print(f"  You can now use --sample-scene flag for faster startup!")
            print(f"{'='*60}\n")
        
        # 전체 씬 변형 저장 모드 (3 rock levels × 2 light conditions × 2 terrain modes = 12 scenes)
        if save_all_variants:
            print(f"\n{'='*60}")
            print(f"[SAVE SCENE VARIANTS] Saving 12 scene variants...")
            print(f"  3 rock levels × 2 light conditions × 2 terrain modes")
            
            # 조명 프리셋
            bright = SceneIO.LIGHT_PRESETS["bright"]
            dim = SceneIO.LIGHT_PRESETS["dim"]
            
            # 변형 정의: (variant_key, rock_ratio, light_preset, hide_outer, description)
            variants = [
                # With outer terrain (s1~s6)
                ("s1",  1.0, bright, False, "Full rocks + Bright"),
                ("s2",  1.0, dim,    False, "Full rocks + Dim"),
                ("s3",  0.5, bright, False, "Half rocks + Bright"),
                ("s4",  0.5, dim,    False, "Half rocks + Dim"),
                ("s5",  0.0, bright, False, "No rocks + Bright"),
                ("s6",  0.0, dim,    False, "No rocks + Dim"),
                # Without outer terrain (s7~s12)
                ("s7",  1.0, bright, True,  "Full rocks + Bright + No outer"),
                ("s8",  1.0, dim,    True,  "Full rocks + Dim + No outer"),
                ("s9",  0.5, bright, True,  "Half rocks + Bright + No outer"),
                ("s10", 0.5, dim,    True,  "Half rocks + Dim + No outer"),
                ("s11", 0.0, bright, True,  "No rocks + Bright + No outer"),
                ("s12", 0.0, dim,    True,  "No rocks + Dim + No outer"),
            ]
            
            for key, rock_ratio, light, hide_outer, desc in variants:
                path = SceneIO.get_scene_variant_path(key)
                runner.save_scene_with_variant(
                    path, rock_ratio=rock_ratio,
                    elevation=light["elevation"], azimuth=light["azimuth"],
                    hide_outer_terrain=hide_outer
                )
                print(f"  [{key}] {desc}: {path}")
            
            print(f"\n  Usage:")
            print(f"    === With outer terrain ===")
            print(f"    ./start_simulation.sh --s1   # Full rocks, Bright")
            print(f"    ./start_simulation.sh --s2   # Full rocks, Dim")
            print(f"    ./start_simulation.sh --s3   # Half rocks, Bright")
            print(f"    ./start_simulation.sh --s4   # Half rocks, Dim")
            print(f"    ./start_simulation.sh --s5   # No rocks, Bright")
            print(f"    ./start_simulation.sh --s6   # No rocks, Dim")
            print(f"    === Without outer terrain ===")
            print(f"    ./start_simulation.sh --s7   # Full rocks, Bright, No outer")
            print(f"    ./start_simulation.sh --s8   # Full rocks, Dim, No outer")
            print(f"    ./start_simulation.sh --s9   # Half rocks, Bright, No outer")
            print(f"    ./start_simulation.sh --s10  # Half rocks, Dim, No outer")
            print(f"    ./start_simulation.sh --s11  # No rocks, Bright, No outer")
            print(f"    ./start_simulation.sh --s12  # No rocks, Dim, No outer")
            print(f"{'='*60}\n")
        
        runner.run()
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
        traceback.print_exc()
    finally:
        print("Closing SimulationApp...")
        simulation_app.close()
        # ROS 2 Shutdown
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
