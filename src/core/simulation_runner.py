# Lproject_sim/src/core/simulation_runner.py
import numpy as np
import random
import math
import traceback
from src.core.sim_manager import SimulationManager
from src.core.scene_manager import SceneManager
from src.terrain.terrain_manager import TerrainManager, TerrainConfig, CraterConfig
from src.core.robot_context import RobotContext
from src.physics.physics_manager import PhysicsManager

# [New Imports]
from src.config.config_loader import ConfigLoader, PROJECT_ROOT
from src.rendering.rendering_manager import RenderingManager
from src.sdg.randomizer import SDGRandomizer
from src.sdg.annotator import Annotator
from src.ui.hud import HUD
from scripts.save_dem import save_simulation_dem

# [Stellar Engine Import]
from src.core.stellar_manager import StellarManager, StellarSceneUpdater, StellarConfig, create_stellar_config_from_yaml

# [Scene IO Import]
from src.core.scene_io import SceneIO, save_current_scene

class SimulationRunner:
    def __init__(self, config_path: str = "config/simulation_config.yaml", 
                 sample_scene_path: str = None):
        """
        시뮬레이션 러너 초기화
        
        Args:
            config_path: 설정 파일 경로
            sample_scene_path: 미리 저장된 씬 파일 경로 (None이면 새로 생성)
        """
        self.config_loader = ConfigLoader(config_path)
        self.sim_config = self.config_loader.get_simulation_config()
        self.terrain_config_data = self.config_loader.get_terrain_config()
        self.assets_config = self.config_loader.get_assets_config()
        self.env_config = self.config_loader.get_environment_config()
        self.robots_config = self.config_loader.get_robots_config()
        self.scene_config = self.config_loader.get_scene_config()
        
        # Sample Scene 모드 여부
        self.sample_scene_path = sample_scene_path
        self.using_sample_scene = sample_scene_path is not None
        
        # Physics Config
        gravity = self.sim_config.get("physics_scene", {}).get("gravity", None)
        dt = self.sim_config.get("dt", 1.0/60.0)

        self.sim = SimulationManager(physics_dt=dt, gravity=gravity)
        
        if self.using_sample_scene:
            # Sample Scene 로드 모드
            self._init_from_sample_scene()
        else:
            # 일반 모드: 씬을 처음부터 생성
            self._init_scene_from_scratch()
        
        # Rendering Manager (공통)
        self.render_manager = RenderingManager()
        renderer_settings = self.sim_config.get("renderer", {})
        self._setup_renderer(renderer_settings)
        
        # Physics Manager
        self.physics_manager = PhysicsManager(self.tm, self.env_config, self.assets_config)
        
        # Robots
        self.robots = []
        self._spawn_robots()
        
        # SDG
        self._init_sdg()

        # HUD
        self.hud = HUD() if self.env_config.get("hud", {}).get("enabled", False) else None
        
        # Warm-up Physics
        print("Warming up physics...")
        for _ in range(60):
            self.sim.world.step(render=False)
    
    def _init_scene_from_scratch(self):
        """씬을 처음부터 생성 (기존 로직)"""
        print("[SimulationRunner] Initializing scene from scratch...")
        
        self.scene = SceneManager(self.sim.world, self.scene_config)
        
        # Stellar Engine (천체 위치 자동 계산)
        self.stellar_manager = None
        self.stellar_updater = None
        self._init_stellar_engine()
        
        # Terrain - resolve relative paths from YAML
        _dem_path = self.terrain_config_data.get("files", {}).get("dem", "")
        _mask_path = self.terrain_config_data.get("files", {}).get("mask", "")
        _crater_profiles = self.terrain_config_data.get("crater_profiles_path", "")
        
        self.terrain_cfg = TerrainConfig(
            type=self.terrain_config_data.get("type", "procedural"),
            x_size=self.terrain_config_data.get("x_size", 50),
            y_size=self.terrain_config_data.get("y_size", 50),
            resolution=self.terrain_config_data.get("resolution", 0.5),
            src_resolution=self.terrain_config_data.get("src_resolution", 5.0),
            z_scale=self.terrain_config_data.get("z_scale", 2.0),
            seed=self.terrain_config_data.get("seed", 42),
            dem_path=self.config_loader.resolve_path(_dem_path) if _dem_path else "",
            mask_path=self.config_loader.resolve_path(_mask_path) if _mask_path else "",
            detail_strength=self.terrain_config_data.get("hybrid", {}).get("detail_strength", 0.1),
            detail_scale=self.terrain_config_data.get("hybrid", {}).get("detail_scale", 5.0), 
            crop_center_meters=self.terrain_config_data.get("files", {}).get("crop_center_meters", [0.0, 0.0])
        )
        
        self.crater_cfg = CraterConfig(
            profiles_path=self.config_loader.resolve_path(_crater_profiles) if _crater_profiles else ""
        )
        
        # Resolve relative paths in asset config
        self.asset_config = self.config_loader.get_assets_config()
        for _key in ("material_path", "robots_dir"):
            if _key in self.asset_config and self.asset_config[_key]:
                self.asset_config[_key] = self.config_loader.resolve_path(self.asset_config[_key])
        
        # 외곽 지형 설정
        outer_terrain_cfg = self.terrain_config_data.get("outer_terrain", {})

        self.tm = TerrainManager(self.sim.world, self.terrain_cfg, self.asset_config, self.crater_cfg, outer_terrain_cfg)

        # 로버 시작 위치 추출 (바위 생성 제외 영역용)
        robot_positions = []
        for r_cfg in self.robots_config:
            pos = r_cfg.get("position", [0, 0, 0])
            robot_positions.append((pos[0], pos[1]))
        
        _rock_dir = self.terrain_config_data.get("rock_assets_dir", "")
        self.tm.scatter_rocks(
            rock_assets_dir=self.config_loader.resolve_path(_rock_dir) if _rock_dir else "",
            num_rocks=self.terrain_config_data.get("num_rocks", 20),
            excluded_positions=robot_positions,
            exclusion_radius=5.0
        )
    
    def _init_from_sample_scene(self):
        """미리 저장된 샘플 씬에서 로드"""
        import time
        start_time = time.time()
        
        print(f"[SimulationRunner] Loading sample scene from: {self.sample_scene_path}")
        
        # USD 파일 로드
        import omni.usd
        from isaacsim.core.api.world import World
        
        # 기존 World를 닫고 새 Stage로 열기
        omni.usd.get_context().open_stage(self.sample_scene_path)
        
        # Stage가 로드될 때까지 대기
        while not omni.usd.get_context().get_stage():
            time.sleep(0.1)
        
        # World 재초기화 (새 Stage로)
        gravity = self.sim_config.get("physics_scene", {}).get("gravity", None)
        dt = self.sim_config.get("dt", 1.0/60.0)
        
        # 새로운 World 객체 생성
        self.sim.world = World(stage_units_in_meters=1.0, physics_dt=dt)
        
        # Gravity 설정
        if gravity:
            self.sim.world.get_physics_context().set_gravity(gravity[2])
            from pxr import UsdPhysics
            stage = self.sim.world.stage
            scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
            if scene_prim and scene_prim.IsValid():
                scene_api = UsdPhysics.Scene(scene_prim)
                scene_api.CreateGravityDirectionAttr().Set(tuple(gravity))
            print(f"[SimulationRunner] Applied Gravity: {gravity}")
        
        # SceneManager는 생성하지만 조명은 이미 씬에 포함됨
        self.scene = SceneManager.__new__(SceneManager)
        self.scene.world = self.sim.world
        self.scene.stage = self.sim.world.stage
        self.scene.config = self.scene_config
        self.scene.sun = None
        self.scene.stars = []
        
        # 기존 Sun Prim 찾기
        sun_prim = self.sim.world.stage.GetPrimAtPath("/World/Sun")
        if sun_prim.IsValid():
            from pxr import UsdLux
            self.scene.sun = UsdLux.DistantLight(sun_prim)
            print("  > Found existing Sun light in scene")
        
        # Stellar Engine 초기화
        self.stellar_manager = None
        self.stellar_updater = None
        self._init_stellar_engine()
        
        # TerrainManager (로드된 씬의 지형 참조만)
        self.terrain_cfg = TerrainConfig(
            type="loaded",  # 로드된 씬 표시
            x_size=self.terrain_config_data.get("x_size", 50),
            y_size=self.terrain_config_data.get("y_size", 50),
            resolution=self.terrain_config_data.get("resolution", 0.5),
            src_resolution=self.terrain_config_data.get("src_resolution", 5.0),
            z_scale=self.terrain_config_data.get("z_scale", 2.0),
            seed=self.terrain_config_data.get("seed", 42),
        )
        self.crater_cfg = None
        self.asset_config = self.config_loader.get_assets_config()
        
        # TerrainManager를 로드 모드로 생성
        self.tm = TerrainManager.__new__(TerrainManager)
        self.tm.world = self.sim.world
        self.tm.stage = self.sim.world.stage
        self.tm.cfg = self.terrain_cfg
        self.tm.asset_cfg = self.asset_config
        self.tm.prim_path = "/World/Terrain"
        self.tm.rock_paths = []
        self.tm.grid_width = int(self.terrain_cfg.x_size / self.terrain_cfg.resolution)
        self.tm.grid_height = int(self.terrain_cfg.y_size / self.terrain_cfg.resolution)
        self.tm.x_offset = -self.terrain_cfg.x_size / 2.0
        self.tm.y_offset = -self.terrain_cfg.y_size / 2.0
        
        # 기존 지형 메시 참조
        terrain_prim = self.sim.world.stage.GetPrimAtPath("/World/Terrain")
        if terrain_prim.IsValid():
            from pxr import UsdGeom
            self.tm.terrain_mesh = UsdGeom.Mesh(terrain_prim)
            print("  > Found existing terrain mesh in scene")
            
            # 지형 높이맵 복원 시도
            points = self.tm.terrain_mesh.GetPointsAttr().Get()
            if points:
                vertices_np = np.array(points, dtype=np.float32)
                self.tm.current_vertices_np = vertices_np.copy()
                
                # DEM 복원
                dem = vertices_np[:, 2].reshape((self.tm.grid_height, self.tm.grid_width))
                self.tm.current_dem = np.flip(dem, 0)
                self.tm.rock_dem = np.zeros_like(self.tm.current_dem)
                print(f"  > Restored DEM: {self.tm.current_dem.shape}")
        else:
            print("  > Warning: Terrain mesh not found in scene!")
            self.tm.terrain_mesh = None
            self.tm.current_dem = np.zeros((self.tm.grid_height, self.tm.grid_width))
            self.tm.rock_dem = np.zeros_like(self.tm.current_dem)
            self.tm.current_vertices_np = None
        
        # 바위 경로 수집
        rocks_prim = self.sim.world.stage.GetPrimAtPath("/World/Rocks")
        if rocks_prim.IsValid():
            for child in rocks_prim.GetChildren():
                self.tm.rock_paths.append(str(child.GetPath()))
            print(f"  > Found {len(self.tm.rock_paths)} rocks in scene")
        
        # 로드에 필요한 TerrainManager 메서드 추가
        self.tm.outer_terrain_cfg = {}
        self.tm.generator = None
        self.tm.crater_gen = None
        
        elapsed = time.time() - start_time
        print(f"[SimulationRunner] Sample scene loaded in {elapsed:.2f} seconds!")
    
    def _setup_renderer(self, renderer_settings):
        """렌더러 설정"""
        # Lens Flare 설정
        flare_config = renderer_settings.get("lens_flare", {})
        if flare_config.get("enabled", False):
            self.render_manager.enable_lens_flare(True)
            self.render_manager.set_lens_flare_params(
                scale=flare_config.get("scale", 5.0), 
                blades=flare_config.get("blades", 9)
            )
            print("Lens Flare enabled.")

        # Motion Blur 설정
        motion_blur_config = renderer_settings.get("motion_blur", {})
        if motion_blur_config.get("enabled", False):
            self.render_manager.enable_motion_blur(True)
            print("Motion Blur enabled.")

        # DLSS 설정
        dlss_config = renderer_settings.get("dlss", {})
        if dlss_config.get("enabled", False):
            self.render_manager.enable_dlss(True)
            print("DLSS enabled.")
    
    def _init_sdg(self):
        """SDG (Synthetic Data Generation) 초기화"""
        self.randomizer = None
        self.annotator = None
        self._sdg_interval = 100
        self._last_sdg_step = 0
        
        if self.env_config.get("sdg", {}).get("enabled", False):
            try:
                print("Initializing SDG...")
                self._sdg_interval = self.env_config.get("sdg", {}).get("interval", 100)
                
                self.randomizer = SDGRandomizer(self.scene, self.tm, self.config_loader.get_assets_config())
                self.randomizer.setup_graph(interval=self._sdg_interval)
                
                import os as _os
                _sdg_output_dir = _os.path.join(PROJECT_ROOT, "data", "sdg_output")
                from src.sdg.annotator import AnnotatorConfig
                annotator_config = AnnotatorConfig(
                    output_dir=_sdg_output_dir,
                    rgb=True,
                    depth=True,
                    semantic_segmentation=True,
                    bounding_box_2d_tight=True,
                    resolution=(1280, 720)
                )
                self.annotator = Annotator(
                    output_dir=_sdg_output_dir,
                    config=annotator_config
                )
                
                # Annotate first robot's camera if available
                camera_path = None
                if self.robots:
                    r = self.robots[0]
                    if "stereo_camera" in r.sensors:
                        try:
                            camera_path = r.sensors["stereo_camera"].left_camera.camera.prim_path
                        except:
                            pass
                    # Fallback: 로봇의 prim_path에서 카메라 찾기
                    if not camera_path:
                        potential_paths = [
                            f"/World/{r.name}/d455/RSD455/Camera_OmniVision_OV9782_Left",
                            f"/World/{r.name}/base_link/StereoCamera_Left/Camera",
                            f"/World/{r.name}/StereoCamera_Left/Camera",
                            f"/World/{r.name}/front_camera/Camera",
                        ]
                        for path in potential_paths:
                            if self.sim.world.stage.GetPrimAtPath(path).IsValid():
                                camera_path = path
                                print(f"[SDG] Found camera at: {camera_path}")
                                break
                                
                if camera_path:
                    self.annotator.setup(camera_path, interval=self._sdg_interval)
                    self.annotator.setup_semantic_labels()
                    print(f"[SDG] Annotator initialized with camera: {camera_path}")
                else:
                    print("[SDG] Warning: No camera found for annotator.")
                    
            except Exception as e:
                print(f"Error initializing SDG: {e}")
                traceback.print_exc()
                self.randomizer = None
                self.annotator = None
    
    def save_scene_to_file(self, output_path: str = None) -> str:
        """
        현재 씬(지형, 바위, 조명 등)을 USD 파일로 저장
        
        Args:
            output_path: 저장 경로 (None이면 기본 경로 사용)
            
        Returns:
            저장된 파일 경로
        """
        scene_io = SceneIO(self.sim.world.stage)
        return scene_io.save_scene(output_path, include_robots=False, flatten=True)
    
    def save_scene_with_rock_ratio(self, output_path: str, ratio: float) -> str:
        """
        지정된 비율의 바위만 포함한 씬을 저장
        
        Args:
            output_path: 저장 경로
            ratio: 바위 비율 (0.0 = 없음, 0.5 = 절반, 1.0 = 전체)
            
        Returns:
            저장된 파일 경로
        """
        import random
        from pxr import Sdf
        
        stage = self.sim.world.stage
        rocks_prim = stage.GetPrimAtPath("/World/Rocks")
        
        if not rocks_prim.IsValid():
            print(f"  Warning: No rocks found in scene")
            return self.save_scene_to_file(output_path)
        
        # 모든 바위 수집
        all_rocks = [child for child in rocks_prim.GetChildren()]
        total_rocks = len(all_rocks)
        
        if total_rocks == 0:
            return self.save_scene_to_file(output_path)
        
        # 유지할 바위 수 계산
        keep_count = int(total_rocks * ratio)
        
        # 랜덤하게 삭제할 바위 선택 (고정 시드로 재현 가능)
        random.seed(42)
        rocks_to_remove = random.sample(all_rocks, total_rocks - keep_count)
        
        # 임시로 바위 숨기기 (visibility)
        hidden_rocks = []
        for rock in rocks_to_remove:
            try:
                vis_attr = rock.GetAttribute("visibility")
                if vis_attr:
                    old_vis = vis_attr.Get()
                    vis_attr.Set("invisible")
                    hidden_rocks.append((rock, old_vis))
                else:
                    rock.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("invisible")
                    hidden_rocks.append((rock, "inherited"))
            except Exception as e:
                print(f"  Warning: Could not hide rock {rock.GetPath()}: {e}")
        
        print(f"  Rocks: {keep_count}/{total_rocks} kept ({ratio*100:.0f}%)")
        
        # 씬 저장
        scene_io = SceneIO(stage)
        result = scene_io.save_scene(output_path, include_robots=False, flatten=True)
        
        # 바위 visibility 복원
        for rock, old_vis in hidden_rocks:
            try:
                rock.GetAttribute("visibility").Set(old_vis)
            except Exception:
                pass
        
        return result
    
    def set_sun_angles(self, elevation: float, azimuth: float):
        """
        태양 고도각과 방위각 설정
        
        Args:
            elevation: 고도각 (0° = 지평선, 90° = 천정)
            azimuth: 방위각 (0° = 북, 90° = 동, 180° = 남)
        """
        from pxr import UsdGeom
        
        sun_prim = self.sim.world.stage.GetPrimAtPath("/World/Sun")
        if not sun_prim.IsValid():
            print("  Warning: Sun prim not found")
            return
        
        xform = UsdGeom.Xformable(sun_prim)
        xform.ClearXformOpOrder()
        xform.AddRotateZOp().Set(azimuth)
        rotation_x = -(90.0 - elevation)
        xform.AddRotateXOp().Set(rotation_x)
        
        print(f"  Sun angles set: elevation={elevation}°, azimuth={azimuth}°")
    
    def save_scene_with_variant(self, output_path: str, rock_ratio: float, 
                                 elevation: float, azimuth: float,
                                 hide_outer_terrain: bool = False) -> str:
        """
        지정된 바위 비율과 조명 조건으로 씬 저장
        
        Args:
            output_path: 저장 경로
            rock_ratio: 바위 비율 (0.0 ~ 1.0)
            elevation: 태양 고도각
            azimuth: 태양 방위각
            hide_outer_terrain: True이면 외곽 지형/수평면 제거 후 저장
            
        Returns:
            저장된 파일 경로
        """
        import random
        from pxr import Sdf
        
        stage = self.sim.world.stage
        
        # 1. 태양 각도 설정
        self.set_sun_angles(elevation, azimuth)
        
        # 2. 바위 visibility 조정
        rocks_prim = stage.GetPrimAtPath("/World/Rocks")
        hidden_rocks = []
        
        if rocks_prim.IsValid():
            all_rocks = [child for child in rocks_prim.GetChildren()]
            total_rocks = len(all_rocks)
            keep_count = int(total_rocks * rock_ratio)
            
            random.seed(42)
            rocks_to_remove = random.sample(all_rocks, total_rocks - keep_count)
            
            for rock in rocks_to_remove:
                try:
                    vis_attr = rock.GetAttribute("visibility")
                    if vis_attr:
                        old_vis = vis_attr.Get()
                        vis_attr.Set("invisible")
                        hidden_rocks.append((rock, old_vis))
                    else:
                        rock.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("invisible")
                        hidden_rocks.append((rock, "inherited"))
                except Exception:
                    pass
            
            print(f"  Rocks: {keep_count}/{total_rocks} ({rock_ratio*100:.0f}%)")
        
        # 3. 외곽 지형 숨기기 (필요 시)
        hidden_outer = []
        if hide_outer_terrain:
            outer_paths = ["/World/OuterTerrain", "/World/HorizonPlane"]
            for path in outer_paths:
                prim = stage.GetPrimAtPath(path)
                if prim.IsValid():
                    vis_attr = prim.GetAttribute("visibility")
                    if vis_attr:
                        old_vis = vis_attr.Get()
                        vis_attr.Set("invisible")
                        hidden_outer.append((prim, old_vis))
                    else:
                        prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("invisible")
                        hidden_outer.append((prim, "inherited"))
            if hidden_outer:
                print(f"  Outer terrain: HIDDEN ({len(hidden_outer)} prims)")
        
        # 4. 씬 저장
        scene_io = SceneIO(stage)
        result = scene_io.save_scene(output_path, include_robots=False, flatten=True)
        
        # 5. 바위 visibility 복원
        for rock, old_vis in hidden_rocks:
            try:
                rock.GetAttribute("visibility").Set(old_vis)
            except Exception:
                pass
        
        # 6. 외곽 지형 visibility 복원
        for prim, old_vis in hidden_outer:
            try:
                prim.GetAttribute("visibility").Set(old_vis)
            except Exception:
                pass
        
        return result
        
    def _spawn_robots(self):
        # Propagate bridge type to environment config for robots to see
        self.env_config["ros_bridge_type"] = self.sim_config.get("ros_bridge_type", "native")
        
        for r_cfg in self.robots_config:
            try:
                robot_ctx = RobotContext(r_cfg, self.env_config, self.tm)
                self.robots.append(robot_ctx)
                print(f"Spawned robot: {robot_ctx.name}")
            except Exception as e:
                print(f"Error spawning robot {r_cfg.get('name')}: {e}")
                traceback.print_exc()
    
    def _setup_keyboard_input(self):
        """
        키보드 입력 설정 (Isaac Sim carb.input)
        R: 모든 로버를 초기 위치로 리셋
        """
        try:
            import carb.input
            self._input = carb.input.acquire_input_interface()
            self._appwindow = None
            try:
                import omni.appwindow
                self._appwindow = omni.appwindow.get_default_app_window()
                self._keyboard = self._appwindow.get_keyboard()
                self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                    self._keyboard, self._on_keyboard_event
                )
                print("[SimulationRunner] Keyboard shortcuts enabled:")
                print("  R - Reset all rovers to initial position")
            except Exception as e:
                print(f"[SimulationRunner] Keyboard input setup failed: {e}")
                self._keyboard_sub = None
        except ImportError:
            print("[SimulationRunner] carb.input not available, keyboard shortcuts disabled")
            self._input = None
            self._keyboard_sub = None
        
        self._reset_rovers_requested = False
    
    def _on_keyboard_event(self, event, *args, **kwargs):
        """Isaac Sim 키보드 이벤트 콜백"""
        import carb.input
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                self._reset_rovers_requested = True
                print("[Keyboard] Reset rovers requested (R key)")
        return True
    
    def reset_all_rovers(self):
        """
        모든 로버를 초기 생성 위치로 복구합니다.
        키보드(R 키) 또는 ROS2 서비스/토픽으로 호출 가능합니다.
        
        ROS2 사용법:
          # 초기 위치로 리셋
          ros2 service call /{robot_name}/reset_pose std_srvs/srv/Trigger
          ros2 topic pub --once /{robot_name}/reset_pose_cmd std_msgs/msg/Empty
          
          # 지정 위치로 리셋
          ros2 topic pub --once /{robot_name}/reset_pose_target geometry_msgs/msg/PoseStamped \
            "{header: {frame_id: 'world'}, pose: {position: {x: 1.0, y: 2.0, z: 0.5}, orientation: {w: 1.0}}}"
          
          # 커스텀 타겟 설정 (이후 R키/Trigger 시 해당 위치로 리셋)
          ros2 topic pub --once /{robot_name}/set_reset_target geometry_msgs/msg/PoseStamped \
            "{header: {frame_id: 'world'}, pose: {position: {x: 5.0, y: 3.0, z: 0.5}, orientation: {w: 1.0}}}"
        """
        print("\n" + "=" * 50)
        print("[SimulationRunner] Resetting all rovers...")
        for robot in self.robots:
            robot.reset_pose()
        print("=" * 50 + "\n")
    
    def _check_reset_requests(self):
        """
        키보드 및 ROS2 리셋 요청을 확인하고 처리합니다.
        """
        # 키보드로 전체 리셋 요청 (초기 위치로)
        if self._reset_rovers_requested:
            self._reset_rovers_requested = False
            self.reset_all_rovers()
            return
        
        # ROS2로 개별 로버 리셋 요청 (초기 위치 또는 커스텀 위치)
        for robot in self.robots:
            if robot.ros2:
                requested, target_pose = robot.ros2.is_reset_requested()
                if requested:
                    if target_pose is not None:
                        position, orientation = target_pose
                        print(f"\n[ROS2] Reset {robot.name} to custom pose: {position}")
                        robot.reset_pose(position=position, orientation=orientation)
                    else:
                        print(f"\n[ROS2] Reset {robot.name} to initial/custom target pose")
                        robot.reset_pose()
    
    def _init_stellar_engine(self):
        """
        천체 엔진 초기화 - 달 표면에서 태양 위치 자동 계산
        """
        stellar_cfg = self.scene_config.get("stellar", {})
        
        if not stellar_cfg.get("enabled", False):
            print("[SimulationRunner] Stellar engine disabled")
            return
        
        try:
            # StellarConfig 생성
            config = create_stellar_config_from_yaml(self.scene_config)
            
            # StellarManager 생성
            import os as _os
            assets_dir = self.assets_config.get("assets_dir", _os.path.join(PROJECT_ROOT, "assets"))
            if not _os.path.isabs(assets_dir):
                assets_dir = self.config_loader.resolve_path(assets_dir)
            self.stellar_manager = StellarManager(config, assets_dir)
            
            # StellarSceneUpdater 생성 (USD Scene과 연결)
            self.stellar_updater = StellarSceneUpdater(
                self.sim.world.stage,
                self.stellar_manager,
                sun_prim_path="/World/Sun"
            )
            
            # 초기 태양 위치 업데이트
            self.stellar_updater.force_update()
            
            alt, az, _ = self.stellar_manager.get_sun_position()
            print(f"[SimulationRunner] Stellar engine initialized:")
            print(f"  - Location: lat={config.latitude}°, lon={config.longitude}°")
            print(f"  - Time: {self.stellar_manager.get_current_time_str()}")
            print(f"  - Time scale: {config.time_scale}x")
            print(f"  - Sun position: altitude={alt:.1f}°, azimuth={az:.1f}°")
            print(f"  - Sun visible: {'Yes ☀️' if self.stellar_manager.is_sun_visible() else 'No 🌑'}")
            
        except Exception as e:
            print(f"[SimulationRunner] Error initializing stellar engine: {e}")
            traceback.print_exc()
            self.stellar_manager = None
            self.stellar_updater = None

    def run(self):
        self.sim.world.reset()
        for r in self.robots:
            r.initialize()
        
        # 키보드 입력 설정 (R: 리셋)
        self._setup_keyboard_input()
        
        # 시뮬레이션 시작 시 Ground Truth DEM 자동 저장
        print("\n" + "=" * 50)
        print("[SimulationRunner] Saving Ground Truth DEM at simulation start...")
        self.save_dem()
        print("=" * 50 + "\n")
            
        print("Starting simulation loop...")
        step_count = 0
        
        while self.sim.is_running():
            step_count += 1
            dt = 1.0/60.0 # Fixed step
            
            try:
                # [Pause Check] Only run logic if simulation is playing
                if self.sim.is_playing():
                    # 로버 리셋 요청 확인 (키보드 R 키 또는 ROS2 서비스/토픽)
                    self._check_reset_requests()
                    
                    # SDG - 태양 + 바위 랜덤화 (지형 높이에 맞춤)
                    if self.randomizer and step_count > 0 and step_count % self._sdg_interval == 0:
                        try:
                            rover_positions = []
                            for robot in self.robots:
                                try:
                                    pos, _ = robot.rover.get_world_pose()
                                    if pos is not None:
                                        rover_positions.append(pos)
                                except Exception:
                                    continue
                            self.randomizer.randomize_all(rover_positions=rover_positions)
                            self._last_sdg_step = step_count
                        except Exception as sdg_e:
                            print(f"[SDG] Randomization error: {sdg_e}")
                        
                self.sim.step()
                
                if self.sim.is_playing():
                    # Stellar Engine Update (태양 위치 자동 갱신)
                    # SDG 모드에서는 태양 위치를 randomizer가 제어하므로 stellar engine 비활성화
                    if self.stellar_updater and self.stellar_manager and not self.randomizer:
                        try:
                            updated = self.stellar_updater.update(dt)
                            # HUD에 시간 정보 전달을 위해 주기적 로그 (1000 스텝마다)
                            if updated and step_count % 1000 == 0:
                                info = self.stellar_manager.get_info()
                        except Exception as stellar_e:
                            if step_count % 1000 == 0:  # 에러 로그 스로틀링
                                print(f"[Stellar] Update error: {stellar_e}")
                    
                    # Physics Update (Terramechanics, Deform, Dust)
                    self.physics_manager.update(dt, self.robots)
                    
                    # Robot Context Update (Energy, Sensors, ROS publishing)
                    for robot in self.robots:
                        robot.update(dt, step_count)
                        
                    # HUD Update (First robot only)
                    if self.hud and self.robots:
                        r = self.robots[0]
                        vel = np.linalg.norm(r.rover.get_linear_velocity())
                        
                        # 로버 위치 및 heading
                        pos, ori = r.rover.get_world_pose()
                        # Quaternion to yaw (heading)
                        import math
                        w, x, y, z = ori[0], ori[1], ori[2], ori[3]
                        
                        # Yaw (Z-axis)
                        siny_cosp = 2 * (w * z + x * y)
                        cosy_cosp = 1 - 2 * (y * y + z * z)
                        yaw_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))

                        # Roll (X-axis)
                        sinr_cosp = 2 * (w * x + y * z)
                        cosr_cosp = 1 - 2 * (x * x + y * y)
                        roll_deg = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

                        # Pitch (Y-axis)
                        sinp = 2 * (w * y - z * x)
                        if abs(sinp) >= 1:
                            pitch_deg = math.degrees(math.copysign(math.pi / 2, sinp))
                        else:
                            pitch_deg = math.degrees(math.asin(sinp))
                        
                        sun = [0,0,1]
                        if "sun_sensor" in r.sensors:
                            try:
                                sun = r.sensors["sun_sensor"].get_sun_vector()
                            except: pass
                            
                        batt = 50.0  # Default 50%
                        batt_wh = 50.0
                        batt_cap = 100.0
                        is_charging = False
                        solar_power = 0.0
                        net_power = 0.0
                        temp = 20.0
                        incidence = 0.0
                        panel_eff = 1.0
                        
                        if "energy_manager" in r.components:
                            try:
                                status = r.components["energy_manager"].get_status()
                                batt = status["percentage"]
                                batt_wh = status.get("charge_wh", 50.0)
                                batt_cap = status.get("capacity_wh", 100.0)
                                solar_power = status.get("solar_power_w", 0.0)
                                net_power = status.get("net_power_w", 0.0)
                                temp = status.get("temperature", 20.0)
                                # sun_incidence는 cos(theta) 값 (0~1), 각도로 변환
                                sun_incidence = status.get("sun_incidence", 0.0)
                                incidence = np.degrees(np.arccos(np.clip(sun_incidence, 0, 1))) if sun_incidence > 0 else 90.0
                                is_charging = net_power > 0
                            except Exception as e:
                                print(f"[HUD] Error getting battery: {e}")
                        else:
                            # energy_manager가 없으면 한 번만 경고
                            if step_count == 1:
                                print(f"[HUD] Warning: energy_manager not found in robot components")
                                print(f"[HUD] Available components: {list(r.components.keys())}")
                                
                        if "solar_panel" in r.components:
                            try:
                                panel_eff = r.components["solar_panel"].get_efficiency_factor()
                            except: pass

                        # Shadow detection status
                        in_shadow = False
                        if "sun_sensor" in r.sensors:
                            try:
                                in_shadow = r.sensors["sun_sensor"].is_in_shadow()
                            except: pass
                            
                        # Communication Latency
                        latency_ms = 0.0
                        signal = 1.0
                        if r.latency_manager:
                            stats = r.latency_manager.get_statistics()
                            latency_ms = r.latency_manager.delay_seconds * 1000
                            
                        # Physics Info (Warp Enabled?)
                        # Import here or get from manager
                        from src.config.physics_config import TerrainMechanicalParameter
                        tp = TerrainMechanicalParameter()
                        phy_info = f"Warp (c={tp.c:.0f}Pa)"
                        
                        # Stellar Info (달 시뮬레이션 시간, 태양 위치)
                        stellar_time = ""
                        sun_altitude = 0.0
                        sun_azimuth = 0.0
                        sun_visible = True
                        if self.stellar_manager:
                            try:
                                stellar_info = self.stellar_manager.get_info()
                                stellar_time = stellar_info.get("time", "")
                                sun_altitude = stellar_info.get("sun_altitude", 0.0)
                                sun_azimuth = stellar_info.get("sun_azimuth", 0.0)
                                sun_visible = stellar_info.get("sun_visible", True)
                            except: pass
                        
                        # HUD 업데이트 (확장된 데이터)
                        self.hud.update_from_dict({
                            "battery_percent": batt,
                            "battery_wh": batt_wh,
                            "battery_capacity": batt_cap,
                            "is_charging": is_charging,
                            "solar_power_w": solar_power,
                            "solar_incidence_deg": incidence,
                            "panel_efficiency": panel_eff,
                            "sun_vector": tuple(sun),
                            "in_shadow": in_shadow,
                            "net_power_w": net_power,
                            "power_consumption_w": solar_power - net_power,
                            "temperature_c": temp,
                            "speed_ms": vel,
                            "position": tuple(pos),
                            "yaw_deg": yaw_deg,
                            "roll_deg": roll_deg,
                            "pitch_deg": pitch_deg,
                            "latency_ms": latency_ms,
                            "signal_strength": signal,
                            "physics_info": phy_info,
                            "sim_time": step_count * dt,
                            "step_count": step_count,
                            # Stellar 정보
                            "stellar_time": stellar_time,
                            "sun_altitude_deg": sun_altitude,
                            "sun_azimuth_deg": sun_azimuth,
                            "sun_visible": sun_visible
                        })
            except Exception as e:
                print(f"Error in simulation loop (Step {step_count}): {e}")
                traceback.print_exc()
                # Optional: break or continue? Continue is safer for long runs, but might spam log.
                # Let's continue but maybe throttle errors if needed.
                pass
                
        self._cleanup()

    def _cleanup(self):
        # 키보드 구독 해제
        if hasattr(self, '_keyboard_sub') and self._keyboard_sub is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            except Exception:
                pass
        if self.hud: self.hud.destroy()
        if self.physics_manager: self.physics_manager.cleanup()
        for r in self.robots:
            r.shutdown()
        self.sim.close()

    def save_dem(self, output_dir: str = None, prefix: str = "sim", 
                 pointcloud_subsample: int = None, max_rock_points_per_rock: int = None):
        """
        시뮬레이션 환경의 전체 DEM을 저장합니다.
        Rock을 포함한 전체 지형 높이맵을 저장하여 nvblox 3D map과 비교 가능.
        
        Args:
            output_dir: 저장 디렉토리 (None이면 data/dem_exports)
            prefix: 파일명 접두사
            pointcloud_subsample: 지형 포인트 서브샘플링 (1=전체, 2=1/4, 4=1/16 점)
            max_rock_points_per_rock: Rock당 최대 점 개수
            
        Returns:
            dict: 저장된 파일 경로들
        """
        try:
            # config에서 기본값 가져오기
            dem_export_cfg = self.env_config.get("dem_export", {})
            if pointcloud_subsample is None:
                pointcloud_subsample = dem_export_cfg.get("pointcloud_subsample", 1)
            if max_rock_points_per_rock is None:
                max_rock_points_per_rock = dem_export_cfg.get("max_rock_points_per_rock", None)
            
            stage = self.sim.world.stage
            result = save_simulation_dem(
                self.tm, stage, output_dir, prefix,
                pointcloud_subsample=pointcloud_subsample,
                max_rock_points_per_rock=max_rock_points_per_rock
            )
            print(f"[SimulationRunner] DEM saved successfully!")
            return result
        except Exception as e:
            print(f"[SimulationRunner] Error saving DEM: {e}")
            traceback.print_exc()
            return None
