# Lproject_sim/src/sdg/randomizer.py
"""
SDG (Synthetic Data Generation) Randomizer

Omni Replicator를 사용한 도메인 랜덤화:
- 태양 위치/강도 랜덤화
- 바위 위치/회전/스케일 랜덤화 (지형 높이에 맞춤)
- 카메라 노출/화이트밸런스 랜덤화
- 텍스처 랜덤화

Note:
- 바위가 공중에 뜨는 문제는 scatter_2d가 메시 표면을 정확히 찾지 못해서 발생
- 해결: scatter_2d 대신 terrain_manager.get_heights()를 사용한 커스텀 배치
"""
import omni.replicator.core as rep
from pxr import UsdGeom, Gf, Usd
import numpy as np
import os
import glob
import random
from typing import Optional, Dict, List


class SDGRandomizer:
    """
    Handles domain randomization using Omni Replicator.
    
    Features:
    - Sun light randomization (position, intensity, color temperature)
    - Rock placement with terrain height sampling (no floating rocks)
    - Camera exposure randomization
    - Texture randomization (optional)
    """
    
    def __init__(self, scene_manager, terrain_manager, assets_config: dict = None):
        self.scene_manager = scene_manager
        self.terrain_manager = terrain_manager
        self.assets_config = assets_config if assets_config is not None else {}
        self.interval = 100
        
        # 랜덤화 설정
        self.sun_config = {
            "elevation_range": (1, 30),        # 태양 고도 (도, 양수 = 수평선 위)
            "azimuth_range": (0, 360),         # 방위각
            "intensity_range": (500.0, 10000.0), # 강도 (W/m²)
            "color_temp_range": (4500, 6500),  # 색온도 (K)
        }
        
        self.rock_config = {
            "scale_range": (0.2, 4.0),        # 스케일 범위
            "rotation_range": (0, 360),       # Z축 회전
            "position_noise": 0.5,            # XY 위치 노이즈 (m)
            "rover_exclusion_radius": 5.0,    # 로버 주변 배치 금지 반경 (m)
        }
        
        self._stage = None
        self._rock_prims = []
        self._rover_positions: List[tuple] = []
        
    def setup_graph(self, interval: int = 100):
        """
        Sets up the Replicator graph with triggers.
        
        Note: 태양과 바위 랜덤화는 모두 수동(manual) 방식으로 전환됨.
              Replicator 그래프는 Writer trigger 전용으로만 사용.
        
        Args:
            interval: 랜덤화 간격 (프레임 수)
        """
        self.interval = interval
        print(f"[SDGRandomizer] Setup complete. Interval: {interval} frames")
        print(f"  Sun: elevation=[{self.sun_config['elevation_range'][0]}, {self.sun_config['elevation_range'][1]}]°, "
              f"intensity=[{self.sun_config['intensity_range'][0]}, {self.sun_config['intensity_range'][1]}]")
        print(f"  Rocks: scale=[{self.rock_config['scale_range'][0]}, {self.rock_config['scale_range'][1]}], "
              f"rotation=[{self.rock_config['rotation_range'][0]}, {self.rock_config['rotation_range'][1]}]°")
        
    def set_rover_positions(self, rover_positions: List[tuple]):
        """
        로버 월드 좌표를 업데이트
        
        Args:
            rover_positions: (x, y, z) 튜플 리스트
        """
        self._rover_positions = rover_positions or []

    def randomize_all(self, rover_positions: Optional[List[tuple]] = None):
        """
        모든 요소를 한 번에 랜덤화 (수동 호출용)
        """
        if rover_positions is not None:
            self.set_rover_positions(rover_positions)
        self._randomize_sun_manual()
        self._randomize_rocks_manual()
        
    def _randomize_sun_manual(self):
        """
        태양 위치/강도를 직접 USD xformOps로 랜덤화
        
        DistantLight는 RotateZ(azimuth) + RotateX(-(90-elevation)) 구조를 사용.
        Replicator 그래프 방식은 이 구조를 무시하므로 수동 설정 필요.
        """
        try:
            try:
                from isaacsim.core.utils.stage import get_current_stage
            except ImportError:
                from omni.isaac.core.utils.stage import get_current_stage
            from pxr import UsdLux
            
            stage = get_current_stage()
            if not stage:
                return
                
            sun_prim = stage.GetPrimAtPath("/World/Sun")
            if not sun_prim or not sun_prim.IsValid():
                print("[SDGRandomizer] Warning: /World/Sun not found")
                return
            
            # 랜덤 고도/방위각/강도 생성
            el_min, el_max = self.sun_config["elevation_range"]
            az_min, az_max = self.sun_config["azimuth_range"]
            int_min, int_max = self.sun_config["intensity_range"]
            
            elevation = random.uniform(el_min, el_max)
            azimuth = random.uniform(az_min, az_max)
            intensity = random.uniform(int_min, int_max)
            
            # SceneManager/StellarSceneUpdater와 동일한 convention:
            #   RotateZ = azimuth,  RotateX = -(90 - elevation)
            rotation_z = azimuth
            rotation_x = -(90.0 - elevation)
            
            # Xform 업데이트
            xform = UsdGeom.Xformable(sun_prim)
            xform_ops = xform.GetOrderedXformOps()
            
            updated_z = False
            updated_x = False
            if len(xform_ops) >= 2:
                for op in xform_ops:
                    op_name = op.GetOpName()
                    if "rotateZ" in op_name:
                        op.Set(rotation_z)
                        updated_z = True
                    elif "rotateX" in op_name:
                        op.Set(rotation_x)
                        updated_x = True
            
            if not updated_z or not updated_x:
                xform.ClearXformOpOrder()
                xform.AddRotateZOp().Set(rotation_z)
                xform.AddRotateXOp().Set(rotation_x)
            
            # Intensity 업데이트
            sun_light = UsdLux.DistantLight(sun_prim)
            sun_light.GetIntensityAttr().Set(intensity)
            
            # 색온도 랜덤화 (옵션)
            ct_min, ct_max = self.sun_config["color_temp_range"]
            color_temp = random.uniform(ct_min, ct_max)
            sun_light.GetColorTemperatureAttr().Set(color_temp)
            
            print(f"[SDGRandomizer] Sun randomized: elevation={elevation:.1f}°, "
                  f"azimuth={azimuth:.1f}°, intensity={intensity:.0f}, "
                  f"color_temp={color_temp:.0f}K")
                  
        except Exception as e:
            print(f"[SDGRandomizer] Error randomizing sun: {e}")
            
    def _randomize_rocks_manual(self):
        """
        바위를 수동으로 랜덤화 (지형 높이에 맞춤)
        
        scatter_2d는 메시 표면 샘플링이 불안정해서 바위가 공중에 뜨는 문제 발생.
        대신 terrain_manager.get_heights()를 사용해 정확한 높이에 배치.
        """
        if not self.terrain_manager:
            print("[SDGRandomizer] Warning: terrain_manager not available")
            return
            
        try:
            # Isaac Sim 버전에 따라 import 경로가 다름
            try:
                from isaacsim.core.utils.stage import get_current_stage
            except ImportError:
                from omni.isaac.core.utils.stage import get_current_stage
            stage = get_current_stage()
            if not stage:
                print("[SDGRandomizer] Warning: Stage not available")
                return
                
            # Rock prims 가져오기
            rocks_scope = stage.GetPrimAtPath("/World/Rocks")
            if not rocks_scope or not rocks_scope.IsValid():
                return
                
            # 지형 범위
            x_min = self.terrain_manager.x_offset
            x_max = x_min + self.terrain_manager.cfg.x_size
            y_min = self.terrain_manager.y_offset
            y_max = y_min + self.terrain_manager.cfg.y_size
            
            scale_min, scale_max = self.rock_config["scale_range"]
            rot_min, rot_max = self.rock_config["rotation_range"]
            pos_noise = self.rock_config["position_noise"]
            rover_exclusion_radius = self.rock_config.get("rover_exclusion_radius", 0.0)
            
            rock_count = 0
            for prim in rocks_scope.GetChildren():
                if not prim.IsValid():
                    continue
                    
                prim_path = str(prim.GetPath())
                if not prim_path.startswith("/World/Rocks/Rock_"):
                    continue
                    
                try:
                    xformable = UsdGeom.Xformable(prim)
                    if not xformable:
                        continue
                        
                    # 새 랜덤 위치 계산 (로버 주변 배치 금지)
                    max_tries = 20
                    placed = False
                    for _ in range(max_tries):
                        x = random.uniform(x_min + 2, x_max - 2)  # 경계에서 2m 여유
                        y = random.uniform(y_min + 2, y_max - 2)
                        
                        # 위치 노이즈 추가
                        x += random.uniform(-pos_noise, pos_noise)
                        y += random.uniform(-pos_noise, pos_noise)
                        
                        # 로버와 거리 체크 (XY 평면)
                        if rover_exclusion_radius > 0.0 and self._rover_positions:
                            too_close = False
                            for rpos in self._rover_positions:
                                dx = x - float(rpos[0])
                                dy = y - float(rpos[1])
                                if (dx * dx + dy * dy) < (rover_exclusion_radius ** 2):
                                    too_close = True
                                    break
                            if too_close:
                                continue
                        
                        placed = True
                        break
                    
                    if not placed:
                        continue
                    
                    # 지형 높이 샘플링 (핵심!)
                    z = self.terrain_manager.sample_height_at_xy(x, y)
                    
                    # Transform 업데이트
                    xformable.ClearXformOpOrder()
                    xformable.AddTranslateOp().Set(Gf.Vec3f(float(x), float(y), float(z)))
                    
                    # 회전 (Z축)
                    rot_z = random.uniform(rot_min, rot_max)
                    xformable.AddRotateZOp().Set(float(rot_z))
                    
                    # 스케일
                    scale = random.uniform(scale_min, scale_max)
                    xformable.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
                    
                    rock_count += 1
                    
                except Exception as e:
                    print(f"[SDGRandomizer] Error randomizing rock {prim_path}: {e}")
                    continue
                    
            print(f"[SDGRandomizer] Randomized {rock_count} rocks with terrain-snapped positions")
            
        except Exception as e:
            print(f"[SDGRandomizer] Error in manual rock randomization: {e}")
            
    def _randomize_rocks_graph(self):
        """
        [DEPRECATED] Replicator scatter_2d 사용 - 공중에 뜨는 문제 있음
        
        대신 _randomize_rocks_manual() 사용 권장
        """
        rocks = rep.get.prims(path_pattern="/World/Rocks/Rock_*", prim_types=["Xform"])
        terrain = rep.get.prim_at_path("/World/Terrain")
        
        with rocks:
            # scatter_2d는 표면에 배치하지만, 복잡한 메시에서 부정확함
            rep.randomizer.scatter_2d(
                surface_prims=terrain,
                check_for_collisions=False  # 충돌 체크 비활성화 (성능)
            )
            rep.modify.pose(
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                scale=rep.distribution.uniform(0.8, 1.5)
            )
            
    def randomize_camera_exposure(self, camera_path: str = None):
        """
        카메라 노출 랜덤화
        
        Args:
            camera_path: 카메라 prim 경로. None이면 기본 카메라 사용
        """
        if camera_path is None:
            camera_path = "/World/husky_1/base_link/StereoCamera_Left/Camera"
            
        try:
            camera = rep.get.prim_at_path(camera_path)
            with camera:
                # 노출 시간 (셔터 스피드)
                rep.modify.attribute(
                    "exposure",
                    rep.distribution.uniform(0.0, 2.0)
                )
        except Exception as e:
            print(f"[SDGRandomizer] Camera exposure randomization skipped: {e}")
            
    def _setup_texture_randomization(self, shader, textures_dir: str):
        """
        텍스처 랜덤화 설정
        
        Args:
            shader: UsdShade.Shader 객체
            textures_dir: 텍스처 디렉토리 경로
        """
        subdirs = [d for d in glob.glob(os.path.join(textures_dir, "*")) if os.path.isdir(d)]
        if not subdirs:
            return
        
        diff_maps = []
        norm_maps = []
        rough_maps = []
        
        for d in subdirs:
            files = os.listdir(d)
            dm = next((f for f in files if "diff" in f.lower() or "albedo" in f.lower()), None)
            nm = next((f for f in files if "nor" in f.lower() or "normal" in f.lower()), None)
            rm = next((f for f in files if "rough" in f.lower()), None)
            
            if dm: diff_maps.append(os.path.join(d, dm))
            if nm: norm_maps.append(os.path.join(d, nm))
            if rm: rough_maps.append(os.path.join(d, rm))
        
        if diff_maps:
            with shader:
                rep.modify.attribute("inputs:diffuse_texture", rep.distribution.choice(diff_maps))
        if norm_maps:
            with shader:
                rep.modify.attribute("inputs:normal_texture", rep.distribution.choice(norm_maps))
        if rough_maps:
            with shader:
                rep.modify.attribute("inputs:roughness_texture", rep.distribution.choice(rough_maps))
