# Lproject_sim/src/core/scene_manager.py
from pxr import UsdLux, Gf, UsdGeom, Sdf, UsdShade, Usd
import numpy as np
import math
import os

# Project root: three levels up from this file (src/core/ -> project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SceneManager:
    def __init__(self, world, config: dict = None):
        self.world = world
        self.stage = world.stage
        self.config = config if config else {}
        self.sun = None
        self.stars = []
        self.setup_lighting()
        self.create_stars()


    def setup_lighting(self):
        """
        달 환경 태양광 설정
        
        달에서의 태양빛 특성:
        - 대기 없음 → 산란 없음 → 순수 태양 스펙트럼
        - 태양 색온도: 5778K (실제)
        - 달 표면에서는 대기 산란이 없어 색온도 변화 없음
        - 극단적 명암 대비 (그림자 영역은 거의 검정)
        
        렌더링 고려사항:
        - Isaac Sim RTX 렌더러에서 색온도 모드 사용 시
          6500K (D65 표준 일광)가 가장 중립적으로 보임
        """
        sun_cfg = self.config.get("sun_light", {})
        dome_cfg = self.config.get("dome_light", {})
        
        self.sun = UsdLux.DistantLight.Define(self.stage, "/World/Sun")
        
        # --- 물리 기반 렌더링(PBR) 속성 설정 ---
        sun_prim = self.sun.GetPrim()
        sun_prim.CreateAttribute("physics:light:enable", Sdf.ValueTypeNames.Bool).Set(True)
        
        # --- 색온도 설정 (YAML에서 읽어오기) ---
        color_temp = sun_cfg.get("color_temperature", 5778.0)
        self.sun.GetEnableColorTemperatureAttr().Set(True)
        self.sun.GetColorTemperatureAttr().Set(color_temp)

        # --- 고도각 설정 ---
        elevation = sun_cfg.get("elevation", 10.0)
        azimuth = sun_cfg.get("azimuth", 180.0)

        # --- 조명 강도 설정 (intensity 직접 사용) ---
        # DistantLight는 고도각에 따른 조도 감쇠를 자동 처리함
        # intensity는 수직 입사 시 기준값
        intensity = sun_cfg.get("intensity", 1500.0)

        print(f"[SceneManager] Sun light: elevation={elevation}°, azimuth={azimuth}°, intensity={intensity:.1f}")
        
        self.sun.GetIntensityAttr().Set(intensity)
        
        # 태양 각지름: 0.53도 (달에서 보는 태양 크기)
        self.sun.GetAngleAttr().Set(0.53)
        
        # --- 고도/방위각 설정 ---
        xform = UsdGeom.Xformable(self.sun)
        xform.ClearXformOpOrder()
        xform.AddRotateZOp().Set(azimuth)
        rotation_x = -(90.0 - elevation)
        xform.AddRotateXOp().Set(rotation_x)

        # ------------------------------------------------------------------
        # Dome Light (희미한 산란광 근사 + 검은 배경)
        # ------------------------------------------------------------------
        # 검은 텍스처를 사용하여 배경은 검정, 조명은 유지
        black_texture = os.path.join(_PROJECT_ROOT, "assets", "Textures", "black_1x1.png")
        
        if dome_cfg.get("enabled", False):
            dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
            shadow_ratio = dome_cfg.get("shadow_ratio", 0.2)
            # DomeLight intensity = Sun intensity * shadow_ratio
            dome_intensity = intensity * shadow_ratio
            dome.GetIntensityAttr().Set(dome_intensity)
            dome.GetColorAttr().Set(Gf.Vec3f(*dome_cfg.get("color", [1.0, 1.0, 1.0])))
            dome.GetTextureFormatAttr().Set("latlong")
            print(f"[SceneManager] DomeLight enabled: intensity={dome_intensity:.2f}, ratio={shadow_ratio}, background=black")
        else:
            # DomeLight 비활성화 시에도 검은 배경은 유지
            dome = UsdLux.DomeLight.Define(self.stage, "/World/BackgroundDome")
            dome.GetIntensityAttr().Set(0.0)
            dome.GetTextureFileAttr().Set(black_texture)
            dome.GetTextureFormatAttr().Set("latlong")
            print("[SceneManager] Background-only DomeLight: black sky, no lighting")

    def create_stars(self):
        """
        달 하늘의 별 생성 (Emissive Sphere 사용 - 조명 효과 없음)
        
        달에서 보는 별 특성:
        - 대기 없음 → 별이 깜빡이지 않음, 점으로 보임
        - 시각적으로만 보이고, 지면 조명에 영향 없음
        """
        stars_cfg = self.config.get("stars", {})
        if not stars_cfg.get("enabled", False):
            print("[SceneManager] Stars disabled")
            return
        
        num_stars = stars_cfg.get("count", 1000)
        min_intensity = stars_cfg.get("min_intensity", 10)
        max_intensity = stars_cfg.get("max_intensity", 100)
        min_angle = stars_cfg.get("min_angle", 0.01)
        max_angle = stars_cfg.get("max_angle", 0.1)
        distance = stars_cfg.get("distance", 1000.0)  # 별까지 거리 (m)
        seed = stars_cfg.get("seed", 42)
        
        np.random.seed(seed)
        
        # 색온도 → RGB 변환 함수 (근사)
        def color_temp_to_rgb(temp):
            """색온도(K)를 RGB로 변환 (Tanner Helland 알고리즘 기반)"""
            temp = temp / 100.0
            if temp <= 66:
                r = 1.0
                g = max(0, min(1, 0.390081579 * math.log(temp) - 0.631841444))
                b = max(0, min(1, 0.543206789 * math.log(temp - 10) - 1.196254089)) if temp > 19 else 0
            else:
                r = max(0, min(1, 1.292936186 * ((temp - 60) ** -0.1332047592)))
                g = max(0, min(1, 1.129890861 * ((temp - 60) ** -0.0755148492)))
                b = 1.0
            return Gf.Vec3f(r, g, b)
        
        # 공통 material 생성 (색상별로 다르게)
        stars_scope = self.stage.DefinePrim("/World/Stars", "Scope")
        
        for i in range(num_stars):
            # 무작위 방향 (구면 좌표계)
            azimuth_rad = np.random.uniform(0, 2 * math.pi)
            elevation_deg = np.random.uniform(0.01, 89.9)  # 수평선 10도 위 ~ 85도
            elevation_rad = math.radians(elevation_deg)
            
            # 구면 좌표 → 직교 좌표 (Z-up)
            x = distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
            y = distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
            z = distance * math.sin(elevation_rad)
            
            # 별 크기 (각도 → 실제 크기)
            angle_deg = np.random.uniform(min_angle, max_angle)
            radius = distance * math.tan(math.radians(angle_deg / 2))
            
            # 밝기 (emissive intensity)
            log_intensity = np.random.uniform(np.log10(min_intensity), np.log10(max_intensity))
            intensity = 10 ** log_intensity
            
            # 색온도
            color_temp = np.random.uniform(3500, 8500)
            color = color_temp_to_rgb(color_temp)
            emissive_color = Gf.Vec3f(color[0] * intensity, color[1] * intensity, color[2] * intensity)
            
            # Sphere 생성
            star_path = f"/World/Stars/Star_{i:04d}"
            star = UsdGeom.Sphere.Define(self.stage, star_path)
            star.GetRadiusAttr().Set(radius)
            
            # 위치 설정
            xform = UsdGeom.Xformable(star)
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            
            # 조명 계산에서 제외 (그림자 안 드리움, 빛 차단 안 함)
            prim = star.GetPrim()
            prim.CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
            prim.CreateAttribute("primvars:invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool).Set(True)
            
            # Emissive Material 생성
            mat_path = f"{star_path}/Mat"
            mat = UsdShade.Material.Define(self.stage, mat_path)
            shader = UsdShade.Shader.Define(self.stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(emissive_color)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            mat.CreateSurfaceOutput().ConnectToSource(shader_out)
            UsdShade.MaterialBindingAPI(prim).Bind(mat)
            
            self.stars.append(star)
        
        print(f"[SceneManager] Stars created: {num_stars} emissive spheres (no lighting effect)")
