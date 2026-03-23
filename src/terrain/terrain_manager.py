import numpy as np
import random
import os
import math

from pxr import UsdGeom, Sdf, Gf, UsdPhysics, UsdShade
import omni
from isaacsim.core.utils.stage import add_reference_to_stage

from .terrain_generator import TerrainGenerator, TerrainConfig, CraterConfig

class TerrainManager:
    def __init__(self, world, cfg: TerrainConfig, asset_cfg: dict, crater_cfg: CraterConfig = None, 
                 outer_terrain_cfg: dict = None):
        self.world = world
        self.stage = world.stage
        self.cfg = cfg
        self.asset_cfg = asset_cfg
        self.outer_terrain_cfg = outer_terrain_cfg or {}

        self.prim_path = "/World/Terrain"
        self.rock_paths = []
        
        self.grid_width = int(self.cfg.x_size / self.cfg.resolution)
        self.grid_height = int(self.cfg.y_size / self.cfg.resolution)
        
        self.x_offset = -self.cfg.x_size / 2.0
        self.y_offset = -self.cfg.y_size / 2.0
        
        self._hide_default_ground()
        
        self.generator = TerrainGenerator(cfg, crater_cfg)
        dem = self.generator.generate()
        self.current_dem = dem
        
        self.rock_dem = np.zeros_like(dem)
        
        self._create_mesh_from_dem(dem)
        
        # 외곽 지형 생성 (저해상도 + 산 능선)
        if self.outer_terrain_cfg.get("enabled", True):
            self._create_outer_terrain()

    def _hide_default_ground(self):
        # [수정] Default Ground Plane 완전히 제거 또는 숨기기
        # 숨기기만 하면 충돌이 여전히 발생할 수 있음!
        ground_path = "/World/defaultGroundPlane"
        ground_prim = self.stage.GetPrimAtPath(ground_path)
        if ground_prim.IsValid():
            # 방법 1: 완전히 비활성화 (visibility + collision 제거)
            ground_prim.GetAttribute("visibility").Set("invisible")
            
            # 방법 2: Collision 비활성화 (충돌 방지)
            try:
                # Ground Plane의 모든 자식에서 Collision 제거
                from pxr import Usd
                for child in Usd.PrimRange(ground_prim):
                    collision_api = UsdPhysics.CollisionAPI(child)
                    if collision_api:
                        child.RemoveAPI(UsdPhysics.CollisionAPI)
                print("[TerrainManager] Default ground plane collision disabled")
            except Exception as e:
                print(f"[TerrainManager] Warning: Could not disable ground collision: {e}")

    def _create_mesh_from_dem(self, dem):
        vertices = []
        indices = []
        tex_coords = []
        
        tiling_factor = self.cfg.x_size / 10.0
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                vertices.append(Gf.Vec3f(x * self.cfg.resolution + self.x_offset, 
                                       y * self.cfg.resolution + self.y_offset, 
                                       0.0))
                u = (x / (self.grid_width - 1)) * tiling_factor
                v = (y / (self.grid_height - 1)) * tiling_factor
                tex_coords.append(Gf.Vec2f(u, v))
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if x > 0 and y > 0:
                    p1 = y * self.grid_width + x
                    p2 = (y - 1) * self.grid_width + x
                    p3 = (y - 1) * self.grid_width + (x - 1)
                    p4 = y * self.grid_width + (x - 1)
                    indices.extend([p1, p3, p2, p1, p4, p3])

        vertices_array = np.array(vertices, dtype=np.float32)
        tex_coords_array = np.array(tex_coords, dtype=np.float32)
        
        self.indices = np.array(indices, dtype=np.int32)
        
        if dem.shape != (self.grid_height, self.grid_width):
            import cv2
            dem = cv2.resize(dem, (self.grid_width, self.grid_height), interpolation=cv2.INTER_LINEAR)

        flipped_dem_for_mesh = np.flip(dem, 0)
        vertices_array[:, 2] = flipped_dem_for_mesh.flatten()
        
        # --- [핵심 수정] 수정 가능한 전체 정점 데이터의 복사본을 클래스 변수로 유지합니다. ---
        self.current_vertices_np = vertices_array.copy()

        mesh_prim = UsdGeom.Mesh.Define(self.stage, self.prim_path)
        mesh_prim.GetPointsAttr().Set(self.current_vertices_np) # 초기 설정
        mesh_prim.GetFaceVertexIndicesAttr().Set(self.indices)
        mesh_prim.GetFaceVertexCountsAttr().Set(np.full(len(indices) // 3, 3, dtype=np.int32))
        
        primvar_api = UsdGeom.PrimvarsAPI(mesh_prim)
        pv = primvar_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(tex_coords_array)
        
        self.terrain_mesh = mesh_prim
        
        collision_prim = self.terrain_mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collision_prim)
        physx_collision_api = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
        physx_collision_api.CreateApproximationAttr().Set("none")
        
        # [추가] Contact Offset 설정 - 휠 충돌 안정성 향상
        self._apply_contact_offset(collision_prim)
        
        # [추가] Physics Material 설정 - 지형의 마찰 계수 설정
        self._apply_physics_material(collision_prim)
        
        # [추가] Semantic Label 설정 - SDG용
        self._apply_semantic_label(collision_prim, "terrain")
        
        self._apply_material()

    def _apply_contact_offset(self, collision_prim):
        """
        Contact Offset과 Rest Offset을 설정하여 휠-지형 충돌의 안정성을 높입니다.
        
        Contact Offset: 물체가 접촉하기 전에 충돌 감지를 시작하는 거리
        Rest Offset: 물체가 "접촉 중"으로 간주되는 거리
        
        적절한 값으로 설정하면 급격한 충돌 반응(튀어오름)을 방지합니다.
        """
        try:
            from pxr import PhysxSchema
            
            # PhysxCollisionAPI 적용
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
            
            # Contact Offset: 0.02m (2cm) - 휠이 가까이 오면 미리 접촉 처리
            physx_collision_api.CreateContactOffsetAttr().Set(0.02)
            
            # Rest Offset: 0.005m (5mm) - 안정적인 접촉 상태 유지
            physx_collision_api.CreateRestOffsetAttr().Set(0.005)
            
            print(f"[TerrainManager] Applied Contact Offset: 0.02m, Rest Offset: 0.005m")
            
        except Exception as e:
            print(f"[TerrainManager] Warning: Failed to apply contact offset: {e}")

    def _apply_physics_material(self, collision_prim):
        """
        지형 충돌 메시에 Physics Material을 적용합니다.
        적절한 마찰 계수를 설정하여 로버가 미끄러지지 않고 안정적으로 주행할 수 있게 합니다.
        """
        try:
            # Physics Material 정의
            physics_material_path = "/World/PhysicsMaterials/TerrainMaterial"
            physics_material = UsdShade.Material.Define(self.stage, physics_material_path)
            
            # PhysX Material API 적용
            from pxr import PhysxSchema
            physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(physics_material.GetPrim())
            
            # 마찰 계수 설정 (달 레골리스 기준)
            # Static Friction: 정지 마찰 계수 (움직이기 시작하는 데 필요한 힘)
            # Dynamic Friction: 동적 마찰 계수 (움직이는 동안의 마찰)
            # Restitution: 반발 계수 (충돌 시 튀어오르는 정도)
            
            STATIC_FRICTION = 0.7   # 달 레골리스: 0.6~0.8
            DYNAMIC_FRICTION = 0.6  # 달 레골리스: 0.5~0.7
            RESTITUTION = 0.0       # 거의 튀지 않음
            
            # UsdPhysics.MaterialAPI 사용
            material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
            material_api.CreateStaticFrictionAttr().Set(STATIC_FRICTION)
            material_api.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
            material_api.CreateRestitutionAttr().Set(RESTITUTION)
            
            # 충돌 프림에 Material 바인딩
            binding_api = UsdShade.MaterialBindingAPI.Apply(collision_prim)
            binding_api.Bind(physics_material, UsdShade.Tokens.weakerThanDescendants, "physics")
            
            print(f"[TerrainManager] Applied Physics Material: friction={STATIC_FRICTION}/{DYNAMIC_FRICTION}, restitution={RESTITUTION}")
            
        except Exception as e:
            print(f"[TerrainManager] Warning: Failed to apply physics material: {e}")

    def _apply_semantic_label(self, prim, label: str):
        """
        프림에 Semantic Label을 적용합니다 (SDG Semantic Segmentation용).
        
        PrimSemanticData API를 사용하여 Replicator annotator가 인식할 수 있는
        형식으로 semantic label을 설정합니다.
        
        Args:
            prim: USD Prim
            label: Semantic 라벨 (예: "terrain", "rock")
        """
        try:
            from semantics.schema.editor import PrimSemanticData
            prim_sd = PrimSemanticData(prim)
            prim_sd.add_entry("class", label)
        except Exception as e:
            # Fallback: 직접 Semantics API 적용
            try:
                from pxr import Semantics
                sem_api_name = "Semantics"
                if not prim.HasAPI(Semantics.SemanticsAPI):
                    Semantics.SemanticsAPI.Apply(prim, sem_api_name)
                sem_api = Semantics.SemanticsAPI.Get(prim, sem_api_name)
                sem_api.CreateSemanticTypeAttr().Set("class")
                sem_api.CreateSemanticDataAttr().Set(label)
            except Exception as inner_e:
                pass  # Semantic label 적용 실패 (경고만)

    def _apply_material(self):
        material_path_str = self.asset_cfg.get("material_path")
        mesh_prim_to_bind = self.terrain_mesh.GetPrim()

        if material_path_str and os.path.exists(material_path_str):
            material = UsdShade.Material.Define(self.stage, "/World/Looks/TerrainMaterial")
            shader = UsdShade.Shader.Define(self.stage, "/World/Looks/TerrainMaterial/Shader")
            shader.SetSourceAsset(material_path_str, "mdl")
            shader.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("LunarRegolith8k")
            mdl_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(mdl_output)
            #UsdShade.MaterialBindingAPI.Apply(mesh_prim_to_bind)
            UsdShade.MaterialBindingAPI(mesh_prim_to_bind).Bind(material)
        else:
            print("Warning: Material MDL file not found. Applying a default grey material.")
            material_path = Sdf.Path("/World/Looks/DefaultTerrainMaterial")
            material = UsdShade.Material.Define(self.stage, material_path)
            shader = UsdShade.Shader.Define(self.stage, material_path.AppendPath("Shader"))
            shader.CreateIdAttr().Set("UsdPreviewSurface")
            shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader_output)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.15, 0.15, 0.18))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
            #UsdShade.MaterialBindingAPI.Apply(mesh_prim_to_bind)
            UsdShade.MaterialBindingAPI(mesh_prim_to_bind).Bind(material)

    def _create_outer_terrain(self):
        """
        메인 지형 주변에 저해상도 외곽 지형 + 둘러싸는 산 능선을 생성합니다.
        
        구조:
        ┌─────────────────────────────────────┐
        │         Mountain Rim (산 능선)        │
        │  ┌─────────────────────────────┐    │
        │  │    Low-Res Outer Ring       │    │
        │  │  ┌─────────────────────┐    │    │
        │  │  │   Main Terrain      │    │    │
        │  │  │   (High-Res 75x75m) │    │    │
        │  │  └─────────────────────┘    │    │
        │  └─────────────────────────────┘    │
        └─────────────────────────────────────┘
        
        - Low-Res Outer Ring: 메인 지형의 10배 낮은 해상도로 확장
        - Mountain Rim: 가장자리에 부드러운 산 능선으로 시야 차단
        """
        cfg = self.outer_terrain_cfg
        
        # 설정값 (YAML에서 가져오거나 기본값 사용)
        outer_size_multiplier = cfg.get("size_multiplier", 4.0)  # 전체 크기 배율 (메인의 4배)
        outer_resolution = cfg.get("resolution", 0.5)  # 저해상도 (메인의 10배 거친)
        rim_height = cfg.get("rim_height", 15.0)  # 산 능선 높이 (m)
        rim_width_ratio = cfg.get("rim_width_ratio", 0.15)  # 산 능선 폭 비율
        blend_width = cfg.get("blend_width", 5.0)  # 메인-외곽 경계 블렌딩 폭 (m)
        
        # 전체 외곽 지형 크기
        total_size_x = self.cfg.x_size * outer_size_multiplier
        total_size_y = self.cfg.y_size * outer_size_multiplier
        
        outer_grid_width = int(total_size_x / outer_resolution)
        outer_grid_height = int(total_size_y / outer_resolution)
        
        outer_x_offset = -total_size_x / 2.0
        outer_y_offset = -total_size_y / 2.0
        
        print(f"[TerrainManager] Creating outer terrain: {total_size_x:.0f}x{total_size_y:.0f}m "
              f"at resolution {outer_resolution}m ({outer_grid_width}x{outer_grid_height} vertices)")
        
        # 1. 기본 노이즈 지형 생성 (저해상도)
        outer_dem = self._generate_outer_noise(outer_grid_width, outer_grid_height, outer_resolution)
        
        # 2. 메인 지형 경계 높이와 블렌딩
        outer_dem = self._blend_with_main_terrain(outer_dem, outer_resolution, 
                                                   total_size_x, total_size_y, blend_width)
        
        # 3. 랜덤 언덕 추가
        outer_dem = self._add_mountain_rim(outer_dem, outer_resolution,
                                           total_size_x, total_size_y,
                                           rim_height, rim_width_ratio)
        
        # 4. 대형 크레이터 추가
        num_craters = cfg.get("num_craters", 5)
        outer_dem = self._add_outer_craters(outer_dem, outer_resolution,
                                            total_size_x, total_size_y,
                                            num_craters)
        
        # 5. 메시 생성 (Collision 포함 - 그림자 raycast용)
        self._create_outer_mesh(outer_dem, outer_resolution, 
                                outer_grid_width, outer_grid_height,
                                outer_x_offset, outer_y_offset)
        
        # 6. Horizon Plane 생성 (태양 차단용 - 매우 넓은 평면)
        horizon_cfg = cfg.get("horizon_plane", {})
        if horizon_cfg.get("enabled", True):
            self._create_horizon_plane(total_size_x, total_size_y, horizon_cfg)
        
        print(f"[TerrainManager] Outer terrain created with hills and {num_craters} large craters")

    def _generate_outer_noise(self, grid_width: int, grid_height: int, resolution: float) -> np.ndarray:
        """저해상도 외곽 지형용 기본 노이즈 생성"""
        import cv2
        
        dem = np.zeros((grid_height, grid_width), dtype=np.float32)
        
        # 부드러운 언덕 생성 (저주파 노이즈)
        rng = np.random.default_rng(self.cfg.seed -183)  # 다른 시드 사용
        
        octaves = 3
        persistence = 0.5
        scale = 8.0  # 큰 스케일 = 완만한 지형
        amplitude = self.cfg.z_scale * 0.3  # 메인 지형보다 낮은 진폭
        
        for i in range(octaves):
            sub_y = max(1, int(grid_height / scale))
            sub_x = max(1, int(grid_width / scale))
            
            noise_layer = rng.uniform(-1.0, 1.0, (sub_y, sub_x))
            if sub_x > 3 and sub_y > 3:
                noise_layer = cv2.GaussianBlur(noise_layer, (5, 5), 0)
            
            noise_layer = cv2.resize(noise_layer, (grid_width, grid_height), 
                                     interpolation=cv2.INTER_CUBIC)
            dem += noise_layer * amplitude
            
            amplitude *= persistence
            scale *= 2.0
        
        return dem

    def _blend_with_main_terrain(self, outer_dem: np.ndarray, outer_resolution: float,
                                  total_size_x: float, total_size_y: float,
                                  blend_width: float) -> np.ndarray:
        """메인 지형 경계와 외곽 지형을 자연스럽게 블렌딩 (정밀 버전)
        
        경계 근처에서는 메인 DEM의 실제 높이를 사용하여 완벽히 연결
        """
        import cv2
        
        grid_height, grid_width = outer_dem.shape
        
        # 메인 지형 경계 좌표 (외곽 그리드 기준) - 정확한 실수값
        main_x_start_f = (total_size_x / 2 - self.cfg.x_size / 2) / outer_resolution
        main_x_end_f = (total_size_x / 2 + self.cfg.x_size / 2) / outer_resolution
        main_y_start_f = (total_size_y / 2 - self.cfg.y_size / 2) / outer_resolution
        main_y_end_f = (total_size_y / 2 + self.cfg.y_size / 2) / outer_resolution
        
        main_x_start = int(main_x_start_f)
        main_x_end = int(main_x_end_f)
        main_y_start = int(main_y_start_f)
        main_y_end = int(main_y_end_f)
        
        # 블렌딩 폭
        blend_pixels = int(blend_width / outer_resolution) * 3
        
        # 메인 DEM (flip 적용)
        flipped_main_dem = np.flip(self.current_dem, axis=0)
        
        # 좌표 그리드 생성
        y_grid, x_grid = np.mgrid[:grid_height, :grid_width]
        
        # 유클리드 거리 맵 계산
        dx = np.maximum(np.maximum(main_x_start - x_grid, x_grid - main_x_end + 1), 0)
        dy = np.maximum(np.maximum(main_y_start - y_grid, y_grid - main_y_end + 1), 0)
        dist_to_main = np.sqrt(dx.astype(np.float32)**2 + dy.astype(np.float32)**2)
        
        # 블렌딩 마스크 (Smootherstep: 6t⁵ - 15t⁴ + 10t³)
        t = np.clip(dist_to_main / blend_pixels, 0, 1)
        blend_mask = t * t * t * (t * (t * 6 - 15) + 10)
        
        # 경계 근처(5픽셀 이내)는 blend_mask를 0으로 (메인 DEM 높이 100% 사용)
        near_boundary_mask = dist_to_main <= 3
        blend_mask[near_boundary_mask] = 0
        
        # 메인 DEM에서 높이 샘플링 (정밀 bilinear interpolation)
        # 각 외곽 그리드 셀의 월드 좌표 계산
        outer_x_offset = -total_size_x / 2.0
        outer_y_offset = -total_size_y / 2.0
        
        world_x = x_grid * outer_resolution + outer_x_offset
        world_y = y_grid * outer_resolution + outer_y_offset
        
        # 월드 좌표를 메인 DEM 그리드 인덱스로 변환
        main_fx = (world_x - self.x_offset) / self.cfg.resolution
        main_fy = (world_y - self.y_offset) / self.cfg.resolution
        main_fy = (self.grid_height - 1) - main_fy  # Y flip
        
        # 클램핑
        main_fx = np.clip(main_fx, 0, self.grid_width - 1.001)
        main_fy = np.clip(main_fy, 0, self.grid_height - 1.001)
        
        # Bilinear interpolation 준비
        x0 = np.floor(main_fx).astype(int)
        y0 = np.floor(main_fy).astype(int)
        x1 = np.minimum(x0 + 1, self.grid_width - 1)
        y1 = np.minimum(y0 + 1, self.grid_height - 1)
        
        sx = main_fx - x0
        sy = main_fy - y0
        
        # 4개 코너에서 샘플링
        v00 = flipped_main_dem[y0, x0]
        v10 = flipped_main_dem[y0, x1]
        v01 = flipped_main_dem[y1, x0]
        v11 = flipped_main_dem[y1, x1]
        
        # Bilinear interpolation
        v0 = v00 * (1 - sx) + v10 * sx
        v1 = v01 * (1 - sx) + v11 * sx
        edge_height_map = v0 * (1 - sy) + v1 * sy
        
        # 블렌딩 적용
        outer_dem = edge_height_map * (1 - blend_mask) + outer_dem * blend_mask
        
        return outer_dem

    def _add_mountain_rim(self, outer_dem: np.ndarray, outer_resolution: float,
                          total_size_x: float, total_size_y: float,
                          rim_height: float, rim_width_ratio: float) -> np.ndarray:
        """랜덤한 언덕/능선을 자연스럽게 배치 (개선된 버전)
        
        산맥 폭/경사 조절 파라미터:
        - hill_radius_min/max: 언덕 반경 범위 (m) - 클수록 넓은 산
        - hill_smoothness: 가우시안 계수 - 낮을수록 완만한 경사
        """
        grid_height, grid_width = outer_dem.shape
        
        # 설정에서 산맥 폭/경사 파라미터 가져오기
        cfg = self.outer_terrain_cfg
        hill_radius_min = cfg.get("hill_radius_min", 30.0)  # 기본 30m (기존 10m에서 증가)
        hill_radius_max = cfg.get("hill_radius_max", 80.0)  # 기본 80m (기존 40m에서 증가)
        hill_smoothness = cfg.get("hill_smoothness", 1.2)   # 기본 1.2 (기존 2.5에서 감소 → 더 완만)
        
        # 메인 지형 영역 (언덕이 침범하지 않도록)
        main_x_start = int((total_size_x / 2 - self.cfg.x_size / 2) / outer_resolution)
        main_x_end = int((total_size_x / 2 + self.cfg.x_size / 2) / outer_resolution)
        main_y_start = int((total_size_y / 2 - self.cfg.y_size / 2) / outer_resolution)
        main_y_end = int((total_size_y / 2 + self.cfg.y_size / 2) / outer_resolution)
        
        # 메인 지형으로부터의 최소 거리 (버퍼)
        min_dist_from_main = int(15.0 / outer_resolution)  # 15m 버퍼
        
        rng = np.random.default_rng(self.cfg.seed -83)
        
        # 랜덤 언덕 개수 (외곽 영역 크기에 비례)
        num_hills = int(30 * rim_width_ratio * 4)  # 대략 15~25개
        
        print(f"[TerrainManager] Adding {num_hills} random hills to outer terrain")
        
        # 좌표 그리드 (언덕 계산용)
        y_grid, x_grid = np.ogrid[:grid_height, :grid_width]
        
        for i in range(num_hills):
            # 랜덤 위치 선택 (메인 영역 밖, 가장자리 근처 선호)
            attempts = 0
            while attempts < 50:
                # 가장자리 근처에 더 많이 배치되도록 분포 조정
                edge_bias = rng.random()
                if edge_bias < 0.7:  # 70% 확률로 가장자리 근처
                    # 4개 변 중 하나 선택
                    side = rng.integers(0, 4)
                    margin = int(grid_width * 0.25)  # 가장자리 25% 영역
                    
                    if side == 0:  # 상단
                        hx = rng.integers(0, grid_width)
                        hy = rng.integers(0, margin)
                    elif side == 1:  # 하단
                        hx = rng.integers(0, grid_width)
                        hy = rng.integers(grid_height - margin, grid_height)
                    elif side == 2:  # 좌측
                        hx = rng.integers(0, margin)
                        hy = rng.integers(0, grid_height)
                    else:  # 우측
                        hx = rng.integers(grid_width - margin, grid_width)
                        hy = rng.integers(0, grid_height)
                else:  # 30% 확률로 중간 외곽 영역
                    hx = rng.integers(0, grid_width)
                    hy = rng.integers(0, grid_height)
                
                # 메인 영역과의 거리 확인
                dx = max(main_x_start - hx, 0, hx - main_x_end + 1)
                dy = max(main_y_start - hy, 0, hy - main_y_end + 1)
                dist_from_main = np.sqrt(dx**2 + dy**2)
                
                if dist_from_main >= min_dist_from_main:
                    break
                attempts += 1
            
            if attempts >= 50:
                continue
            
            # 랜덤 언덕 크기와 높이 (설정에서 가져온 범위 사용)
            hill_radius = rng.uniform(hill_radius_min, hill_radius_max) / outer_resolution
            hill_height = rng.uniform(0.3, 1.0) * rim_height  # 높이 변동
            
            # 언덕 모양 (타원형으로 변형) - 더 넓은 산맥을 위해 aspect_ratio 범위 조정
            aspect_ratio = rng.uniform(0.6, 1.8)  # 가로세로 비율 (덜 극단적)
            rotation = rng.uniform(0, 2 * np.pi)  # 회전
            
            # 각 픽셀에서 언덕 중심까지의 거리 계산 (회전된 타원)
            dx = x_grid - hx
            dy = y_grid - hy
            
            # 회전 적용
            dx_rot = dx * np.cos(rotation) + dy * np.sin(rotation)
            dy_rot = -dx * np.sin(rotation) + dy * np.cos(rotation)
            
            # 타원 거리
            dist = np.sqrt((dx_rot / aspect_ratio)**2 + dy_rot**2)
            
            # 부드러운 언덕 프로파일 (가우시안) - hill_smoothness가 낮을수록 더 완만
            hill_profile = np.exp(-hill_smoothness * (dist / hill_radius)**2)
            
            # 언덕이 메인 영역에 영향 주지 않도록 마스킹
            main_mask = (
                (x_grid >= main_x_start - min_dist_from_main//2) & 
                (x_grid < main_x_end + min_dist_from_main//2) &
                (y_grid >= main_y_start - min_dist_from_main//2) & 
                (y_grid < main_y_end + min_dist_from_main//2)
            )
            hill_profile[main_mask] = 0
            
            # 언덕 추가
            outer_dem += hill_height * hill_profile
        
        return outer_dem

    def _add_outer_craters(self, outer_dem: np.ndarray, outer_resolution: float,
                           total_size_x: float, total_size_y: float,
                           num_craters: int) -> np.ndarray:
        """외곽 지형에 대형 크레이터 추가"""
        grid_height, grid_width = outer_dem.shape
        
        # 메인 지형 영역 (크레이터가 침범하지 않도록)
        main_x_start = int((total_size_x / 2 - self.cfg.x_size / 2) / outer_resolution)
        main_x_end = int((total_size_x / 2 + self.cfg.x_size / 2) / outer_resolution)
        main_y_start = int((total_size_y / 2 - self.cfg.y_size / 2) / outer_resolution)
        main_y_end = int((total_size_y / 2 + self.cfg.y_size / 2) / outer_resolution)
        
        # 메인 지형으로부터의 최소 거리
        min_dist_from_main = int(20.0 / outer_resolution)  # 20m 버퍼
        
        rng = np.random.default_rng(self.cfg.seed + 17)
        
        # 좌표 그리드
        y_grid, x_grid = np.ogrid[:grid_height, :grid_width]
        
        print(f"[TerrainManager] Adding {num_craters} large craters to outer terrain")
        
        placed_craters = 0
        for i in range(num_craters):
            attempts = 0
            while attempts < 100:
                # 랜덤 위치 선택
                cx = rng.integers(int(grid_width * 0.1), int(grid_width * 0.9))
                cy = rng.integers(int(grid_height * 0.1), int(grid_height * 0.9))
                
                # 메인 영역과의 거리 확인
                dx = max(main_x_start - cx, 0, cx - main_x_end + 1)
                dy = max(main_y_start - cy, 0, cy - main_y_end + 1)
                dist_from_main = np.sqrt(dx**2 + dy**2)
                
                if dist_from_main >= min_dist_from_main:
                    break
                attempts += 1
            
            if attempts >= 100:
                continue
            
            # 크레이터 크기 (대형: 20~80m 반경)
            crater_radius_m = rng.uniform(20, 60)
            crater_radius = crater_radius_m / outer_resolution
            
            # 크레이터 깊이 (반경의 15~25%)
            crater_depth = crater_radius_m * rng.uniform(0.15, 0.25)
            
            # 림(테두리) 높이 (깊이의 10~15%)
            rim_height = crater_depth * rng.uniform(0.1, 0.15)
            
            # 타원형 변형
            aspect_ratio = rng.uniform(0.7, 1.0)
            rotation = rng.uniform(0, 2 * np.pi)
            
            # 각 픽셀에서 크레이터 중심까지의 거리
            dx = x_grid - cx
            dy = y_grid - cy
            
            # 회전 적용
            dx_rot = dx * np.cos(rotation) + dy * np.sin(rotation)
            dy_rot = -dx * np.sin(rotation) + dy * np.cos(rotation)
            
            # 타원 거리 (정규화: 0 = 중심, 1 = 림)
            dist_norm = np.sqrt((dx_rot / aspect_ratio)**2 + dy_rot**2) / crater_radius
            
            # 크레이터 프로파일 (실제 달 크레이터 형태 근사)
            # - 중심부: 평평한 바닥
            # - 벽: 급경사
            # - 림: 약간 솟아오름
            # - 외부: 점진적으로 원래 높이로
            
            crater_profile = np.zeros_like(dist_norm)
            
            # 중심부 (r < 0.3): 평평한 바닥
            floor_mask = dist_norm < 0.3
            crater_profile[floor_mask] = -crater_depth
            
            # 벽 (0.3 <= r < 0.85): 급경사
            wall_mask = (dist_norm >= 0.3) & (dist_norm < 0.85)
            t_wall = (dist_norm[wall_mask] - 0.3) / 0.55
            # 부드러운 S자 곡선
            crater_profile[wall_mask] = -crater_depth * (1 - t_wall**2 * (3 - 2*t_wall))
            
            # 림 (0.85 <= r < 1.2): 솟아오른 테두리
            rim_mask = (dist_norm >= 0.85) & (dist_norm < 1.2)
            t_rim = (dist_norm[rim_mask] - 0.85) / 0.35
            # 가우시안 형태의 림
            crater_profile[rim_mask] = rim_height * np.exp(-4 * (t_rim - 0.3)**2)
            
            # 외부 (r >= 1.2): 점진적으로 0으로
            outer_mask = (dist_norm >= 1.2) & (dist_norm < 1.8)
            t_outer = (dist_norm[outer_mask] - 1.2) / 0.6
            crater_profile[outer_mask] = rim_height * np.exp(-4 * 0.49) * (1 - t_outer**2)
            
            # 메인 영역에 영향 주지 않도록 마스킹 (부드러운 페이드아웃)
            main_center_x = (main_x_start + main_x_end) / 2
            main_center_y = (main_y_start + main_y_end) / 2
            dist_to_main_center = np.sqrt((x_grid - main_center_x)**2 + (y_grid - main_center_y)**2)
            
            # 메인 영역 경계에서 부드럽게 페이드아웃
            fade_start = min_dist_from_main
            fade_end = min_dist_from_main * 2
            
            main_dist_x = np.maximum(np.maximum(main_x_start - x_grid, x_grid - main_x_end + 1), 0)
            main_dist_y = np.maximum(np.maximum(main_y_start - y_grid, y_grid - main_y_end + 1), 0)
            main_dist = np.sqrt(main_dist_x.astype(np.float32)**2 + main_dist_y.astype(np.float32)**2)
            
            fade_mask = np.clip((main_dist - fade_start) / (fade_end - fade_start), 0, 1)
            crater_profile *= fade_mask
            
            # 크레이터 적용
            outer_dem += crater_profile
            placed_craters += 1
        
        print(f"[TerrainManager] Placed {placed_craters} large craters")
        return outer_dem

    def _sample_main_dem_at_world(self, world_x: float, world_y: float, 
                                   flipped_main_dem: np.ndarray) -> float:
        """메인 DEM에서 월드 좌표의 높이를 bilinear interpolation으로 샘플링
        
        Args:
            world_x, world_y: 월드 좌표
            flipped_main_dem: Y축 flip된 메인 DEM (메시 좌표계와 일치)
        
        Returns:
            해당 위치의 높이값
        """
        # 월드 좌표를 메인 DEM 그리드 인덱스로 변환
        fx = (world_x - self.x_offset) / self.cfg.resolution
        fy = (world_y - self.y_offset) / self.cfg.resolution
        
        # Y축 flip (메시 좌표계)
        fy = (self.grid_height - 1) - fy
        
        # 경계 체크
        fx = np.clip(fx, 0, self.grid_width - 1)
        fy = np.clip(fy, 0, self.grid_height - 1)
        
        # Bilinear interpolation
        x0 = int(np.floor(fx))
        x1 = min(x0 + 1, self.grid_width - 1)
        y0 = int(np.floor(fy))
        y1 = min(y0 + 1, self.grid_height - 1)
        
        sx = fx - x0
        sy = fy - y0
        
        v00 = float(flipped_main_dem[y0, x0])
        v10 = float(flipped_main_dem[y0, x1])
        v01 = float(flipped_main_dem[y1, x0])
        v11 = float(flipped_main_dem[y1, x1])
        
        v0 = v00 * (1 - sx) + v10 * sx
        v1 = v01 * (1 - sx) + v11 * sx
        return v0 * (1 - sy) + v1 * sy

    def _create_outer_mesh(self, dem: np.ndarray, resolution: float,
                           grid_width: int, grid_height: int,
                           x_offset: float, y_offset: float):
        """외곽 지형 메시 생성 (시각적 용도만, 충돌 없음)
        
        경계 gap 해결: 외곽 지형을 메인 지형 안쪽으로 오버랩시키고,
        오버랩 영역의 높이를 메인 DEM에서 직접 샘플링하여 완벽히 연결
        """
        
        # 메인 지형 영역 계산 (구멍을 뚫을 부분)
        total_size_x = grid_width * resolution
        total_size_y = grid_height * resolution
        
        # 원래 메인 영역 경계 (외곽 그리드 기준)
        main_x_start_exact = (total_size_x / 2 - self.cfg.x_size / 2) / resolution
        main_x_end_exact = (total_size_x / 2 + self.cfg.x_size / 2) / resolution
        main_y_start_exact = (total_size_y / 2 - self.cfg.y_size / 2) / resolution
        main_y_end_exact = (total_size_y / 2 + self.cfg.y_size / 2) / resolution
        
        # 오버랩을 위해 구멍을 더 작게 (2픽셀 안쪽으로) - gap 제거
        overlap_pixels = 2
        main_x_start = int(main_x_start_exact) + overlap_pixels
        main_x_end = int(main_x_end_exact) - overlap_pixels
        main_y_start = int(main_y_start_exact) + overlap_pixels
        main_y_end = int(main_y_end_exact) - overlap_pixels
        
        # 메인 DEM을 flip (메시 좌표계와 일치시키기 위해)
        flipped_main_dem = np.flip(self.current_dem, axis=0)
        
        vertices = []
        tex_coords = []
        vertex_map = {}  # (x, y) -> vertex index
        
        tiling_factor = total_size_x / 20.0  # 텍스처 타일링
        
        # 버텍스 생성 (메인 지형 영역 제외, 경계는 메인 DEM에서 샘플링)
        for y in range(grid_height):
            for x in range(grid_width):
                # 메인 지형 영역 내부(오버랩 제외)는 건너뜀
                if main_x_start <= x < main_x_end and main_y_start <= y < main_y_end:
                    continue
                
                vertex_idx = len(vertices)
                vertex_map[(x, y)] = vertex_idx
                
                world_x = float(x * resolution + x_offset)
                world_y = float(y * resolution + y_offset)
                
                # 경계 오버랩 영역인지 확인 (메인 지형과 겹치는 부분)
                is_overlap = (
                    (int(main_x_start_exact) <= x < int(main_x_end_exact)) and
                    (int(main_y_start_exact) <= y < int(main_y_end_exact))
                )
                
                if is_overlap:
                    # 오버랩 영역: 메인 DEM에서 bilinear interpolation으로 높이 샘플링
                    world_z = self._sample_main_dem_at_world(world_x, world_y, flipped_main_dem)
                else:
                    world_z = float(dem[y, x])
                
                vertices.append(Gf.Vec3f(world_x, world_y, world_z))
                
                u = float(x / (grid_width - 1)) * tiling_factor
                v = float(y / (grid_height - 1)) * tiling_factor
                tex_coords.append(Gf.Vec2f(u, v))
        
        # 인덱스 생성 (삼각형)
        indices = []
        for y in range(1, grid_height):
            for x in range(1, grid_width):
                # 4개의 버텍스가 모두 존재하는 경우에만 삼각형 생성
                p1 = (x, y)
                p2 = (x, y - 1)
                p3 = (x - 1, y - 1)
                p4 = (x - 1, y)
                
                if all(p in vertex_map for p in [p1, p2, p3, p4]):
                    # 두 개의 삼각형으로 쿼드 구성
                    i1 = vertex_map[p1]
                    i2 = vertex_map[p2]
                    i3 = vertex_map[p3]
                    i4 = vertex_map[p4]
                    indices.extend([i1, i3, i2, i1, i4, i3])
                    
                # 메인 지형 경계에서 삼각형 연결 (경계 보간)
                # 일부 버텍스만 존재하는 경우 처리
                elif sum(p in vertex_map for p in [p1, p2, p3, p4]) == 3:
                    valid_points = [p for p in [p1, p2, p3, p4] if p in vertex_map]
                    if len(valid_points) == 3:
                        indices.extend([vertex_map[valid_points[0]], 
                                       vertex_map[valid_points[1]], 
                                       vertex_map[valid_points[2]]])
        
        if not vertices or not indices:
            print("[TerrainManager] Warning: No outer terrain vertices generated")
            return
        
        vertices_array = np.array(vertices, dtype=np.float32)
        tex_coords_array = np.array(tex_coords, dtype=np.float32)
        indices_array = np.array(indices, dtype=np.int32)
        
        # USD 메시 생성
        outer_mesh_path = "/World/OuterTerrain"
        mesh_prim = UsdGeom.Mesh.Define(self.stage, outer_mesh_path)
        mesh_prim.GetPointsAttr().Set(vertices_array)
        mesh_prim.GetFaceVertexIndicesAttr().Set(indices_array)
        mesh_prim.GetFaceVertexCountsAttr().Set(np.full(len(indices) // 3, 3, dtype=np.int32))
        
        # 텍스처 좌표
        primvar_api = UsdGeom.PrimvarsAPI(mesh_prim)
        pv = primvar_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(tex_coords_array)
        
        # [Shadow Detection] 외부 지형에도 Collision 추가 (raycast 그림자 감지용)
        outer_collision_prim = mesh_prim.GetPrim()
        UsdPhysics.CollisionAPI.Apply(outer_collision_prim)
        outer_mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(outer_collision_prim)
        outer_mesh_collision.CreateApproximationAttr().Set("none")  # 정확한 메시 충돌
        print("[TerrainManager] Outer terrain collision enabled for shadow raycast")
        
        # 메인 지형과 동일한 재질 적용
        terrain_mat_prim = self.stage.GetPrimAtPath("/World/Looks/TerrainMaterial")
        if not terrain_mat_prim or not terrain_mat_prim.IsValid():
            terrain_mat_prim = self.stage.GetPrimAtPath("/World/Looks/DefaultTerrainMaterial")
        
        if terrain_mat_prim and terrain_mat_prim.IsValid():
            mat = UsdShade.Material(terrain_mat_prim)
            UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(mat)
        
        self.outer_terrain_mesh = mesh_prim
        
        # [중요] Outer terrain의 실제 월드 좌표 경계 저장 (Horizon Plane용)
        # vertices_array[:, 0] = x좌표, [:, 1] = y좌표
        self.outer_terrain_bounds = {
            'x_min': float(vertices_array[:, 0].min()),
            'x_max': float(vertices_array[:, 0].max()),
            'y_min': float(vertices_array[:, 1].min()),
            'y_max': float(vertices_array[:, 1].max()),
        }
        print(f"[TerrainManager] Outer terrain bounds: x=[{self.outer_terrain_bounds['x_min']:.1f}, {self.outer_terrain_bounds['x_max']:.1f}], "
              f"y=[{self.outer_terrain_bounds['y_min']:.1f}, {self.outer_terrain_bounds['y_max']:.1f}]")
        print(f"[TerrainManager] Outer mesh created: {len(vertices)} vertices, {len(indices)//3} triangles")

    def scatter_rocks(self, rock_assets_dir: str, num_rocks: int = 50, excluded_positions: list = None, exclusion_radius: float = 5.0):
        """
        바위를 지형 위에 랜덤하게 배치합니다.
        
        Args:
            rock_assets_dir: 바위 USD 에셋 디렉토리
            num_rocks: 배치할 바위 개수
            excluded_positions: 바위 배치를 제외할 위치 목록 [(x, y), ...]
            exclusion_radius: 제외 위치로부터의 최소 거리 (미터)
        """
        excluded_positions = excluded_positions or []
        
        if not os.path.isdir(rock_assets_dir):
            print(f"Warning: Rock assets directory not found: {rock_assets_dir}")
            return
        rock_files = []
        # Collect only .usdz packages from all subdirectories
        for root, dirs, files in os.walk(rock_assets_dir):
            for file in files:
                if file.lower().endswith((".usd", ".usda", ".usdz")):
                    rock_files.append(os.path.join(root, file))
        if not rock_files:
            print(f"Warning: No rock USDs found in {rock_assets_dir} or its subdirectories.")
            return
        
        print(f"Found {len(rock_files)} rock assets. Scattering {num_rocks} rocks...")
        if excluded_positions:
            print(f"  Excluding {len(excluded_positions)} position(s) with radius {exclusion_radius}m")
        
        placed_rocks = 0
        max_attempts = num_rocks * 5  # 최대 시도 횟수 (제외 영역 때문에 실패할 수 있음)
        attempt = 0
        
        while placed_rocks < num_rocks and attempt < max_attempts:
            attempt += 1
            
            rock_usd_path = random.choice(rock_files)
            x_idx = random.randint(0, self.grid_width - 1)
            y_idx = random.randint(0, self.grid_height - 1)
            x = x_idx * self.cfg.resolution + self.x_offset
            y = y_idx * self.cfg.resolution + self.y_offset
            
            # 제외 영역 확인 (로버 시작 위치 등)
            too_close = False
            for pos in excluded_positions:
                dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
                if dist < exclusion_radius:
                    too_close = True
                    break
            if too_close:
                continue
            
            # Sample height from the final DEM (with craters/noise) using
            # bilinear interpolation so rocks don't float above craters.
            z = self.sample_height_at_xy(x, y)
            rock_prim_path = f"/World/Rocks/Rock_{placed_rocks}"
            add_reference_to_stage(usd_path=rock_usd_path, prim_path=rock_prim_path)
            prim = self.stage.GetPrimAtPath(rock_prim_path)
            if not prim.IsValid(): continue
            xform_api = UsdGeom.Xformable(prim)
            xform_api.ClearXformOpOrder()
            xform_api.AddTranslateOp().Set(Gf.Vec3f(float(x), float(y), float(z)))
            rot_z_rad = random.uniform(0, 2 * np.pi)
            rot_z_deg = np.degrees(rot_z_rad)
            xform_api.AddRotateZOp().Set(rot_z_deg)
            # --- [수정] 돌의 크기를 다양한 폭으로 랜덤하게 배치 ---
            rock_scale = random.uniform(0.5, 3.0) # 0.3~2.0배 사이 랜덤 스케일
            xform_api.AddScaleOp().Set(Gf.Vec3f(rock_scale, rock_scale, rock_scale))
            UsdPhysics.CollisionAPI.Apply(prim)
            physx_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            physx_collision_api.CreateApproximationAttr().Set("none")
            # Bind the terrain material (or default) to every mesh under the
            # referenced prim so rock instances use the terrain material.
            terrain_mat_prim = self.stage.GetPrimAtPath("/World/Looks/TerrainMaterial")
            if not terrain_mat_prim or not terrain_mat_prim.IsValid():
                terrain_mat_prim = self.stage.GetPrimAtPath("/World/Looks/DefaultTerrainMaterial")
            if terrain_mat_prim and terrain_mat_prim.IsValid():
                try:
                    mat = UsdShade.Material(terrain_mat_prim)
                    from pxr import Usd
                    for p in Usd.PrimRange(prim):
                        try:
                            if p.IsA(UsdGeom.Mesh):
                                try:
                                    UsdShade.MaterialBindingAPI(p).Unbind()
                                except Exception:
                                    pass
                                try:
                                    UsdShade.MaterialBindingAPI(p).Bind(mat)
                                    print(f"Bound terrain material to {p.GetPath()}")
                                except Exception:
                                    print(f"Warning: Failed to bind material to {p.GetPath()}")
                        except Exception:
                            continue
                except Exception:
                    print(f"Warning: Could not create UsdShade.Material for terrain material; leaving originals for {rock_prim_path}")
            else:
                print(f"Warning: Terrain material not found; leaving original materials for {rock_prim_path}")
            
            # [추가] Rock에 Semantic Label 적용 (SDG용)
            self._apply_semantic_label(prim, "rock")
            
            self.rock_paths.append(rock_prim_path)
            rock_radius_idx = int(0.5 * rock_scale / self.cfg.resolution)
            for dx in range(-rock_radius_idx, rock_radius_idx + 1):
                for dy in range(-rock_radius_idx, rock_radius_idx + 1):
                    nx, ny = x_idx + dx, y_idx + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        self.rock_dem[ny, nx] = 2.0
            
            placed_rocks += 1
        
        print(f"  Placed {placed_rocks} rocks (attempted {attempt} times)")
    
    def get_heights(self, world_positions_xy):
        # ... (이전 코드와 동일, 생략)
        if not isinstance(world_positions_xy, np.ndarray):
            world_positions_xy = np.array(world_positions_xy)
        grid_indices = ((world_positions_xy - np.array([self.x_offset, self.y_offset])) / self.cfg.resolution).astype(int)
        gx = np.clip(grid_indices[:, 0], 0, self.grid_width - 1)
        gy = np.clip(grid_indices[:, 1], 0, self.grid_height - 1)
        z_values = self.current_dem[gy, gx]
        return z_values

    def sample_height_at_xy(self, x: float, y: float) -> float:
        """Bilinearly sample terrain height at world coordinates (x,y).

        This uses the flipped DEM array (matching mesh coordinates) and performs
        bilinear interpolation so rocks are placed accurately on sloped/cratered terrain.
        """
        # convert world xy to fractional grid indices
        fx = (x - self.x_offset) / self.cfg.resolution
        fy = (y - self.y_offset) / self.cfg.resolution
        
        # [핵심 수정] Y축 flip - 메시는 Y축이 flip된 DEM을 사용하므로 일치시킴
        fy = (self.grid_height - 1) - fy

        if fx < 0 or fy < 0 or fx > (self.grid_width - 1) or fy > (self.grid_height - 1):
            # out of bounds: clamp to nearest
            ix = int(np.clip(round(fx), 0, self.grid_width - 1))
            iy = int(np.clip(round(fy), 0, self.grid_height - 1))
            return float(self.current_dem[iy, ix])

        x0 = int(np.floor(fx))
        x1 = min(x0 + 1, self.grid_width - 1)
        y0 = int(np.floor(fy))
        y1 = min(y0 + 1, self.grid_height - 1)

        sx = fx - x0
        sy = fy - y0

        v00 = float(self.current_dem[y0, x0])
        v10 = float(self.current_dem[y0, x1])
        v01 = float(self.current_dem[y1, x0])
        v11 = float(self.current_dem[y1, x1])

        # bilinear interpolation
        v0 = v00 * (1 - sx) + v10 * sx
        v1 = v01 * (1 - sx) + v11 * sx
        v = v0 * (1 - sy) + v1 * sy
        return v

    def update_mesh_patch(self, dem_patch: np.ndarray, start_x: int, start_y: int):
        """
        [벡터화 최적화] NumPy로 패치 영역의 버텍스 높이값을 한 번에 업데이트
        Python 루프 대신 NumPy 인덱싱 사용으로 ~100배 속도 향상
        """
        if dem_patch.size == 0 or self.terrain_mesh is None:
            return

        patch_height, patch_width = dem_patch.shape
        
        # Y축 flip된 DEM 패치
        flipped_patch = np.flip(dem_patch, axis=0)
        
        # 패치 영역에 해당하는 버텍스 인덱스 계산 (벡터화)
        y_indices = np.arange(start_y, start_y + patch_height)
        x_indices = np.arange(start_x, start_x + patch_width)
        
        # 2D 그리드 인덱스 생성
        yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')
        vertex_indices = yy * self.grid_width + xx
        
        # 경계 체크
        valid_mask = vertex_indices < len(self.current_vertices_np)
        
        # [벡터화] 모든 높이값을 한 번에 업데이트
        self.current_vertices_np[vertex_indices[valid_mask].flatten(), 2] = flipped_patch[valid_mask].flatten()
        
        # 메시에 적용 (한 번만 호출)
        self.terrain_mesh.GetPointsAttr().Set(self.current_vertices_np)

    def update_mesh_from_dem(self, dem, flip_input=True):
        # ... (이전 코드와 동일, 생략)
        if dem.shape != (self.grid_height, self.grid_width):
             import cv2
             dem = cv2.resize(dem, (self.grid_width, self.grid_height), interpolation=cv2.INTER_LINEAR)
        if flip_input:
            flipped_dem = np.flip(dem, 0)
        else:
            flipped_dem = dem
        new_vertices = self.base_vertices.copy() # Use copy to be safe
        new_vertices[:, 2] = flipped_dem.flatten()
        self.terrain_mesh.GetPointsAttr().Set(new_vertices)

    def _create_horizon_plane(self, outer_size_x: float, outer_size_y: float, 
                               horizon_cfg: dict):
        """
        태양 차단용 매우 넓은 Horizon Plane을 생성합니다.
        
        목적:
        - 태양 고도가 0° 이하일 때 빛을 완전히 차단
        - 외곽 지형 바깥에 거대한 평면을 배치하여 그림자 raycast가 hit 하도록 함
        
        구조:
        ┌─────────────────────────────────────────────────────┐
        │                 Horizon Plane (수 km)               │
        │     ┌─────────────────────────────────────┐         │
        │     │         Outer Terrain               │         │
        │     │     ┌─────────────────────┐         │         │
        │     │     │   Main Terrain      │         │         │
        │     │     └─────────────────────┘         │         │
        │     └─────────────────────────────────────┘         │
        └─────────────────────────────────────────────────────┘
        
        Args:
            outer_size_x, outer_size_y: 외곽 지형 크기 (m)
            horizon_cfg: 설정 딕셔너리
        """
        # 설정값
        size_multiplier = horizon_cfg.get("size_multiplier", 20.0)  # 외곽 지형의 20배
        height_offset = horizon_cfg.get("height_offset", -5.0)  # 지형보다 약간 아래
        resolution = horizon_cfg.get("resolution", 50.0)  # 매우 저해상도 (50m/cell)
        
        # Horizon Plane 크기 (예: 외곽 500m × 20 = 10km)
        horizon_size_x = outer_size_x * size_multiplier
        horizon_size_y = outer_size_y * size_multiplier
        
        # 그리드 크기
        grid_width = max(4, int(horizon_size_x / resolution))
        grid_height = max(4, int(horizon_size_y / resolution))
        
        # 실제 해상도 재계산
        actual_res_x = horizon_size_x / grid_width
        actual_res_y = horizon_size_y / grid_height
        
        x_offset = -horizon_size_x / 2.0
        y_offset = -horizon_size_y / 2.0
        
        print(f"[TerrainManager] Creating Horizon Plane: {horizon_size_x/1000:.1f}km x {horizon_size_y/1000:.1f}km")
        print(f"  Grid: {grid_width}x{grid_height}, Resolution: {actual_res_x:.1f}m")
        
        # 외곽 지형 영역 (구멍을 뚫을 부분) - Outer terrain의 실제 버텍스 경계 사용
        if not hasattr(self, 'outer_terrain_bounds'):
            print("  [Warning] outer_terrain_bounds not found, using default calculation")
            outer_world_x_min = -outer_size_x / 2.0
            outer_world_x_max = outer_size_x / 2.0
            outer_world_y_min = -outer_size_y / 2.0
            outer_world_y_max = outer_size_y / 2.0
        else:
            # Outer terrain 메시의 실제 버텍스 경계 사용
            outer_world_x_min = self.outer_terrain_bounds['x_min']
            outer_world_x_max = self.outer_terrain_bounds['x_max']
            outer_world_y_min = self.outer_terrain_bounds['y_min']
            outer_world_y_max = self.outer_terrain_bounds['y_max']
        
        print(f"  Outer terrain actual bounds: x=[{outer_world_x_min:.1f}, {outer_world_x_max:.1f}], "
              f"y=[{outer_world_y_min:.1f}, {outer_world_y_max:.1f}]")
        
        # 월드 좌표를 grid 인덱스로 변환: grid_x = (world_x - x_offset) / actual_res_x
        inner_x_start_exact = (outer_world_x_min - x_offset) / actual_res_x
        inner_x_end_exact = (outer_world_x_max - x_offset) / actual_res_x
        inner_y_start_exact = (outer_world_y_min - y_offset) / actual_res_y
        inner_y_end_exact = (outer_world_y_max - y_offset) / actual_res_y
        
        # 오버랩을 위해 구멍을 안쪽으로 축소 (gap 방지)
        overlap_cells = 1
        inner_x_start = int(math.ceil(inner_x_start_exact)) + overlap_cells
        inner_x_end = int(math.floor(inner_x_end_exact)) - overlap_cells + 1  # +1 for inclusive end
        inner_y_start = int(math.ceil(inner_y_start_exact)) + overlap_cells
        inner_y_end = int(math.floor(inner_y_end_exact)) - overlap_cells + 1
        
        print(f"  Inner hole grid: x=[{inner_x_start}, {inner_x_end}), y=[{inner_y_start}, {inner_y_end})")
        
        # 버텍스 생성 (외곽 지형 영역 제외 - 도넛 모양)
        vertices = []
        vertex_map = {}
        
        for y in range(grid_height):
            for x in range(grid_width):
                # 외곽 지형 영역 내부는 건너뜀
                if inner_x_start <= x < inner_x_end and inner_y_start <= y < inner_y_end:
                    continue
                
                vertex_idx = len(vertices)
                vertex_map[(x, y)] = vertex_idx
                
                world_x = float(x * actual_res_x + x_offset)
                world_y = float(y * actual_res_y + y_offset)
                world_z = float(height_offset)  # 평평한 높이
                
                vertices.append(Gf.Vec3f(world_x, world_y, world_z))
        
        # 인덱스 생성
        indices = []
        for y in range(1, grid_height):
            for x in range(1, grid_width):
                p1 = (x, y)
                p2 = (x, y - 1)
                p3 = (x - 1, y - 1)
                p4 = (x - 1, y)
                
                if all(p in vertex_map for p in [p1, p2, p3, p4]):
                    i1 = vertex_map[p1]
                    i2 = vertex_map[p2]
                    i3 = vertex_map[p3]
                    i4 = vertex_map[p4]
                    indices.extend([i1, i3, i2, i1, i4, i3])
        
        if not vertices or not indices:
            print("[TerrainManager] Warning: No horizon plane vertices generated")
            return
        
        vertices_array = np.array(vertices, dtype=np.float32)
        indices_array = np.array(indices, dtype=np.int32)
        
        # USD 메시 생성
        horizon_path = "/World/HorizonPlane"
        mesh_prim = UsdGeom.Mesh.Define(self.stage, horizon_path)
        mesh_prim.GetPointsAttr().Set(vertices_array)
        mesh_prim.GetFaceVertexIndicesAttr().Set(indices_array)
        mesh_prim.GetFaceVertexCountsAttr().Set(np.full(len(indices) // 3, 3, dtype=np.int32))
        
        # Collision 추가 (raycast 그림자 감지용 - 핵심!)
        horizon_prim = mesh_prim.GetPrim()
        UsdPhysics.CollisionAPI.Apply(horizon_prim)
        horizon_mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(horizon_prim)
        horizon_mesh_collision.CreateApproximationAttr().Set("none")
        
        # 시각적으로 보이지 않도록 설정 (선택적)
        # 옵션 1: 완전히 투명하게 (렌더링 안 함)
        # 옵션 2: 어두운 재질 적용
        hide_visual = horizon_cfg.get("hide_visual", False)
        
        if hide_visual:
            # 렌더링에서 제외 (Collision은 유지)
            mesh_prim.GetPrim().GetAttribute("visibility").Set("invisible")
            # 또는: mesh_prim.MakeInvisible()
            print(f"  Horizon plane invisible (collision only)")
        else:
            # 지형과 동일한 재질 적용 (시각적 연속성)
            terrain_mat_prim = self.stage.GetPrimAtPath("/World/Looks/TerrainMaterial")
            if not terrain_mat_prim or not terrain_mat_prim.IsValid():
                terrain_mat_prim = self.stage.GetPrimAtPath("/World/Looks/DefaultTerrainMaterial")
            
            if terrain_mat_prim and terrain_mat_prim.IsValid():
                mat = UsdShade.Material(terrain_mat_prim)
                UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(mat)
        
        self.horizon_plane = mesh_prim
        print(f"  Created: {len(vertices)} vertices, {len(indices)//3} triangles")
        print(f"  Height: {height_offset}m (below terrain)")
        print(f"  Purpose: Block sun rays at low elevation angles")
