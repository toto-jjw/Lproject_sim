# Lproject_sim/src/physics/deformation.py (최종 수정본)

import numpy as np
import warp as wp
import omni.warp
from src.config.physics_config import DeformationEngineConf

# Initialize Warp
wp.init()

@wp.kernel
def deform_terrain_kernel_sinkage(
    dem: wp.array(dtype=float, ndim=2),
    deformed_mask: wp.array(dtype=float, ndim=2),
    wheel_pos: wp.array(dtype=wp.vec3),      # [x, y, wheel_bottom_z]
    terrain_z: wp.array(dtype=float),         # 각 바퀴 위치의 원래 지형 높이
    sinkages: wp.array(dtype=float),          # 침하량
    resolution: float,
    total_x_offset: float,
    total_y_offset: float,
    start_grid_x: int,
    start_grid_y: int,
    footprint_radius: float,
    wheel_radius: float,
    contact_threshold: float                  # 접촉 판정 임계값
):
    """
    Wheel Track 생성을 위한 지형 변형 커널
    
    핵심 로직:
    1. 바퀴 바닥(wheel_bottom)이 지형(terrain_z)과 접촉할 때만 변형
    2. 지형을 아래로 파내는 방향으로만 변형 (높이 증가 불가)
    3. 접촉 깊이(penetration)에 비례하여 변형량 결정
    """
    x_local, y_local = wp.tid()
    
    x_global = start_grid_x + x_local
    y_global = start_grid_y + y_local
        
    # 월드 좌표 계산
    wx = float(x_global) * resolution + total_x_offset
    wy = float(y_global) * resolution + total_y_offset
    
    # 현재 DEM 높이
    current_dem_h = dem[y_global, x_global]
    
    num_wheels = wheel_pos.shape[0]
    max_deform_depth = float(0.0)
    
    for i in range(num_wheels):
        w_pos = wheel_pos[i]
        wheel_bottom_z = w_pos[2]  # 바퀴 바닥 높이
        original_terrain_z = terrain_z[i]  # 해당 바퀴 위치의 원래 지형 높이
        sink = sinkages[i]
        
        # [핵심] 접촉 판정: 바퀴 바닥이 지형 근처에 있는지 확인
        # penetration > 0: 바퀴가 지형 안으로 들어감 (접촉)
        # penetration < 0: 바퀴가 지형 위에 떠 있음 (비접촉)
        penetration = original_terrain_z - wheel_bottom_z
        
        # 접촉 임계값 이내일 때만 변형 적용
        if penetration > -contact_threshold:
            dx = wx - w_pos[0]
            dy = wy - w_pos[1]
            dist_sq = dx*dx + dy*dy
            
            # 바퀴 풋프린트 내부인지 확인
            if dist_sq < footprint_radius * footprint_radius:
                dist = wp.sqrt(dist_sq)
                normalized_dist = dist / footprint_radius
                
                # 가우시안 형태의 침하 프로파일 (중심이 가장 깊음)
                falloff = wp.exp(-2.0 * normalized_dist * normalized_dist)
                
                # 접촉 강도에 따른 변형량 조절
                # penetration이 클수록 (깊이 파고들수록) 더 많이 변형
                contact_factor = wp.min(1.0, wp.max(0.0, (penetration + contact_threshold) / contact_threshold))
                
                depth = sink * falloff * contact_factor
                
                if depth > max_deform_depth:
                    max_deform_depth = depth
    
    # 변형 적용 (1mm 이상만)
    if max_deform_depth > 0.001:
        already_deformed = deformed_mask[y_global, x_global]
        
        # 누적 변형: 이미 변형된 깊이보다 더 깊은 경우만 추가 변형
        if max_deform_depth > already_deformed:
            additional_deform = max_deform_depth - already_deformed
            
            # [핵심] 지형을 낮추는 방향으로만 변형 (절대 높이지 않음)
            new_height = current_dem_h - additional_deform
            dem[y_global, x_global] = new_height
            deformed_mask[y_global, x_global] = max_deform_depth


@wp.kernel
def deform_terrain_kernel(
    dem: wp.array(dtype=float, ndim=2),
    deformed_mask: wp.array(dtype=float, ndim=2),  # 이미 변형된 위치 추적
    wheel_pos: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=float),
    resolution: float,
    total_x_offset: float,
    total_y_offset: float,
    start_grid_x: int,
    start_grid_y: int,
    amplitude_slope: float,
    max_deformation: float,
    footprint_radius: float
):
    x_local, y_local = wp.tid()
    
    x_global = start_grid_x + x_local
    y_global = start_grid_y + y_local
        
    wx = float(x_global) * resolution + total_x_offset
    wy = float(y_global) * resolution + total_y_offset
    
    num_wheels = wheel_pos.shape[0]
    target_depth = float(0.0)
    
    for i in range(num_wheels):
        w_pos = wheel_pos[i]
        force = forces[i]
        
        dx = wx - w_pos[0]
        dy = wy - w_pos[1]
        dist_sq = dx*dx + dy*dy
        
        if dist_sq < footprint_radius * footprint_radius:
            # 거리에 따른 감쇠 (중심에서 멀수록 얕게)
            dist = wp.sqrt(dist_sq)
            falloff = 1.0 - (dist / footprint_radius)
            falloff = falloff * falloff  # 부드러운 가장자리
            
            depth = amplitude_slope * force * falloff
            if depth > target_depth:
                target_depth = depth
    
    # 최대 변형 깊이 제한 (예: 2cm)
    target_depth = wp.min(target_depth, max_deformation)
    
    if target_depth > 0.0:
        current_h = dem[y_global, x_global]
        already_deformed = deformed_mask[y_global, x_global]
        
        # 이미 변형된 깊이보다 더 깊게만 변형 (누적 방지)
        if target_depth > already_deformed:
            additional_deform = target_depth - already_deformed
            dem[y_global, x_global] = current_h - additional_deform
            deformed_mask[y_global, x_global] = target_depth

class DeformationEngine:
    """
    Terrain deformation engine using NVIDIA Warp.
    Optimized for real-time performance with async GPU operations.
    """
    def __init__(self, config: DeformationEngineConf):
        self.cfg = config
        self.resolution = config.terrain_resolution
        self.width = config.terrain_width
        self.height = config.terrain_height
        
        self.x_offset_total = -self.width / 2.0
        self.y_offset_total = -self.height / 2.0
        
        self.grid_w = int(self.width / self.resolution)
        self.grid_h = int(self.height / self.resolution)

        # 변형 파라미터 조정
        self.amplitude_slope = 0.0002
        self.max_deformation = 0.02  # 최대 2cm 변형
        
        # GPU 배열 (재사용을 위해 미리 할당)
        self.dem_d = None
        self.deformed_mask_d = None  # 이미 변형된 깊이 추적
        self.pos_d = None
        self.force_d = None
        self.sinkage_d = None
        self.terrain_z_d = None
        
        self.footprint_radius = (self.cfg.footprint.width + self.cfg.footprint.height) / 4.0
        
        # 마지막 변형 영역 추적
        self.last_start_x = 0
        self.last_start_y = 0
        self.last_end_x = 0
        self.last_end_y = 0
        
        # 누적 변형 dirty 플래그 (sync 필요 여부)
        self._has_pending_deform = False
        
        self._initialized = False
        print(f"[DeformationEngine] Grid: {self.grid_w}x{self.grid_h}, "
              f"Terrain: {self.width}x{self.height}m @ {self.resolution}m/cell")

    def _init_gpu_arrays(self, dem_np: np.ndarray):
        """GPU 배열 초기화 (한 번만 실행)"""
        if not self._initialized:
            # Validate DEM shape matches expected grid
            expected_shape = (self.grid_h, self.grid_w)
            if dem_np.shape != expected_shape:
                print(f"[DeformationEngine] WARNING: DEM shape {dem_np.shape} != expected {expected_shape}")
                print(f"[DeformationEngine] Adjusting grid to match DEM: {dem_np.shape[1]}x{dem_np.shape[0]}")
                self.grid_h, self.grid_w = dem_np.shape
            
            # DEM on GPU (float32 for Warp kernel)
            dem_f32 = dem_np.astype(np.float32)
            self.dem_d = wp.array(dem_f32, dtype=float, device="cuda")
            self.deformed_mask_d = wp.zeros(dem_np.shape, dtype=float, device="cuda")
            
            # Pre-allocate GPU arrays for 4 wheels
            self.pos_d = wp.zeros(4, dtype=wp.vec3, device="cuda")
            self.force_d = wp.zeros(4, dtype=float, device="cuda")
            self.sinkage_d = wp.zeros(4, dtype=float, device="cuda")
            self.terrain_z_d = wp.zeros(4, dtype=float, device="cuda")
            
            # Pre-allocate CPU staging arrays (avoids GPU malloc per frame)
            self.pos_staging = wp.zeros(4, dtype=wp.vec3, device="cpu")
            self.sink_staging = wp.zeros(4, dtype=float, device="cpu")
            self.tz_staging = wp.zeros(4, dtype=float, device="cpu")
            
            self._initialized = True
            print(f"[DeformationEngine] GPU arrays initialized: DEM {dem_np.shape}, "
                  f"Grid {self.grid_w}x{self.grid_h}")

    def deform_async(self, positions_np: np.ndarray, forces_np: np.ndarray, aabb_np: np.ndarray):
        """
        비동기 지형 변형 (CPU 동기화 없이 GPU에서만 실행)
        """
        if not self._initialized:
            return
            
        # 임시 배열 생성 후 복사 (Warp의 copy 함수 사용)
        temp_pos = wp.array(positions_np, dtype=wp.vec3, device="cuda")
        temp_force = wp.array(forces_np, dtype=float, device="cuda")
        wp.copy(self.pos_d, temp_pos)
        wp.copy(self.force_d, temp_force)
        
        min_xy = aabb_np[0]
        max_xy = aabb_np[1]

        start_x = max(0, int((min_xy[0] - self.x_offset_total) / self.resolution))
        start_y = max(0, int((min_xy[1] - self.y_offset_total) / self.resolution))
        end_x = min(self.grid_w, int((max_xy[0] - self.x_offset_total) / self.resolution) + 1)
        end_y = min(self.grid_h, int((max_xy[1] - self.y_offset_total) / self.resolution) + 1)

        dim_x = end_x - start_x
        dim_y = end_y - start_y
        
        if dim_x <= 0 or dim_y <= 0:
            return
        
        # 영역 저장
        self.last_start_x = start_x
        self.last_start_y = start_y
        self.last_end_x = end_x
        self.last_end_y = end_y

        # GPU 커널 실행 (비동기 - 즉시 반환)
        wp.launch(
            kernel=deform_terrain_kernel,
            dim=(dim_x, dim_y),
            inputs=[
                self.dem_d, self.deformed_mask_d,
                self.pos_d, self.force_d, 
                self.resolution,
                self.x_offset_total, self.y_offset_total,
                start_x, start_y,
                self.amplitude_slope, self.max_deformation, 
                self.footprint_radius
            ],
            device="cuda"
        )

    def sync_to_cpu(self, terrain_manager):
        """
        GPU 결과를 CPU로 동기화하고 메시 업데이트 (가끔 호출)
        """
        if not self._initialized:
            return
            
        # GPU -> CPU 동기화 (여기서만 블로킹)
        wp.synchronize()
        
        start_x = self.last_start_x
        start_y = self.last_start_y
        end_x = self.last_end_x
        end_y = self.last_end_y
        
        if end_x <= start_x or end_y <= start_y:
            return
        
        # 변형된 패치만 복사
        dem_patch_d = self.dem_d[start_y:end_y, start_x:end_x]
        dem_patch_np = dem_patch_d.numpy()
        
        # CPU DEM 업데이트
        patch_h, patch_w = dem_patch_np.shape
        terrain_manager.current_dem[start_y:start_y+patch_h, start_x:start_x+patch_w] = dem_patch_np
        
        # 메시 업데이트
        terrain_manager.update_mesh_patch(dem_patch_np, start_x, start_y)

    def deform(self, dem_np: np.ndarray, positions_np: np.ndarray, forces_np: np.ndarray, aabb_np: np.ndarray) -> tuple:
        """
        기존 동기식 API (하위 호환성 유지)
        """
        self._init_gpu_arrays(dem_np)
        
        pos_d = wp.array(positions_np, dtype=wp.vec3, device="cuda")
        force_d = wp.array(forces_np, dtype=float, device="cuda")
        
        min_xy = aabb_np[0]
        max_xy = aabb_np[1]

        start_x = max(0, int((min_xy[0] - self.x_offset_total) / self.resolution))
        start_y = max(0, int((min_xy[1] - self.y_offset_total) / self.resolution))
        end_x = min(self.grid_w, int((max_xy[0] - self.x_offset_total) / self.resolution) + 1)
        end_y = min(self.grid_h, int((max_xy[1] - self.y_offset_total) / self.resolution) + 1)

        dim_x = end_x - start_x
        dim_y = end_y - start_y
        
        if dim_x <= 0 or dim_y <= 0:
            return (np.array([[]]), 0, 0)

        wp.launch(
            kernel=deform_terrain_kernel,
            dim=(dim_x, dim_y),
            inputs=[
                self.dem_d, self.deformed_mask_d,
                pos_d, force_d, 
                self.resolution,
                self.x_offset_total, self.y_offset_total,
                start_x, start_y,
                self.amplitude_slope, self.max_deformation, 
                self.footprint_radius
            ],
            device="cuda"
        )
        
        dem_patch_d = self.dem_d[start_y : end_y, start_x : end_x]
        dem_patch_np = dem_patch_d.numpy()
        
        return (dem_patch_np, start_x, start_y)
    def deform_with_sinkage(self, positions_np: np.ndarray, 
                            sinkages_np: np.ndarray, terrain_z_np: np.ndarray,
                            wheel_radius: float, aabb_np: np.ndarray) -> tuple:
        """
        Wheel Track 생성을 위한 지형 변형 (비동기 최적화 버전)
        
        Args:
            positions_np: 바퀴 위치 (4, 3) [x, y, wheel_bottom_z] float32
            sinkages_np: 각 바퀴의 침하량 (4,) [meters] float32
            terrain_z_np: 각 바퀴 위치의 원래 지형 높이 (4,) [meters] float32
            wheel_radius: 바퀴 반지름 [meters]
            aabb_np: 변형 영역 AABB [[min_x, min_y], [max_x, max_y]]
        """
        if not self._initialized:
            return
        
        min_xy = aabb_np[0]
        max_xy = aabb_np[1]

        # 그리드 좌표 계산
        start_x = max(0, int((min_xy[0] - self.x_offset_total) / self.resolution))
        start_y = max(0, int((min_xy[1] - self.y_offset_total) / self.resolution))
        end_x = min(self.grid_w, int((max_xy[0] - self.x_offset_total) / self.resolution) + 1)
        end_y = min(self.grid_h, int((max_xy[1] - self.y_offset_total) / self.resolution) + 1)

        dim_x = end_x - start_x
        dim_y = end_y - start_y
        
        if dim_x <= 0 or dim_y <= 0:
            return

        # CPU staging → GPU (zero-allocation path)
        self.pos_staging.numpy()[:] = positions_np.astype(np.float32)
        self.sink_staging.numpy()[:] = sinkages_np.astype(np.float32)
        self.tz_staging.numpy()[:] = terrain_z_np.astype(np.float32)
        wp.copy(self.pos_d, self.pos_staging)
        wp.copy(self.sinkage_d, self.sink_staging)
        wp.copy(self.terrain_z_d, self.tz_staging)

        # 바퀴 풋프린트 파라미터
        wheel_width = self.cfg.footprint.width  # 0.1m
        footprint_radius = wheel_width * 1.5  # 바퀴 폭의 1.5배 반경
        
        # 접촉 임계값 (바퀴 반지름의 10% = 약 1.6cm)
        contact_threshold = wheel_radius * 0.1

        # 커널 실행 (비동기 - 블로킹 없음)
        wp.launch(
            kernel=deform_terrain_kernel_sinkage,
            dim=(dim_x, dim_y),
            inputs=[
                self.dem_d, self.deformed_mask_d,
                self.pos_d, self.terrain_z_d, self.sinkage_d,
                self.resolution,
                self.x_offset_total, self.y_offset_total,
                start_x, start_y,
                footprint_radius, wheel_radius, contact_threshold
            ],
            device="cuda"
        )
        
        # 변형 영역 확장 (이전 영역과 합산)
        if self._has_pending_deform:
            self.last_start_x = min(self.last_start_x, start_x)
            self.last_start_y = min(self.last_start_y, start_y)
            self.last_end_x = max(self.last_end_x, end_x)
            self.last_end_y = max(self.last_end_y, end_y)
        else:
            self.last_start_x = start_x
            self.last_start_y = start_y
            self.last_end_x = end_x
            self.last_end_y = end_y
        
        self._has_pending_deform = True
        return (np.array([[]]), start_x, start_y)
    
    def sync_deformation(self, terrain_manager) -> bool:
        """
        GPU 변형 결과를 CPU로 동기화하고 메시 업데이트
        성능을 위해 가끔만 호출 (예: 30프레임마다)
        
        Returns:
            bool: 업데이트 성공 여부
        """
        if not self._initialized or not self._has_pending_deform:
            return False
            
        start_x = self.last_start_x
        start_y = self.last_start_y
        end_x = self.last_end_x
        end_y = self.last_end_y
        
        if end_x <= start_x or end_y <= start_y:
            self._has_pending_deform = False
            return False
        
        # GPU -> CPU 동기화 (여기서만 블로킹)
        wp.synchronize()
        
        # 변형된 패치 복사
        dem_patch_d = self.dem_d[start_y:end_y, start_x:end_x]
        dem_patch_np = dem_patch_d.numpy()
        
        # CPU DEM 업데이트
        patch_h, patch_w = dem_patch_np.shape
        terrain_manager.current_dem[start_y:start_y+patch_h, start_x:start_x+patch_w] = dem_patch_np
        
        # 메시 업데이트
        terrain_manager.update_mesh_patch(dem_patch_np, start_x, start_y)
        
        # dirty 플래그 리셋
        self._has_pending_deform = False
        
        return True