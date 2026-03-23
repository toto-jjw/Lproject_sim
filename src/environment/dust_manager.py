# Lproject_sim/src/environment/dust_manager.py
from isaacsim.core.utils.prims import create_prim, define_prim
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom, Gf, Sdf, UsdShade, Usd, Vt
import numpy as np
import random
import warp as wp
import omni.warp
import os


# Initialize Warp
wp.init()

@wp.kernel
def integrate_particles(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    life: wp.array(dtype=float),
    active: wp.array(dtype=int),
    dt: float,
    gravity: float
):
    tid = wp.tid()
    if active[tid] == 0:
        return
        
    # Gravity
    v = vel[tid]
    v = v + wp.vec3(0.0, 0.0, -gravity * dt)
    vel[tid] = v
    
    # Position
    p = pos[tid]
    p = p + v * dt
    pos[tid] = p
    
    # Lifetime
    l = life[tid] - dt
    life[tid] = l
    
    # Kill
    if l <= 0.0 or p[2] < -0.5:
        active[tid] = 0
        # Move out of view (optional)
        pos[tid] = wp.vec3(0.0, 0.0, -10.0)

@wp.kernel
def init_particles(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    life: wp.array(dtype=float),
    active: wp.array(dtype=int),
    new_pos: wp.array(dtype=wp.vec3),
    new_vel: wp.array(dtype=wp.vec3),
    new_life: wp.array(dtype=float),
    start_idx: int,
    count: int,
    max_particles: int
):
    tid = wp.tid()
    # tid goes from 0 to count-1
    
    idx = (start_idx + tid) % max_particles
    
    pos[idx] = new_pos[tid]
    vel[idx] = new_vel[tid]
    life[idx] = new_life[tid]
    active[idx] = 1

class DustManager:
    """
    달 먼지 (Lunar Dust / Regolith) 시뮬레이션
    
    아폴로 미션 관찰 결과 기반:
    - 달 먼지는 진공에서 탄도 궤적을 따름 (공기 저항 없음)
    - "Rooster Tail" 패턴: 바퀴 뒤쪽으로 분사되는 먼지 궤적
    - 입자 크기: 20-100 μm (미세 분진) - 시각화를 위해 확대
    - 달 중력: 1.62 m/s² (지구의 1/6)
    """
    
    # 달 먼지 물리 상수
    LUNAR_GRAVITY = 1.62          # m/s²
    MIN_EJECT_SPEED = 0.5         # 최소 방출 속도 (m/s)
    MAX_EJECT_SPEED = 1.5         # 최대 방출 속도 (m/s)
    EJECT_ANGLE_MIN = 0.0        # 최소 방출 각도 (도)
    EJECT_ANGLE_MAX = 50.0        # 최대 방출 각도 (도)
    PARTICLE_LIFETIME_MIN = 0.8   # 최소 수명 (초)
    PARTICLE_LIFETIME_MAX = 2.5   # 최대 수명 (초)
    
    def __init__(self, stage, asset_cfg: dict, max_particles: int = 100000):
        self.stage = stage
        self.max_particles = max_particles
        self.instancer_path = "/World/Dust/Instancer"
        self.asset_cfg = asset_cfg
        self._frame_count = 0
        
        # Create PointInstancer
        self.instancer = UsdGeom.PointInstancer.Define(self.stage, self.instancer_path)
        
        # Create prototype (크기를 시각적으로 보이게 설정)
        # 실제 달 먼지: 20-100 μm, 시각화용: 1-3 cm
        proto_path = f"{self.instancer_path}/ProtoSphere"
        proto = UsdGeom.Sphere.Define(self.stage, proto_path)
        proto.GetRadiusAttr().Set(0.0005)  # 0.5mm radius (더 큰 입자로 가시성 향상)
        
        # Apply material (High roughness for regolith look)
        self._create_dust_material(proto_path)
        
        # Add prototype to instancer
        self.instancer.GetPrototypesRel().AddTarget(proto_path)
        
        # Warp Arrays (GPU)
        self.pos_d = wp.zeros(self.max_particles, dtype=wp.vec3, device="cuda")
        self.vel_d = wp.zeros(self.max_particles, dtype=wp.vec3, device="cuda")
        self.life_d = wp.zeros(self.max_particles, dtype=float, device="cuda")
        self.active_d = wp.zeros(self.max_particles, dtype=int, device="cuda") # 0=Inactive, 1=Active
        
        self.next_idx = 0
        

    def _create_dust_material(self, prim_path):
        """지형과 동일한 MDL 파일을 올바르게 참조하여 먼지 재질을 생성합니다."""
        
        # YAML 파일에서 지정한 MDL 파일 경로를 가져옵니다.
        material_path_str = self.asset_cfg.get("material_path")

        if material_path_str and os.path.exists(material_path_str):
            print(f"Applying MDL material to dust particles from: {material_path_str}")
            
            # A. 재질(Material) 프리미티브를 생성합니다.
            #    경로가 겹치지 않도록 고유한 이름을 사용합니다.
            material = UsdShade.Material.Define(self.stage, "/World/Looks/DustParticleMaterial")
            
            # B. 셰이더(Shader) 프리미티브를 생성합니다.
            shader = UsdShade.Shader.Define(self.stage, "/World/Looks/DustParticleMaterial/Shader")
            
            # C. [핵심] 셰이더의 소스를 MDL 파일로 지정합니다.
            shader.SetSourceAsset(material_path_str, "mdl")
            
            # D. [핵심] MDL 파일 안의 'LunarRegolith8k' 재질을 사용하도록 지정합니다.
            shader.GetPrim().CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("LunarRegolith8k")
            
            # E. [핵심] 셰이더의 'out' 출력을 명시적으로 생성합니다.
            mdl_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            
            # F. [핵심] 재질의 표면 출력을 위에서 생성한 'mdl_output'에 연결합니다.
            material.CreateSurfaceOutput().ConnectToSource(mdl_output)
            
            # G. 최종적으로 완성된 재질을 먼지 입자 프로토타입에 바인딩합니다.
            prim_to_bind = self.stage.GetPrimAtPath(prim_path)
            UsdShade.MaterialBindingAPI(prim_to_bind).Bind(material)

        else:
            # MDL 파일이 없을 경우를 대비한 Fallback 로직
            print("Warning: Dust material MDL file not found. Applying a default grey material.")
            # ... (이 부분의 코드는 그대로 유지)



    def emit(self, position: np.array, velocity: np.array, count: int = 100):
        """
        Rooster Tail 패턴으로 먼지 방출 (아폴로 LRV 관찰 기반)
        
        달 환경 특성:
        - 진공: 공기 저항 없음 → 순수 탄도 궤적
        - 낮은 중력: 1.62 m/s² → 더 높이, 더 멀리 비행
        - 정전기: 입자가 표면에 쉽게 부착
        
        Args:
            position: 바퀴 접지점 위치 [x, y, z]
            velocity: 기본 속도 벡터 (방향 참고용)
            count: 방출 입자 수
        """
        if count <= 0: 
            return
        
        count = min(count, self.max_particles // 10)
        
        # ============================================================
        # Rooster Tail 물리 모델 (아폴로 LRV 기반)
        # ============================================================
        # - 방출 각도: 수직에서 20-50° (뒤쪽-위쪽 방향)
        # - 수평 분산: ±30° (좌우 퍼짐)
        # - 속도: 기본 속도의 0.3~1.0배
        # ============================================================
        
        # 방출 각도 (수직면에서, 라디안)
        elevation_angles = np.random.uniform(
            np.radians(self.EJECT_ANGLE_MIN),
            np.radians(self.EJECT_ANGLE_MAX),
            count
        )
        
        # 수평 분산 각도 (좌우 퍼짐)
        azimuth_angles = np.random.uniform(-np.pi/6, np.pi/6, count)  # ±30°
        
        # ============================================================
        # 방출 속도 계산
        # ============================================================
        # 달 중력에서 최대 높이: h = v²sin²θ/(2g)
        # 0.4 m/s, 45°에서: h = (0.4*0.707)²/(2*1.62) ≈ 2.5cm
        # ============================================================
        eject_speeds = np.random.uniform(
            self.MIN_EJECT_SPEED,
            self.MAX_EJECT_SPEED,
            count
        )
        
        # 속도 벡터 계산
        cos_elev = np.cos(elevation_angles)
        sin_elev = np.sin(elevation_angles)
        cos_azim = np.cos(azimuth_angles)
        sin_azim = np.sin(azimuth_angles)
        
        # ============================================================
        # 먼지 방출 방향: velocity 벡터 방향으로 (바퀴가 밀어내는 방향)
        # - 전진 시: velocity가 앞쪽 → 먼지도 앞쪽으로 튀어나감 (실제로는 뒤로 날림)
        # - 후진 시: velocity가 뒤쪽 → 먼지도 뒤쪽으로 튀어나감 (실제로는 앞으로 날림)
        # - velocity는 physics_manager에서 바퀴 이동 방향의 반대로 전달됨
        # ============================================================
        vel_mag = np.linalg.norm(velocity[:2])
        if vel_mag > 0.001:
            eject_dir = velocity[:2] / vel_mag  # velocity 방향 그대로 사용
        else:
            # 속도가 거의 없으면 랜덤 방향
            random_angle = np.random.uniform(0, 2*np.pi)
            eject_dir = np.array([np.cos(random_angle), np.sin(random_angle)])
        
        right = np.array([-eject_dir[1], eject_dir[0]])  # 수직 방향
        
        # 3D 속도 벡터 구성
        new_vel = np.zeros((count, 3))
        new_vel[:, 0] = (eject_dir[0] * cos_azim + right[0] * sin_azim) * cos_elev * eject_speeds
        new_vel[:, 1] = (eject_dir[1] * cos_azim + right[1] * sin_azim) * cos_elev * eject_speeds
        new_vel[:, 2] = sin_elev * eject_speeds  # 위쪽 성분
        
        # 위치 (바퀴 접지점 주변, 약간의 분산)
        new_pos = np.tile(position, (count, 1))
        new_pos += np.random.uniform(-0.03, 0.03, (count, 3))  # 적당한 분산
        new_pos[:, 2] = position[2] + 0.01  # 지면 약간 위에서 시작
        
        # 수명 (탄도 비행 시간 기반)
        # 달 중력에서: t = 2 * v_z / g
        flight_times = 2 * np.abs(new_vel[:, 2]) / self.LUNAR_GRAVITY
        new_life = np.clip(flight_times + 0.5, self.PARTICLE_LIFETIME_MIN, self.PARTICLE_LIFETIME_MAX)
        
        # GPU 업로드
        new_pos_d = wp.array(new_pos, dtype=wp.vec3, device="cuda")
        new_vel_d = wp.array(new_vel, dtype=wp.vec3, device="cuda")
        new_life_d = wp.array(new_life, dtype=float, device="cuda")
        
        # Launch Init Kernel
        wp.launch(
            kernel=init_particles,
            dim=count,
            inputs=[
                self.pos_d, self.vel_d, self.life_d, self.active_d,
                new_pos_d, new_vel_d, new_life_d,
                self.next_idx, count, self.max_particles
            ],
            device="cuda"
        )
        
        self.next_idx = (self.next_idx + count) % self.max_particles

    def update(self, dt: float):
        """Update particle physics via Warp and sync to USD."""
        self._frame_count += 1
        
        # 1. Physics Step (GPU)
        wp.launch(
            kernel=integrate_particles,
            dim=self.max_particles,
            inputs=[
                self.pos_d,
                self.vel_d,
                self.life_d,
                self.active_d,
                dt,
                self.LUNAR_GRAVITY
            ],
            device="cuda"
        )
        
        # 2. Rendering Update (Host Sync)
        active_np = self.active_d.numpy()
        
        # 기본 시간 코드 사용 (시간 샘플링 경고 방지)
        time_code = Usd.TimeCode.Default()
        
        if not np.any(active_np):
            self.instancer.GetPositionsAttr().Set([], time_code)
            self.instancer.GetProtoIndicesAttr().Set([], time_code)
            return
            
        pos_np = self.pos_d.numpy()
        
        # Filter Active
        active_mask = (active_np == 1)
        active_positions = pos_np[active_mask]
        
        if len(active_positions) > 0:
            # Vt.Vec3fArray로 변환하여 시간 샘플링 경고 방지
            positions_vt = Vt.Vec3fArray.FromNumpy(active_positions.astype(np.float32))
            indices_vt = Vt.IntArray([0] * len(active_positions))
            
            self.instancer.GetPositionsAttr().Set(positions_vt, time_code)
            self.instancer.GetProtoIndicesAttr().Set(indices_vt, time_code)
           
        else:
            self.instancer.GetPositionsAttr().Set(Vt.Vec3fArray(), time_code)
            self.instancer.GetProtoIndicesAttr().Set(Vt.IntArray(), time_code)
    
    def cleanup(self):
        """리소스 정리"""
        time_code = Usd.TimeCode.Default()
        self.instancer.GetPositionsAttr().Set(Vt.Vec3fArray(), time_code)
        self.instancer.GetProtoIndicesAttr().Set(Vt.IntArray(), time_code)
        print("[DustManager] Cleaned up")
