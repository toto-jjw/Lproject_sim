# Lproject_sim/src/physics/physics_manager.py

import numpy as np
import math
import random
from src.physics.terramechanics import TerramechanicsSolver
from src.physics.wheel_track_renderer import WheelTrackRenderer
from src.environment.dust_manager import DustManager
from src.config.physics_config import RobotParameter, TerrainMechanicalParameter

class PhysicsManager:
    """
    Manages global physics interactions: Terramechanics, Wheel Tracks, and Dust Emission.
    
    Hybrid deformation approach:
      1. WheelTrackRenderer (ribbon mesh) → fast visual wheel tracks
      2. CPU DEM update → accurate get_heights() for terramechanics
      3. No USD mesh modification → avoids 4M vertex upload bottleneck
    """
    def __init__(self, terrain_manager, env_config, asset_cfg: dict):
        self.tm = terrain_manager
        self.env_config = env_config
        self.asset_cfg = asset_cfg

        # Parameters
        self.robot_param = RobotParameter()
        self.terrain_param = TerrainMechanicalParameter()
        
        # Solvers
        self.solver = TerramechanicsSolver(self.robot_param, self.terrain_param)
        
        # Wheel Track Renderer (바퀴자국 시각화)
        self.wheel_track_renderer = None
        
        self.dust_manager = None
        if self.env_config.get("dust", {}).get("enabled", False):
            self.dust_manager = DustManager(self.tm.stage, self.asset_cfg)

        self.tm_enabled = self.env_config.get("terramechanics", {}).get("enabled", False)
        self.deform_enabled = self.env_config.get("deformation", {}).get("enabled", False)
        
        # Deformation config
        deform_cfg = self.env_config.get("deformation", {})
        self.use_track_renderer = deform_cfg.get("use_track_renderer", True)
        self.deform_dem = deform_cfg.get("deform_dem", True)       # CPU DEM만 업데이트
        self.update_mesh = deform_cfg.get("update_mesh", False)    # USD 메시도 수정 (느림)
        self.dem_update_interval = deform_cfg.get("dem_update_interval", 15)
        self.visual_scale = deform_cfg.get("visual_scale", 5.0)
        
        # 바퀴자국 렌더러 초기화
        if self.deform_enabled and self.use_track_renderer:
            track_cfg = deform_cfg.get("track_renderer", {})
            self.wheel_track_renderer = WheelTrackRenderer(
                stage=self.tm.stage,
                terrain_manager=self.tm,
                num_wheels=4,
                track_width=track_cfg.get("track_width", 0.1),
                max_points_per_wheel=track_cfg.get("max_points", 500),
                min_distance=track_cfg.get("min_distance", 0.05),
                track_height_offset=track_cfg.get("height_offset", 0.002)
            )
        
        self.frame_count = 0
        self.debug_interval = 120  # 약 1초마다 디버그 출력 (120Hz 기준)
        
        # [추가] 저항력 스무딩을 위한 상태 변수
        self._prev_resistance = {}  # robot_name -> previous resistance magnitude
        self._stopped_frames = {}   # robot_name -> frames since stopped
        
        # 상태 로깅
        print(f"[PhysicsManager] Terramechanics: {'ENABLED' if self.tm_enabled else 'DISABLED'}")
        print(f"[PhysicsManager] Deformation: {'ENABLED' if self.deform_enabled else 'DISABLED'}")
        if self.deform_enabled:
            modes = []
            if self.use_track_renderer: modes.append("RibbonMesh(visual)")
            if self.deform_dem: modes.append("DEM(physics)")
            if self.update_mesh: modes.append("USD-Mesh(slow!)")
            print(f"[PhysicsManager] Deformation modes: {' + '.join(modes)}")
            print(f"[PhysicsManager]   dem_update_interval={self.dem_update_interval}, visual_scale={self.visual_scale}")

    def update(self, dt, robots):
        """
        Update physics for all robots.
        """
        for robot_ctx in robots:
            self._process_robot(robot_ctx, dt)
            
        if self.dust_manager:
            self.dust_manager.update(dt)

    def _process_robot(self, robot_ctx, dt):
        rover = robot_ctx.rover
        
        # 1. Get State
        pos, ori = rover.get_world_pose()
        lin_vel = rover.get_linear_velocity()
        wheel_omegas = rover.get_wheel_angular_velocities()
        
        # 2. Estimate Wheel Positions (from USD wheel_link centers)
        wx = self.robot_param.wheel_x_offset   # 0.256m
        wy = self.robot_param.wheel_y_offset   # 0.2854m
        offsets = np.array([
            [ wx,  wy], [ wx, -wy],   # front_left, front_right
            [-wx,  wy], [-wx, -wy]    # rear_left,  rear_right
        ])
        
        q = ori
        yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        
        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        rotated_offsets = offsets @ R.T
        wheel_positions_xy = pos[:2] + rotated_offsets
        
        terrain_z_values = self.tm.get_heights(wheel_positions_xy)
        
        # ============================================================
        # Wong-Bekker Sinkage Model (정적 침하량 계산)
        # ============================================================
        # Bekker 압력-침하 관계: p = (k_c/b + k_phi) * z^n
        # 역으로: z = (p / (k_c/b + k_phi))^(1/n)
        # 
        # 여기서:
        #   p = W / A (접지압력)
        #   W = 바퀴당 하중 (로버 무게 / 바퀴 수)
        #   A = 접지 면적 (근사: 2*sqrt(r*z) * b)
        #   k_c, k_phi, n = 토양 파라미터
        #   b = 바퀴 폭
        #   r = 바퀴 반지름
        # ============================================================
        
        # 토양 파라미터 (달 레골리스)
        k_c = self.terrain_param.k_c      # 1400 N/m^(n+1)
        k_phi = self.terrain_param.k_phi  # 820000 N/m^(n+2)
        n = self.terrain_param.n          # 1.0
        
        # 바퀴 파라미터
        r = self.robot_param.wheel_radius  # 0.165 m
        b = self.robot_param.wheel_width   # 0.1 m
        
        # 바퀴당 하중 (달 중력 1.62 m/s^2 기준)
        gravity = 1.62
        wheel_load = (self.robot_param.mass * gravity) / self.robot_param.num_wheels
        
        # Bekker 모듈러스: k = k_c/b + k_phi
        k_bekker = (k_c / b) + k_phi
        
        # 정적 침하량 계산 (반복법으로 근사)
        # 초기 근사: z = (W / (k * sqrt(r) * b))^(2/(2n+1))
        # 단순화된 버전 사용
        z_static = (wheel_load / (k_bekker * b * math.sqrt(r))) ** (1.0 / (n + 0.5))
        z_static = min(z_static, r * 0.3)  # 최대 침하량 제한 (반지름의 30%)
        
        # 기하학적 침하량: 바퀴 바닥이 지형 아래로 들어간 깊이
        # wheel_center_world_z = root_z + wheel_center_z_local
        # wheel_bottom_world_z = wheel_center_world_z - wheel_radius
        wheel_center_z_offset = self.robot_param.wheel_center_z  # 0.17775m
        wheel_bottom = pos[2] + wheel_center_z_offset - r
        geometric_sinkages = np.maximum(0.0, terrain_z_values - wheel_bottom)
        
        # Bekker 침하량과 기하학적 침하량 중 작은 값 사용 (물리적으로 합리적)
        sinkages = np.minimum(geometric_sinkages, z_static)
        sinkages = np.maximum(sinkages, 0.001)  # 최소값 보장
        
        v_mag = np.linalg.norm(lin_vel[:2])
        
        # ============================================================
        # 각 바퀴의 개별 속도 계산 (바퀴 각속도 기반)
        # ============================================================
        # 차동 조향 (differential steering): 좌우 바퀴 속도 차이로 회전
        # v_wheel = omega_wheel * r_wheel (바퀴 표면 선속도)
        # ============================================================
        wheel_radius = self.robot_param.wheel_radius
        
        # 각 바퀴의 선형 속도 (각속도 * 반경)
        wheel_velocities = np.zeros(4)
        if wheel_omegas is not None and len(wheel_omegas) >= 4:
            for i in range(4):
                wheel_velocities[i] = abs(wheel_omegas[i]) * wheel_radius
        else:
            wheel_velocities = np.full(4, v_mag)
        
        wheel_world_positions = np.column_stack((wheel_positions_xy, terrain_z_values))

        normal_forces = np.zeros(self.robot_param.num_wheels)
        
        # [공통] 이동 상태 판별 (Terramechanics와 Deformation 모두 사용)
        speed = np.linalg.norm(lin_vel[:2])
        is_moving = speed > 0.1

        # 3. Solve Terramechanics (if enabled)
        # Wong-Bekker 모델 기반 Rolling Resistance 계산
        if self.tm_enabled:
            # ============================================================
            # Wong-Bekker Rolling Resistance Model
            # ============================================================
            # 압축 저항력 (Compaction Resistance):
            #   R_c = b * integral(p * dz) from 0 to z
            #       = b * (k_c/b + k_phi) * z^(n+1) / (n+1)
            #
            # Bulldozing 저항력 (토양 밀어내기):
            #   R_b = 0.5 * rho * g * z^2 * b * N_gamma
            #   N_gamma = tan^2(45 + phi/2) - 1 (지지력 계수)
            #
            # 총 저항력: R_total = R_c + R_b
            # ============================================================
            
            avg_sinkage = np.mean(sinkages)
            sinkage_cm = avg_sinkage * 100.0
            
            # 토양 파라미터
            rho = self.terrain_param.rho       # 1600 kg/m^3
            phi = self.terrain_param.phi       # 0.5 rad (~28.6°)
            
            # 1. Compaction Resistance (압축 저항)
            R_compaction_per_wheel = b * k_bekker * (avg_sinkage ** (n + 1)) / (n + 1)
            
            # 2. Bulldozing Resistance (토양 밀어내기 저항)
            N_gamma = (math.tan(math.pi/4 + phi/2) ** 2) - 1
            R_bulldozing_per_wheel = 0.5 * rho * gravity * (avg_sinkage ** 2) * b * N_gamma
            
            # 총 저항력 (모든 바퀴)
            R_total_per_wheel = R_compaction_per_wheel + R_bulldozing_per_wheel
            target_resistance = R_total_per_wheel * self.robot_param.num_wheels
            
            # 스케일 조정 (실제 시뮬레이션에서 적절한 느낌을 위해)
            # Wong-Bekker 모델은 실제 값이 작을 수 있음, 0.5배 스케일
            scale_factor = 0.5
            target_resistance *= scale_factor
            
            # 최대 저항력 제한 (로버 무게의 15% 이내 - 기존 20%에서 감소)
            max_resistance = self.robot_param.mass * gravity * 0.15
            target_resistance = min(target_resistance, max_resistance)
            
            speed = np.linalg.norm(lin_vel[:2])
            
            # 로버의 전진 방향 (yaw 각도 기반)
            forward_dir = np.array([cos_yaw, sin_yaw])
            
            # 속도의 전진 방향 성분만 추출 (전진/후진 판별)
            forward_speed = np.dot(lin_vel[:2], forward_dir)
            
            # 이동 감지: 속도 기반 (is_moving은 이미 위에서 계산됨)
            robot_name = rover.name
            
            # 정지 프레임 카운터 관리
            if robot_name not in self._stopped_frames:
                self._stopped_frames[robot_name] = 0
                self._prev_resistance[robot_name] = 0.0
            
            if not is_moving:
                self._stopped_frames[robot_name] += 1
            else:
                self._stopped_frames[robot_name] = 0
            
            # 정지 상태 판단 (30프레임 = 약 0.25초 동안 멈춰있으면 정지)
            is_stopped = self._stopped_frames[robot_name] > 30
            
            if is_stopped:
                # 정지 상태 - 저항력 리셋
                self._prev_resistance[robot_name] = 0.0
                resistance_force_mag = 0.0
            elif self._stopped_frames[robot_name] > 0:
                # 정지 직후 (0~30프레임) - 저항력 급감
                resistance_force_mag = self._prev_resistance[robot_name] * 0.8
                self._prev_resistance[robot_name] = resistance_force_mag
            else:
                # 이동 중 - 저항력 스무딩 (급격한 변화 방지)
                prev = self._prev_resistance[robot_name]
                smoothing = 0.15  # 스무딩 계수 (기존 0.1 → 0.15로 약간 빠르게)
                resistance_force_mag = prev + smoothing * (target_resistance - prev)
                self._prev_resistance[robot_name] = resistance_force_mag
            
            # 저항력 적용 (이동 중이고 저항력이 의미있을 때만)
            if is_moving and resistance_force_mag > 0.1:
                sign = np.sign(forward_speed)
                resistance_vec = np.array([-sign * forward_dir[0] * resistance_force_mag,
                                           -sign * forward_dir[1] * resistance_force_mag,
                                           0.0])
                rover.apply_force(resistance_vec)
            
            # Normal forces for deformation (Bekker 기반)
            normal_forces.fill(wheel_load)
            
            # 디버그 출력 (약 1초마다)
            if self.frame_count % self.debug_interval == 0:
                state = "MOVING" if is_moving else f"STOP({self._stopped_frames[robot_name]})"
                print(f"[Terramechanics] {state} | Spd: {forward_speed:.2f}m/s | "
                      f"Sink: {sinkage_cm:.2f}cm (static:{z_static*100:.2f}cm) | "
                      f"R_c: {R_compaction_per_wheel:.2f}N | R_b: {R_bulldozing_per_wheel:.2f}N | "
                      f"Total: {resistance_force_mag:.1f}N")

        self.frame_count += 1
        
        # 5. Wheel Track / Deformation (Hybrid)
        # ============================================================
        # Layer 1: Ribbon mesh → instant visual wheel tracks (no mesh edit)
        # Layer 2: CPU DEM stamp → updates current_dem for get_heights()
        # Layer 3: (optional) USD mesh update → very slow, off by default
        # ============================================================
        if self.deform_enabled and is_moving:
            # Layer 1: Visual groove mesh tracks (U-shaped depression)
            if self.wheel_track_renderer:
                forward_dir = np.array([cos_yaw, sin_yaw])
                # Pass visual-scaled sinkages for groove depth
                vis_sinkages = sinkages * self.visual_scale
                self.wheel_track_renderer.update(
                    wheel_positions=wheel_positions_xy,
                    forward_dir=forward_dir,
                    terrain_z_values=terrain_z_values,
                    is_moving=is_moving,
                    sinkages=vis_sinkages
                )
            
            # Layer 2: Lightweight CPU DEM deformation
            if self.deform_dem and self.frame_count % self.dem_update_interval == 0:
                self._stamp_dem_at_wheels(
                    wheel_positions_xy, sinkages, terrain_z_values
                )

        # 6. Dust Emission (Rooster Tail 패턴)
        # ============================================================
        # 달 먼지 방출 모델 (아폴로 LRV 관찰 기반)
        # - 각 바퀴의 회전 방향과 속도에 따라 먼지 발생
        # - 바퀴가 지면을 밀어내는 방향으로 먼지 방출
        # - 달 진공: 공기 저항 없이 탄도 궤적
        # ============================================================
        if self.dust_manager and wheel_omegas is not None:
            # 로버 전진 방향 (yaw 각도 기반)
            forward_dir = np.array([cos_yaw, sin_yaw])
            right_dir = np.array([sin_yaw, -cos_yaw])  # 오른쪽 방향
            
            for i, (wx, wy, wz) in enumerate(wheel_world_positions):
                # 바퀴 속도 임계값
                if wheel_velocities[i] > 0.02 and sinkages[i] > 0.0:
                    # 방출량: 속도와 침하량에 비례
                    base_count = int(wheel_velocities[i] * 15.0)
                    sinkage_factor = 1.0 + sinkages[i] * 30.0
                    count = int(base_count * sinkage_factor)
                    count = min(count, 1000)  # 최대 1000개/바퀴
                    
                    if count <= 0:
                        continue
                    
                    # ============================================================
                    # 바퀴 회전 방향에 따른 먼지 방출 방향
                    # ============================================================
                    # wheel_omega > 0: 바퀴가 전진 방향으로 회전 → 먼지는 뒤로
                    # wheel_omega < 0: 바퀴가 후진 방향으로 회전 → 먼지는 앞으로
                    # ============================================================
                    wheel_omega = wheel_omegas[i] if len(wheel_omegas) > i else 0.0
                    
                    # 바퀴 회전 방향에 따른 먼지 방향
                    if abs(wheel_omega) > 0.01:
                        # 바퀴가 전진 회전(+) → 먼지는 뒤쪽(-forward_dir)
                        # 바퀴가 후진 회전(-) → 먼지는 앞쪽(+forward_dir)
                        eject_dir = -np.sign(wheel_omega) * forward_dir
                    else:
                        # 바퀴가 거의 안 돌면 먼지 없음
                        continue
                    
                    # 약간의 좌우 분산 추가
                    scatter = np.random.uniform(-0.2, 0.2)
                    eject_dir = eject_dir + right_dir * scatter
                    eject_dir = eject_dir / (np.linalg.norm(eject_dir) + 0.001)
                    
                    # 방출 속도 벡터 (먼지 관리자에 전달)
                    vel_vec = np.array([eject_dir[0], eject_dir[1], 0.0])
                    
                    self.dust_manager.emit(np.array([wx, wy, wz]), vel_vec, count=count)
                    
                    # 태양광 패널 먼지 축적
                    if "solar_panel" in robot_ctx.components:
                        accumulation_rate = 0.00001 * count * dt
                        robot_ctx.components["solar_panel"].accumulate_dust(accumulation_rate)
    
    # ================================================================
    # Lightweight CPU DEM stamping (no GPU, no USD mesh modification)
    # ================================================================
    def _stamp_dem_at_wheels(self, wheel_xy: np.ndarray, sinkages: np.ndarray,
                             terrain_z: np.ndarray):
        """
        Stamp wheel footprint into current_dem using pure NumPy.
        ~0.1ms per call (vs ~50ms for Warp+USD sync).
        
        Only modifies the terrain_manager.current_dem array so that
        subsequent get_heights() calls return the deformed surface.
        The visual mesh is NOT modified (ribbon tracks handle visuals).
        """
        dem = self.tm.current_dem
        res = self.tm.cfg.resolution
        x_off = self.tm.x_offset
        y_off = self.tm.y_offset
        grid_h, grid_w = dem.shape
        
        wheel_width = self.robot_param.wheel_width  # 0.1m
        fp_radius = wheel_width * 1.2               # stamp radius
        fp_cells = int(fp_radius / res) + 1         # ~3 cells at 0.05m res
        
        for i in range(len(wheel_xy)):
            sink = sinkages[i] * self.visual_scale
            if sink < 0.0005:  # skip < 0.5mm
                continue
            
            # World XY → grid indices
            gx = int((wheel_xy[i, 0] - x_off) / res)
            gy = int((wheel_xy[i, 1] - y_off) / res)
            
            # Clamp patch bounds
            x0 = max(0, gx - fp_cells)
            x1 = min(grid_w, gx + fp_cells + 1)
            y0 = max(0, gy - fp_cells)
            y1 = min(grid_h, gy + fp_cells + 1)
            
            if x1 <= x0 or y1 <= y0:
                continue
            
            # Build distance-based Gaussian falloff for this patch
            xs = np.arange(x0, x1) * res + x_off - wheel_xy[i, 0]
            ys = np.arange(y0, y1) * res + y_off - wheel_xy[i, 1]
            dx, dy = np.meshgrid(xs, ys)
            dist_sq = dx * dx + dy * dy
            
            mask = dist_sq < (fp_radius * fp_radius)
            if not np.any(mask):
                continue
            
            # Gaussian sinkage profile (center = deepest)
            falloff = np.exp(-2.0 * dist_sq / (fp_radius * fp_radius))
            depth = sink * falloff * mask
            
            # Only lower terrain (never raise)
            patch = dem[y0:y1, x0:x1]
            dem[y0:y1, x0:x1] = np.minimum(patch, patch - depth)
        
        # Optionally push to USD mesh (very expensive — off by default)
        if self.update_mesh:
            # Only push every ~300 frames to avoid stalling
            if self.frame_count % 300 == 0:
                self.tm.update_mesh_from_dem(dem, flip_input=True)
    
    def cleanup(self):
        if self.dust_manager:
            self.dust_manager.cleanup()
        if self.wheel_track_renderer:
            self.wheel_track_renderer.cleanup()

