# Lproject_sim/src/core/sim_manager.py
import omni.kit.app

# class SimulationManager:
#     def __init__(self):
#         # Assumes SimulationApp is started externally (e.g. in main.py)
#         from isaacsim.core.api.world import World
#         self.world = World(stage_units_in_meters=1.0)
#         #self.world.scene.add_default_ground_plane()


class SimulationManager:
    def __init__(self, physics_dt: float = 1.0/60.0, gravity: list = None):
        # Assumes SimulationApp is started externally (e.g. in main.py)
        from isaacsim.core.api.world import World
        # 1. Initialize World
        gravity_tensor = -9.81 # Default
        if gravity and len(gravity) == 3:
            # Check Z component
            gravity_tensor = gravity[2]

        self.world = World(stage_units_in_meters=1.0, physics_dt=physics_dt)
        
        # [핵심 수정] Default Ground Plane을 추가하지 않음 - 커스텀 지형과 충돌 문제 방지
        # self.world.scene.add_default_ground_plane()  # 비활성화!
        
        # [추가] PhysX Scene 설정 강화 - 로버 안정성 향상
        self._configure_physics_scene()

        if gravity:
            
            import numpy as np
            #g_vec = np.array(gravity)
            
            self.world.get_physics_context().set_gravity(gravity[2]) # Fallback to Z-gravity mostly used
            from pxr import UsdPhysics
            stage = self.world.stage
            scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
            if scene_prim:
                scene_api = UsdPhysics.Scene(scene_prim)
                scene_api.CreateGravityDirectionAttr().Set(tuple(gravity)) 
                
            print(f"[SimulationManager] Applied Gravity: {gravity}")

    def _configure_physics_scene(self):
        """
        PhysX Scene의 solver 설정을 강화하여 충돌 안정성을 높입니다.
        
        문제: 기본 solver iteration이 낮으면 휠-지면 접촉이 불안정해집니다.
        해결: Position/Velocity iteration 증가, Contact offset 조정
        """
        try:
            from pxr import PhysxSchema, UsdPhysics
            stage = self.world.stage
            
            scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
            if not scene_prim or not scene_prim.IsValid():
                print("[SimulationManager] Warning: PhysicsScene not found")
                return
            
            # PhysxSceneAPI 적용
            physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
            
            # === Solver 설정 ===
            physx_scene_api.CreateSolverTypeAttr().Set("TGS")  # TGS solver가 더 안정적
            
            # === [핵심] Solver Iterations 증가 (Z축 흔들림 해결) ===
            # Position iterations: 접촉점 위치 해결 (기본값: 4 → 16)
            # Velocity iterations: 속도 제약 해결 (기본값: 1 → 4)
            # 높을수록 안정적이지만 성능 비용 증가
            physics_scene = UsdPhysics.Scene(scene_prim)
            
            # PhysX-specific iteration counts
            physx_scene_api.CreateMinPositionIterationCountAttr().Set(16)
            physx_scene_api.CreateMinVelocityIterationCountAttr().Set(4)
            
            # GPU dynamics 비활성화 (CPU가 더 안정적)
            physx_scene_api.CreateEnableGPUDynamicsAttr().Set(False)
            
            # === Contact 설정 (떨림 방지) ===
            # Bounce threshold: 이 속도 이하의 충돌은 반발하지 않음
            physx_scene_api.CreateBounceThresholdVelocityAttr().Set(0.5)  # 기존 0.2 → 0.5
            
            # Friction offset threshold
            physx_scene_api.CreateFrictionOffsetThresholdAttr().Set(0.01)
            
            # === Stabilization 설정 (정지 시 흔들림 방지) ===
            # Enable stabilization: 정지 상태의 객체를 안정화
            physx_scene_api.CreateEnableStabilizationAttr().Set(True)
            
            # === [핵심] Enhanced Determinism (결정적 시뮬레이션) ===
            # 이 옵션은 더 정밀한 물리 시뮬레이션을 위해 필요
            try:
                physx_scene_api.CreateEnableEnhancedDeterminismAttr().Set(True)
            except:
                pass  # 구버전 PhysX에서는 지원 안 될 수 있음
            
            print("[SimulationManager] PhysX Scene configured for stability:")
            print("  - Solver: TGS")
            print("  - Position Iterations: 16")
            print("  - Velocity Iterations: 4")
            print("  - Bounce Threshold: 0.5 m/s")
            print("  - Stabilization: Enabled")
            
        except Exception as e:
            print(f"[SimulationManager] Warning: Failed to configure PhysX scene: {e}")


    def step(self, render: bool = True):
        self.world.step(render=render)

    def close(self):
        # App close is handled in main.py
        pass

    def is_running(self):
        return omni.kit.app.get_app().is_running()

    def is_playing(self):
        """Returns True if the simulation timeline is playing."""
        return self.world.is_playing()
