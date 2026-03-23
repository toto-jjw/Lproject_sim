# Lproject_sim/src/robots/rover.py
from typing import Optional
import numpy as np
from .robot_base import RobotBase
from pxr import UsdLux, UsdGeom, Gf, Sdf, UsdPhysics
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
import rclpy
from std_msgs.msg import Float32

class Rover(RobotBase):
    """
    Control class for the Rover robot in Isaac Sim.
    Manages drive and steering joints.
    """
    def __init__(
        self,
        prim_path: str,
        name: str = "rover",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,

        robot_param = None # [New] Pass params for graph generation
    ) -> None:
        
        self.robot_param = robot_param

        
        if usd_path:
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
            
            # [Refactor]
            # We no longer remove existing OmniGraphs. We rely on the USD file's internal graphs.
            # self._remove_existing_action_graphs(prim_path)
            
        super().__init__(prim_path=prim_path, name=name, usd_path=usd_path, position=position, orientation=orientation)
        
        # Husky specific joint names (adjust based on actual USD)
        self.wheel_joint_names = ["front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel"]
        self.wheel_dof_indices = []
        self.wheel_indices_array = np.array([], dtype=np.int32)
        
        # Solar Panel
        self.solar_joint_name = "solar_panel_joint" 
        self.solar_dof_index = None
        self._solar_sub = None

    def initialize(self, physics_sim_view=None) -> None:
        super().initialize(physics_sim_view)
        
        # [Improvement] Dynamic Joint Discovery
        all_dof_names = self.dof_names
        found_indices = [self.get_dof_index(name) for name in self.wheel_joint_names]
        
        if any(idx is None for idx in found_indices):
            print(f"Warning: Hardcoded joint names {self.wheel_joint_names} not found. Searching dynamically...")
            new_indices = []
            new_names = []
            for i, name in enumerate(all_dof_names):
                if "wheel" in name.lower():
                    new_indices.append(i)
                    new_names.append(name)
            
            if len(new_indices) == 4:
                print(f"Found 4 wheel joints dynamically: {new_names}")
                new_names.sort()
                self.wheel_dof_indices = [self.get_dof_index(n) for n in new_names]
                self.wheel_joint_names = new_names
            else:
                print(f"Error: Could not find 4 wheel joints. Found: {new_names}")
        else:
            self.wheel_dof_indices = found_indices
        
        # wheel_indices_array 업데이트 (먼지 방출에 필요)
        if self.wheel_dof_indices:
            self.wheel_indices_array = np.array(self.wheel_dof_indices, dtype=np.int32)
            print(f"[{self.name}] Wheel indices array set: {self.wheel_indices_array}")

        print(f"[{self.name}] Initialization Complete (Using built-in USD OmniGraphs).")
        

        try:
            pass 
        except Exception as e:
            print(f"Error initializing ROS sub: {e}")

        ## Add LED Headlight
        self._add_led_headlight()
        
        ## [핵심 수정] 휠 조인트 Drive 파라미터 수정 - 흔들림 방지
        self._fix_wheel_drive_params()

    def _fix_wheel_drive_params(self):
        """
        USD 파일의 휠 조인트 Angular Drive 파라미터를 안정적인 값으로 수정합니다.
        
        문제: USD에서 Damping=10,000,000, Stiffness=100 설정은 
              극단적으로 높은 감쇠와 낮은 강성의 조합으로 스틱-슬립 진동을 유발합니다.
              
        해결: 속도 제어에 적합한 파라미터로 수정합니다.
              - Stiffness: 0 (순수 속도 제어)
              - Damping: 1000~5000 (적절한 저항)
        """
        stage = get_current_stage()
        
        # 휠 조인트 경로 패턴들 (USD 구조에 따라 다를 수 있음)
        wheel_joint_patterns = [
            f"{self.prim_path}/front_left_wheel",
            f"{self.prim_path}/front_right_wheel", 
            f"{self.prim_path}/rear_left_wheel",
            f"{self.prim_path}/rear_right_wheel",
            # 대체 경로 패턴
            f"{self.prim_path}/base_link/front_left_wheel",
            f"{self.prim_path}/base_link/front_right_wheel",
            f"{self.prim_path}/base_link/rear_left_wheel",
            f"{self.prim_path}/base_link/rear_right_wheel",
        ]
        
        # 새로운 Drive 파라미터 (안정적인 속도 제어용)
        # Z축 흔들림 방지를 위한 최적화된 값
        NEW_STIFFNESS = 0.0        # 순수 속도 제어 (위치 유지 안함)
        NEW_DAMPING = 5000.0       # 충분한 감쇠로 진동 억제 (기존 1000 → 5000)
        NEW_MAX_FORCE = 500.0      # 최대 토크 제한 (급격한 힘 방지)
        
        fixed_count = 0
        
        for joint_path in wheel_joint_patterns:
            prim = stage.GetPrimAtPath(joint_path)
            if not prim.IsValid():
                continue
            
            # RevoluteJoint인지 확인
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                # 자식 프림에서 찾기
                for child in prim.GetChildren():
                    if child.IsA(UsdPhysics.RevoluteJoint):
                        prim = child
                        break
                else:
                    continue
            
            try:
                # Angular Drive API 가져오기 또는 생성
                drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not drive_api:
                    drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
                
                # 기존 값 출력 (디버깅용)
                old_stiffness = drive_api.GetStiffnessAttr().Get()
                old_damping = drive_api.GetDampingAttr().Get()
                
                # 새 값 설정
                drive_api.GetStiffnessAttr().Set(NEW_STIFFNESS)
                drive_api.GetDampingAttr().Set(NEW_DAMPING)
                
                # MaxForce가 있으면 설정
                max_force_attr = drive_api.GetMaxForceAttr()
                if max_force_attr:
                    max_force_attr.Set(NEW_MAX_FORCE)
                
                print(f"[{self.name}] Fixed wheel drive: {prim.GetPath()}")
                print(f"  Stiffness: {old_stiffness} -> {NEW_STIFFNESS}")
                print(f"  Damping: {old_damping} -> {NEW_DAMPING}")
                fixed_count += 1
                
            except Exception as e:
                print(f"[{self.name}] Warning: Failed to fix drive params for {joint_path}: {e}")
        
        if fixed_count > 0:
            print(f"[{self.name}] Successfully fixed {fixed_count} wheel joint drive parameters.")
        else:
            print(f"[{self.name}] Warning: No wheel joints found to fix. Trying alternative method...")
            self._fix_wheel_drive_params_by_search()
        
        # [추가] RigidBody Damping 설정 - Z축 진동 억제
        self._apply_rigid_body_damping()
    
    def _apply_rigid_body_damping(self):
        """
        로버의 RigidBody에 Linear/Angular Damping을 적용하여 진동을 억제합니다.
        
        Z축 진동의 근본 원인:
        - PhysX에서 RigidBody가 미세한 힘에도 계속 움직임
        - Damping이 없으면 에너지가 사라지지 않아 진동 지속
        """
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(self.prim_path)
        
        if not root_prim.IsValid():
            return
        
        try:
            from pxr import PhysxSchema, Usd
            
            # Damping 설정값
            # Linear Damping: 직선 운동 감쇠 (Z축 진동 억제)
            # Angular Damping: 회전 운동 감쇠
            LINEAR_DAMPING = 0.1    # 약한 선형 감쇠 (너무 크면 이동이 느려짐)
            ANGULAR_DAMPING = 0.05  # 약한 회전 감쇠
            
            fixed_count = 0
            for prim in Usd.PrimRange(root_prim):
                # RigidBodyAPI가 있는 프림 찾기
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    try:
                        # PhysxRigidBodyAPI 적용 (damping 설정용)
                        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                            rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
                        else:
                            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                        
                        # 기존 값 확인
                        existing_linear = rigid_api.GetLinearDampingAttr().Get()
                        existing_angular = rigid_api.GetAngularDampingAttr().Get()
                        
                        # 값이 너무 작으면 설정
                        if existing_linear is None or existing_linear < LINEAR_DAMPING:
                            rigid_api.CreateLinearDampingAttr().Set(LINEAR_DAMPING)
                        if existing_angular is None or existing_angular < ANGULAR_DAMPING:
                            rigid_api.CreateAngularDampingAttr().Set(ANGULAR_DAMPING)
                        
                        fixed_count += 1
                        
                    except Exception as inner_e:
                        pass
            
            if fixed_count > 0:
                print(f"[{self.name}] Applied RigidBody damping to {fixed_count} bodies")
                print(f"  Linear: {LINEAR_DAMPING}, Angular: {ANGULAR_DAMPING}")
                
        except Exception as e:
            print(f"[{self.name}] Warning: Failed to apply rigid body damping: {e}")
    

    def _add_led_headlight(self):
        """
        Adds a high-intensity LED headlight to the front of the rover.
        Specs: Front Top, 45 deg cone, 30 deg down tilt, 6.67 W/m^2/sr
        """
        # Define Path: Attach to base_link to move with rover
        light_path = f"{self.prim_path}/base_link/LED_Headlight"
        stage = get_current_stage()
       
        # Check if exists
        if stage.GetPrimAtPath(light_path):
            return

        # Create Sphere Light
        light = UsdLux.SphereLight.Define(stage, light_path)
       
        # 1. Geometry & Intensity
        # Radius 5cm for the LED cluster
        light.GetRadiusAttr().Set(0.5)
        light.GetIntensityAttr().Set(20000.0)
        light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))  # Pure white (neutral color)
        light.GetExposureAttr().Set(0.0)
       
        # 2. Shaping (Cone)
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr().Set(45.0) # 45 degree cone
        shaping.CreateShapingConeSoftnessAttr().Set(0.1)
        # Focus/FocusTint not strictly needed but available
       
        # 3. Position & Orientation
        # Front Top of the rover.
        # Rover Base Link Frame: X-Forward, Y-Left, Z-Up (Standard ROS)
        # Position: x=0.4 (Front), z=0.3 (Top), y=0.0 (Center) -> Tweak based on visual mesh
        xform = UsdGeom.Xformable(light.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.0, 0.8))
       
        xform.AddRotateYOp().Set(-60.0)

    def _fix_wheel_drive_params_by_search(self):
        """
        대체 방법: USD 트리를 순회하며 모든 RevoluteJoint를 찾아 Drive 파라미터를 수정합니다.
        """
        stage = get_current_stage()
        root_prim = stage.GetPrimAtPath(self.prim_path)
        
        if not root_prim.IsValid():
            print(f"[{self.name}] Error: Robot prim not found at {self.prim_path}")
            return
        
        NEW_STIFFNESS = 0.0
        NEW_DAMPING = 1000.0
        fixed_count = 0
        
        # 재귀적으로 모든 프림 순회
        from pxr import Usd
        for prim in Usd.PrimRange(root_prim):
            prim_name = prim.GetName().lower()
            
            # 휠 관련 프림인지 확인
            if 'wheel' not in prim_name:
                continue
            
            # RevoluteJoint 또는 Drive API가 있는지 확인
            try:
                # PhysX Drive API 직접 확인
                drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive_api:
                    old_stiffness = drive_api.GetStiffnessAttr().Get()
                    old_damping = drive_api.GetDampingAttr().Get()
                    
                    drive_api.GetStiffnessAttr().Set(NEW_STIFFNESS)
                    drive_api.GetDampingAttr().Set(NEW_DAMPING)
                    
                    print(f"[{self.name}] Fixed (search): {prim.GetPath()}")
                    print(f"  Stiffness: {old_stiffness} -> {NEW_STIFFNESS}")
                    print(f"  Damping: {old_damping} -> {NEW_DAMPING}")
                    fixed_count += 1
            except Exception as e:
                pass  # Drive API가 없으면 무시
        
        print(f"[{self.name}] Search method fixed {fixed_count} joints.")



    def get_wheel_indices(self):
        return self.wheel_dof_indices

    def get_wheel_positions(self):
        """Returns list of joint angles (radians) for each wheel."""
        if len(self.wheel_indices_array) == 0:
            return [0.0]*4
        return self.get_joint_positions(joint_indices=self.wheel_indices_array)


        
    def get_wheel_angular_velocities(self):
        """Get angular velocities of wheels."""
        if len(self.wheel_indices_array) == 0:
            return [0.0]*4
        return self.get_joint_velocities(joint_indices=self.wheel_indices_array)
        


    def apply_force(self, force: np.ndarray, position: Optional[np.ndarray] = None) -> None:
        """
        Apply an external force to the robot's base link.
        Args:
           force (np.ndarray): Force vector [fx, fy, fz] in global frame (or local? usually global API).
           position (Optional[np.ndarray]): Position to apply force (default: Center of Mass).
        """
        
        if self._articulation_view:
            from omni.isaac.dynamic_control import _dynamic_control
            if not  hasattr(self, "_dc"):
                self._dc = _dynamic_control.acquire_dynamic_control_interface()
                print(f"DEBUG DC Methods: {dir(self._dc)}")
            if not hasattr(self, "body_handle") or self.body_handle ==_dynamic_control.INVALID_HANDLE:
                path=f"{self.prim_path}/base_link"
                self.body_handle = self._dc.get_rigid_body(path)
                if self.body_handle == _dynamic_control.INVALID_HANDLE:
                    self.body_handle = self._dc.get_rigid_body(self.prim_path)
            if self.body_handle != _dynamic_control.INVALID_HANDLE:
                f_list= [float(force[0]), float(force[1]), float(force[2])]
                self._dc.apply_body_force(self.body_handle, f_list, [0, 0, 0], True)
            else:
                pass
        else:
            print(f"[{self.name}] Warning: Cannot apply force, Robot not initialized.")
