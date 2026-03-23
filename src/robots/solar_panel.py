# Lproject_sim/src/robots/solar_panel.py
import numpy as np
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade, Sdf
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid

class SolarPanel:
    """
    Manages a rotatable solar panel attached to the rover.
    """
    def __init__(self, parent_path: str, name: str = "SolarPanel", position: np.array = np.array([0, 0, 0.3])):
        self.stage = get_current_stage()
        self.path = f"{parent_path}/{name}"
        self.name = name
        self.parent_path = parent_path
        self.panel_position = position
        
        # Create the solar panel prim if it doesn't exist
        if not is_prim_path_valid(self.path):
            # Create a Xform for the joint/rotation
            self.xform = UsdGeom.Xform.Define(self.stage, self.path)
            UsdGeom.Xformable(self.xform).AddTranslateOp().Set(Gf.Vec3f(*position))
            
            # Create the visual mesh (Cube for now)
            panel_geom_path = f"{self.path}/PanelVisual"
            self.panel_geom = UsdGeom.Cube.Define(self.stage, panel_geom_path)
            
            # Scale it to look like a panel (flat)
            # Size 2.0 -> Scale to e.g. 0.5m x 0.5m x 0.02m
            # Cube is 2 units size. Scale 0.25 -> 0.5m
            self.panel_geom.AddScaleOp().Set(Gf.Vec3f(0.25, 0.25, 0.01))
            self.panel_geom.AddTranslateOp().Set(Gf.Vec3f(0, 0, 0.0)) # Center
            
            # Add blue color
            self.panel_geom.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 0.5)])
            
        self.current_angle = 0.0 # Radians
        self.rotation_op = None
        
        # Get rotation op
        prim = get_prim_at_path(self.path)
        xform = UsdGeom.Xformable(prim)
        # We want to rotate around Z (Up)
        # Check if rotate op exists
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                self.rotation_op = op
                break
        
        if not self.rotation_op:
            self.rotation_op = xform.AddRotateZOp()
            
        self.update_vertical_geometry()
        
        # 지지대 생성
        self._create_support_pole()
        
        self.dust_accumulation = 0.0 # 0.0 (Clean) to 1.0 (Fully Covered)

    def accumulate_dust(self, amount: float):
        """Accumulate dust on the panel."""
        self.dust_accumulation = min(1.0, self.dust_accumulation + amount)
        
        # Visual feedback: Darken/Brown the front panel (solar cell side)
        # Original Blue: (0.05, 0.1, 0.5)
        # Dust Brown: (0.5, 0.4, 0.3)
        blue = np.array([0.05, 0.1, 0.5])
        brown = np.array([0.5, 0.4, 0.3])
        color = blue * (1.0 - self.dust_accumulation) + brown * self.dust_accumulation
        
        front_face_path = f"{self.path}/PanelVisual/FrontFace"
        if is_prim_path_valid(front_face_path):
            prim = get_prim_at_path(front_face_path)
            geom = UsdGeom.Gprim(prim)
            geom.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

    def get_efficiency_factor(self) -> float:
        """Return efficiency factor (1.0 = clean, 0.0 = covered)."""
        return max(0.0, 1.0 - self.dust_accumulation)

    def set_angle(self, angle_rad: float):
        """Set the angle of the solar panel (radians). Range: -pi to pi."""
        # Clamp to +/- 180 degrees (pi radians)
        angle_rad = max(-np.pi, min(np.pi, angle_rad))
        
        # Optimization: Only update if changed significantly
        if abs(angle_rad - self.current_angle) < 0.001:
            return
            
        self.current_angle = angle_rad
        # USD uses degrees
        self.rotation_op.Set(np.degrees(angle_rad))
        
    def get_angle(self) -> float:
        """Get current angle in radians."""
        return self.current_angle

    def get_normal_vector(self) -> np.array:
        """
        Get the normal vector of the panel in WORLD frame.
        """
        return self.get_world_normal()
        
    def update_vertical_geometry(self):
        """Update geometry to be vertical and facing backwards (180 deg).
        Creates front (blue) and back (gray) faces for visual distinction.
        """
        panel_geom_path = f"{self.path}/PanelVisual"
        
        # 기존 PanelVisual 삭제
        if is_prim_path_valid(panel_geom_path):
            self.stage.RemovePrim(panel_geom_path)
        
        # 패널 컨테이너 생성
        panel_container = UsdGeom.Xform.Define(self.stage, panel_geom_path)
        xform = UsdGeom.Xformable(panel_container)
        xform.ClearXformOpOrder()
        
        # 단일 Transform 사용 (translate + rotate 통합) - 경고 방지
        transform_op = xform.AddTransformOp()
        mat = Gf.Matrix4d()
        mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), 180.0))
        mat.SetTranslateOnly(Gf.Vec3d(0, 0, 0.25))
        transform_op.Set(mat)
        
        # 패널 두께와 크기 정의
        panel_thickness = 0.02  # 2cm 두께
        panel_width = 0.5       # Y방향 50cm
        panel_height = 0.5      # Z방향 50cm
        
        # 앞면 (파란색 - 태양광 수광면) - Scale만 사용, Translate는 offset으로 처리
        front_path = f"{panel_geom_path}/FrontFace"
        front_cube = UsdGeom.Cube.Define(self.stage, front_path)
        front_xform = UsdGeom.Xformable(front_cube)
        front_xform.ClearXformOpOrder()
        front_xform.AddTranslateOp().Set(Gf.Vec3f(panel_thickness/4, 0, 0))
        front_xform.AddScaleOp().Set(Gf.Vec3f(panel_thickness/4, panel_width/2, panel_height/2))
        front_cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.1, 0.5)])
        
        # 뒷면 (회색 - 구조물)
        back_path = f"{panel_geom_path}/BackFace"
        back_cube = UsdGeom.Cube.Define(self.stage, back_path)
        back_xform = UsdGeom.Xformable(back_cube)
        back_xform.ClearXformOpOrder()
        back_xform.AddTranslateOp().Set(Gf.Vec3f(-panel_thickness/4, 0, 0))
        back_xform.AddScaleOp().Set(Gf.Vec3f(panel_thickness/4, panel_width/2, panel_height/2))
        back_cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.4, 0.4)])
        
        # 테두리/프레임 (은색)
        frame_path = f"{panel_geom_path}/Frame"
        frame_cube = UsdGeom.Cube.Define(self.stage, frame_path)
        frame_xform = UsdGeom.Xformable(frame_cube)
        frame_xform.ClearXformOpOrder()
        frame_xform.AddScaleOp().Set(Gf.Vec3f(panel_thickness/8, (panel_width + 0.02)/2, (panel_height + 0.02)/2))
        frame_cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.7, 0.7, 0.7)])
        
    def _create_support_pole(self):
        """패널을 base_link에 연결하는 지지대 생성"""
        pole_path = f"{self.parent_path}/SolarPanelPole"
        
        if is_prim_path_valid(pole_path):
            return  # 이미 존재하면 스킵
        
        # 지지대 높이 계산 (패널 위치까지)
        pole_height = self.panel_position[2]  # 패널 Z 위치
        pole_radius = 0.02  # 2cm 반경
        
        # 원통형 지지대 생성
        pole = UsdGeom.Cylinder.Define(self.stage, pole_path)
        pole_xform = UsdGeom.Xformable(pole)
        pole_xform.ClearXformOpOrder()
        
        # 위치만 설정 (불필요한 회전 제거 - Cylinder는 기본적으로 Z축 정렬)
        pole_xform.AddTranslateOp().Set(Gf.Vec3f(
            float(self.panel_position[0]),
            float(self.panel_position[1]),
            float(pole_height / 2)  # 절반 높이에 중심
        ))
        
        # 크기 조정 (반경, 높이)
        pole.GetRadiusAttr().Set(pole_radius)
        pole.GetHeightAttr().Set(pole_height)
        
        # 은색/알루미늄 색상
        pole.GetDisplayColorAttr().Set([Gf.Vec3f(0.6, 0.6, 0.65)])
        
        # 회전 베이스 (패널이 회전할 때 시각적으로 회전축 표시)
        base_path = f"{self.parent_path}/SolarPanelBase"
        if not is_prim_path_valid(base_path):
            base = UsdGeom.Cylinder.Define(self.stage, base_path)
            base_xform = UsdGeom.Xformable(base)
            base_xform.ClearXformOpOrder()
            base_xform.AddTranslateOp().Set(Gf.Vec3f(
                float(self.panel_position[0]),
                float(self.panel_position[1]),
                float(pole_height - 0.02)  # 지지대 상단
            ))
            base.GetRadiusAttr().Set(0.04)  # 4cm 반경
            base.GetHeightAttr().Set(0.04)  # 4cm 높이
            base.GetDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.3, 0.35)])  # 어두운 회색

    def get_world_normal(self) -> np.array:
        """
        Returns the normal vector of the panel in World Frame.
        """
        # Local Normal (assuming X-axis is normal)
        c = np.cos(self.current_angle)
        s = np.sin(self.current_angle)
        local_normal = np.array([c, s, 0.0])
        
        # Get Parent World Transform
        prim = get_prim_at_path(self.path)
        xform = UsdGeom.Xformable(prim)
        mat = xform.ComputeLocalToWorldTransform(0)
        
        # Vec3d
        x_axis = Gf.Vec3d(1, 0, 0)
        # TransformDir (only rotation)
        world_normal = mat.TransformDir(x_axis)
        
        return np.array([world_normal[0], world_normal[1], world_normal[2]])

    def get_world_position(self) -> np.array:
        """
        Returns the position of the panel center in World Frame.
        Used for shadow detection raycast origin.
        """
        prim = get_prim_at_path(self.path)
        if not prim.IsValid():
            return np.array([0.0, 0.0, 0.5])
        
        xform = UsdGeom.Xformable(prim)
        mat = xform.ComputeLocalToWorldTransform(0)
        translation = mat.ExtractTranslation()
        
        return np.array([translation[0], translation[1], translation[2]], dtype=np.float64)
