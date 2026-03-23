# src/sensors/sun_sensor.py (최종 검증 완료된 코드)

import numpy as np
from pxr import UsdGeom, Gf, Usd
from isaacsim.core.utils.stage import get_current_stage
import omni # omni.usd.get_context()를 위해 필요

class SunSensor:
    """
    Virtual Sun Sensor that calculates the vector to the sun in the sensor's local frame.
    Also performs shadow detection using PhysX raycast.
    """
    def __init__(self, prim_path: str, sun_prim_path: str = "/World/Sun", name: str = "SunSensor", position: np.array = None, orientation: np.array = None):
        self.prim_path = prim_path
        self.sun_prim_path = sun_prim_path
        self.name = name
        self.stage = get_current_stage()
        
        # Shadow detection state
        self._in_shadow = False
        self._shadow_factor = 1.0  # 1.0 = full sun, 0.0 = full shadow
        self._physx_interface = None
        
        if not self.stage.GetPrimAtPath(prim_path):
            UsdGeom.Xform.Define(self.stage, prim_path)
            
        self.set_pose(position, orientation)
        self._init_physx_raycast()
        
    def set_pose(self, position: np.array, orientation: np.array):
        """Set sensor pose using standard transform operations."""
        prim = self.stage.GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            return
            
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        
        if position is not None:
            xform.AddTranslateOp().Set(Gf.Vec3f(*position))
            
        if orientation is not None:
            q = Gf.Quatf(orientation[0], orientation[1], orientation[2], orientation[3])
            xform.AddOrientOp().Set(q)

    def get_sun_vector(self):
        """
        Returns the normalized vector pointing TO the sun in the SENSOR's local frame.
        Used for solar panel tracking control.
        """
        time = Usd.TimeCode.Default()

        sun_prim = self.stage.GetPrimAtPath(self.sun_prim_path)
        if not sun_prim.IsValid():
            return np.array([0.0, 0.0, 1.0])
            
        sun_xform = UsdGeom.Xformable(sun_prim)
        sun_world_transform = sun_xform.ComputeLocalToWorldTransform(time)
        vec_to_sun_world = sun_world_transform.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
        
        sensor_prim = self.stage.GetPrimAtPath(self.prim_path)
        if not sensor_prim.IsValid():
            return np.array([0.0, 0.0, 1.0])
            
        sensor_xform = UsdGeom.Xformable(sensor_prim)
        sensor_world_transform = sensor_xform.ComputeLocalToWorldTransform(time)
        world_to_sensor_transform = sensor_world_transform.GetInverse()
        
        vec_to_sun_sensor = world_to_sensor_transform.TransformDir(vec_to_sun_world)
        
        result = np.array(vec_to_sun_sensor, dtype=np.float64)
        if np.isnan(result).any():
             return np.array([0.0, 0.0, 1.0])
             
        return result

    def get_sun_direction_world(self):
        """
        Returns the normalized direction FROM which sunlight is coming in WORLD frame.
        This is the negative of the "vector pointing to sun" - i.e., the light ray direction.
        Used for solar power calculation (dot product with panel normal).
        """
        time = Usd.TimeCode.Default()

        sun_prim = self.stage.GetPrimAtPath(self.sun_prim_path)
        if not sun_prim.IsValid():
            return np.array([0.0, 0.0, -1.0])  # Default: light coming from above
            
        sun_xform = UsdGeom.Xformable(sun_prim)
        sun_world_transform = sun_xform.ComputeLocalToWorldTransform(time)
        
        # Sun's local Z-axis points "where the light goes" (direction the sun is facing)
        # For a DistantLight, this is the light direction
        sun_direction = sun_world_transform.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
        
        # Negate to get "where light comes FROM" for dot product with panel normal
        # When panel faces sun: panel_normal · (-sun_direction) = 1 (max power)
        light_from_direction = -np.array([sun_direction[0], sun_direction[1], sun_direction[2]], dtype=np.float64)
        
        if np.isnan(light_from_direction).any():
            return np.array([0.0, 0.0, -1.0])
             
        return light_from_direction

    def _init_physx_raycast(self):
        """Initialize PhysX scene query interface for raycast."""
        try:
            from omni.physx import get_physx_scene_query_interface
            self._physx_interface = get_physx_scene_query_interface()
            print(f"[{self.name}] PhysX raycast interface initialized for shadow detection")
        except Exception as e:
            print(f"[{self.name}] Warning: Could not init PhysX raycast: {e}")
            self._physx_interface = None

    def _get_sensor_world_position(self) -> np.ndarray:
        """Get sensor position in world frame."""
        time = Usd.TimeCode.Default()
        sensor_prim = self.stage.GetPrimAtPath(self.prim_path)
        if not sensor_prim.IsValid():
            return np.array([0.0, 0.0, 0.5])
        
        sensor_xform = UsdGeom.Xformable(sensor_prim)
        world_transform = sensor_xform.ComputeLocalToWorldTransform(time)
        translation = world_transform.ExtractTranslation()
        return np.array([translation[0], translation[1], translation[2]], dtype=np.float64)

    def check_shadow(self, panel_world_position: np.ndarray = None) -> bool:
        """
        Check if the sensor/panel is in shadow by raycasting toward the sun.
        
        Args:
            panel_world_position: Optional position to check. If None, uses sensor position.
            
        Returns:
            True if in shadow (raycast hit obstacle), False if in sunlight.
        """
        if self._physx_interface is None:
            self._in_shadow = False
            self._shadow_factor = 1.0
            return False
        
        # Get origin position (panel or sensor)
        if panel_world_position is not None:
            origin = panel_world_position
        else:
            origin = self._get_sensor_world_position()
        
        # Get sun direction (where light comes FROM)
        sun_dir_world = self.get_sun_direction_world()
        
        # Ray direction should point TOWARD the sun (opposite of light direction)
        ray_direction = -sun_dir_world
        
        # Normalize
        ray_dir_norm = np.linalg.norm(ray_direction)
        if ray_dir_norm < 1e-6:
            self._in_shadow = False
            self._shadow_factor = 1.0
            return False
        ray_direction = ray_direction / ray_dir_norm
        
        # Offset origin slightly in ray direction to avoid self-intersection
        origin_offset = origin + ray_direction * 0.1
        
        # Raycast parameters
        ray_distance = 1000.0  # Large enough to reach any obstacle
        
        try:
            # Use PhysX scene query for raycast
            hit_info = self._physx_interface.raycast_closest(
                tuple(origin_offset),
                tuple(ray_direction),
                ray_distance
            )
            
            if hit_info["hit"]:
                # Something blocks the sun
                hit_distance = hit_info.get("distance", 0)
                hit_prim_path = hit_info.get("rigidBody", "unknown")
                
                # Ignore very close hits (could be self or parent robot)
                if hit_distance > 0.5:
                    self._in_shadow = True
                    self._shadow_factor = 0.0
                    # print(f"[{self.name}] Shadow detected! Hit: {hit_prim_path} at {hit_distance:.1f}m")
                    return True
                    
            # No blocking obstacle
            self._in_shadow = False
            self._shadow_factor = 1.0
            return False
            
        except Exception as e:
            # print(f"[{self.name}] Raycast error: {e}")
            self._in_shadow = False
            self._shadow_factor = 1.0
            return False

    def get_shadow_factor(self) -> float:
        """
        Returns shadow factor for power calculation.
        1.0 = full sunlight, 0.0 = full shadow.
        
        Call check_shadow() first to update this value.
        """
        return self._shadow_factor
    
    def is_in_shadow(self) -> bool:
        """
        Returns whether the sensor is currently in shadow.
        
        Call check_shadow() first to update this value.
        """
        return self._in_shadow
