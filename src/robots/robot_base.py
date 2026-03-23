# Lproject_sim/src/robots/robot_base.py
from typing import Optional, Tuple
import numpy as np
from omni.isaac.core.articulations import Articulation
class RobotBase(Articulation):
    def __init__(
        self,
        prim_path: str,
        name: str = "robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the RobotBase.

        Args:
            prim_path (str): The prim path of the robot.
            name (str): The name of the robot.
            usd_path (str, optional): The path to the USD file to load.
            position (np.ndarray, optional): The initial position of the robot.
            orientation (np.ndarray, optional): The initial orientation of the robot.
        """
        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation)
        self.usd_path = usd_path

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize the robot.
        """
        super().initialize(physics_sim_view)
        # Additional initialization if needed

    def apply_wheel_velocity(self, left_velocity: float, right_velocity: float) -> None:
        """
        Apply velocity to the wheels. Must be implemented by subclasses.
        """
        raise NotImplementedError
