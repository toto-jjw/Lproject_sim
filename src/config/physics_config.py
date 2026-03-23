# Lproject_sim/src/config/physics_config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RobotParameter:
    num_wheels: int = 4
    wheel_radius: float = 0.165  # Husky wheel radius
    wheel_width: float = 0.1
    wheel_base: float = 0.512    # Front-rear axle distance (0.256 * 2)
    wheel_track: float = 0.5708  # Left-right track width (0.2854 * 2)
    # Wheel center offsets from robot root frame (from USD inspection)
    wheel_x_offset: float = 0.256    # Front/rear distance from center
    wheel_y_offset: float = 0.2854   # Left/right distance from center
    wheel_center_z: float = 0.17775  # Wheel axle Z in local frame
    mass: float = 50.0
    wheel_stiffness: float = 100000.0 # Set to high stiffness (Solid/Rigid tire)
    wheel_damping: float = 10000.0
    max_effort: float = 100.0
    max_velocity: float = 10.0

@dataclass
class TerrainMechanicalParameter:
    k_c: float = 1400.0      # Cohesive modulus (Lunar Regolith: ~1400)
    k_phi: float = 820000.0  # Frictional modulus (Lunar Regolith: ~820k)
    n: float = 1.0           # Sinkage exponent
    c: float = 170.0         # Cohesion (Pascal) - Corrected for electrostatic properties
    phi: float = 0.5         # Friction angle (Radians)
    K: float = 0.015         # Shear deformation modulus (m)
    rho: float = 1600.0      # Density (kg/m^3)
    a_0: float = 0.4         # Tire-soil parameter
    a_1: float = 0.15        # Tire-soil parameter

@dataclass
class FootprintConf:
    width: float = 0.1
    height: float = 0.18

@dataclass
class DeformConstrainConf:
    horizontal_deform_offset: float = 0.0
    vertical_deform_offset: float = 0.0
    deform_decay_ratio: float = 0.5

@dataclass
class DepthDistributionConf:
    distribution: str = "sinusoidal"
    wave_frequency: float = 1.0

@dataclass
class BoundaryDistributionConf:
    distribution: str = "trapezoidal"
    angle_of_repose: float = 0.6

@dataclass
class ForceDepthRegressionConf:
    amplitude_slope: float = 0.01 # Increased 10x for visibility (Moon Gravity)
    amplitude_intercept: float = 0.0
    mean_slope: float = 0.005 # Updated to match amplitude scale (approx 0.5x amplitude)
    mean_intercept: float = 0.0

@dataclass
class DeformationEngineConf:
    terrain_resolution: float = 0.05
    terrain_width: float = 1.0
    terrain_height: float = 1.0
    num_links: int = 4
    footprint: FootprintConf = field(default_factory=FootprintConf)
    deform_constrain: DeformConstrainConf = field(default_factory=DeformConstrainConf)
    depth_distribution: DepthDistributionConf = field(default_factory=DepthDistributionConf)
    boundary_distribution: BoundaryDistributionConf = field(default_factory=BoundaryDistributionConf)
    force_depth_regression: ForceDepthRegressionConf = field(default_factory=ForceDepthRegressionConf)
    