# Lproject_sim/src/physics/terramechanics.py
import numpy as np
import scipy.integrate as integ
from src.config.physics_config import RobotParameter, TerrainMechanicalParameter

class TerramechanicsSolver:
    """
    Terramechanics solver class. Computes forces and torques acting on the robot wheels.
    Based on Wong's terramechanics models.
    """

    def __init__(self, robot_param: RobotParameter, terrain_param: TerrainMechanicalParameter):
        self.robot_param = robot_param
        self.terrain_param = terrain_param
        self.num_wheels = robot_param.num_wheels
        
        # Internal state variables
        self.slip_ratio = 0.0
        self.theta_f = 0.0
        self.theta_r = 0.0
        self.theta_m = 0.0
        self.sigma_max = 0.0

    def compute_slip_ratio(self, v: float, omega: float) -> float:
        """
        Compute the slip ratio.
        Args:
            v (float): The linear velocity of the wheel center.
            omega (float): The angular velocity of the wheel.
        """
        rw = self.robot_param.wheel_radius
        if abs(omega) < 1e-6:
             return 0.0 # Avoid division by zero
             
        v_theoretical = omega * rw
        
        if v_theoretical == 0:
            return 0.0

        if v <= v_theoretical:
            # Driving state
            # s = 1 - v / (r * w)
            self.slip_ratio = 1.0 - v / v_theoretical
        else:
            # Braking/Skidding state (simplified)
            # s = (r * w) / v - 1
            self.slip_ratio = v_theoretical / v - 1.0
            
        return self.slip_ratio

    def compute_angles(self, z: float) -> None:
        """
        Compute entry (theta_f), exit (theta_r), and max stress (theta_m) angles.
        Args:
            z (float): sinkage
        """
        # Entry angle
        # cos(theta_f) = 1 - z / r
        # theta_f = arccos(1 - z/r)
        # Note: OmniLRS used arctan(1 - z/r) which seems incorrect for the standard formula, 
        # but let's check if they meant geometric relation. 
        # Standard: z = r * (1 - cos(theta_f)) => cos(theta_f) = (r - z) / r = 1 - z/r
        # OmniLRS code: self.theta_f = np.arctan(1 - z / self.robot_param.wheel_radius) -> This looks like a bug or specific approximation in OmniLRS.
        # Let's use the standard geometric definition: arccos(1 - z/r)
        
        # Clamp z to avoid domain errors
        z = np.clip(z, 0, self.robot_param.wheel_radius)
        self.theta_f = np.arccos(1.0 - z / self.robot_param.wheel_radius)
        
        self.theta_r = 0.0 # Simplified: assume exit angle is 0 for now (or small)
        
        # Theta m: angle of maximum normal stress
        # theta_m = (a0 + a1 * s) * theta_f
        self.theta_m = (self.terrain_param.a_0 + self.terrain_param.a_1 * abs(self.slip_ratio)) * self.theta_f

    def _sigma(self, theta):
        """Normal stress distribution (Vectorized)"""
        # theta can be scalar or array
        
        # Front region: theta_m < theta < theta_f
        # Rear region: theta_r < theta < theta_m
        
        # z_theta calculation is common
        z_theta = self.robot_param.wheel_radius * (np.cos(theta) - np.cos(self.theta_f))
        
        # Pressure k = (kc / b + kphi)
        k = (self.terrain_param.k_c / self.robot_param.wheel_width) + self.terrain_param.k_phi
        
        sigma_val = k * (z_theta ** self.terrain_param.n)
        
        # Ensure non-negative (physically, z >= 0)
        # z_theta should be positive if theta < theta_f
        # but let's clip for safety if theta > theta_f due to numerical noise
        
        # In this simplified model, we use the same formula.
        # If we had different formulas for front/rear, we would use np.where(theta >= self.theta_m, front_calc, rear_calc)
        
        return sigma_val

    def _tau(self, theta):
        """Shear stress distribution (Vectorized)"""
        sigma = self._sigma(theta)
        
        # Shear displacement j
        j = self.robot_param.wheel_radius * (
            (self.theta_f - theta) - (1 - self.slip_ratio) * (np.sin(self.theta_f) - np.sin(theta))
        )
        
        shear_strength = self.terrain_param.c + sigma * np.tan(self.terrain_param.phi)
        return shear_strength * (1.0 - np.exp(-j / self.terrain_param.K))

    def compute_forces(self):
        """Compute Fx, Fz, My"""
        b = self.robot_param.wheel_width
        r = self.robot_param.wheel_radius
        
        # Integration limits
        # We integrate from theta_r to theta_f
        
        # Discretize theta for faster integration (Trapezoidal Rule)
        num_points = 20
        if abs(self.theta_f - self.theta_r) < 1e-4:
            return 0.0, 0.0, 0.0
            
        thetas = np.linspace(self.theta_r, self.theta_f, num_points)
        
        # Vectorized computation of stresses
        # Vectorized computation of stresses
        sigmas = self._sigma(thetas)
        taus = self._tau(thetas)
        
        # Fx = r * b * int(tau * cos(theta) - sigma * sin(theta))
        fx_integrand = taus * np.cos(thetas) - sigmas * np.sin(thetas)
        fx = r * b * np.trapz(fx_integrand, thetas)
        
        # Fz = r * b * int(tau * sin(theta) + sigma * cos(theta))
        fz_integrand = taus * np.sin(thetas) + sigmas * np.cos(thetas)
        fz = r * b * np.trapz(fz_integrand, thetas)
        
        # My = r^2 * b * int(tau)
        my_integrand = taus
        my = r * r * b * np.trapz(my_integrand, thetas)
        
        return fx, fz, my

    def solve(self, velocity: np.ndarray, omega: np.ndarray, sinkage: np.ndarray) -> tuple:
        """
        Solve for all wheels.
        Args:
            velocity (np.ndarray): Linear velocities (num_wheels,)
            omega (np.ndarray): Angular velocities (num_wheels,)
            sinkage (np.ndarray): Sinkages (num_wheels,)
        Returns:
            forces (np.ndarray): (num_wheels, 3) [Fx, Fy, Fz]
            torques (np.ndarray): (num_wheels, 3) [Mx, My, Mz]
        """
        forces = np.zeros((self.num_wheels, 3))
        torques = np.zeros((self.num_wheels, 3))
        
        for i in range(self.num_wheels):
            self.compute_slip_ratio(velocity[i], omega[i])
            self.compute_angles(sinkage[i])
            
            fx, fz, my = self.compute_forces()
            
            # Coordinate system:
            # Fx: Forward (Drawbar pull)
            # Fz: Vertical (Normal force)
            # My: Rolling resistance torque (opposing rotation)
            
            forces[i, 0] = fx
            forces[i, 2] = fz
            
            # Torque opposes rotation
            torques[i, 1] = -np.sign(omega[i]) * my
            
        return forces, torques