# Lproject_sim/src/core/robot_context.py
import numpy as np
from src.robots.rover import Rover
from src.core.ros_manager import ROSManager
from src.sensors.sun_sensor import SunSensor
from src.robots.solar_panel import SolarPanel
from src.core.energy_manager import EnergyManager
from src.core.latency_manager import LatencyManager
from src.core.thermal_manager import ThermalModel
from src.config.physics_config import RobotParameter

class RobotContext:
    """
    Encapsulates a single robot and its associated sensors, components, and managers.
    Updated for Native ROS 2 (Lproject_ros).
    """
    def __init__(self, config: dict, env_config: dict, terrain_manager):
        self.config = config
        self.env_config = env_config
        self.tm = terrain_manager
        self.name = config["name"]
        
        self.robot_param = RobotParameter()
        
        # [Fix] Load physics parameters from config if available
        if "physics_parameters" in config:
            physics_params = config["physics_parameters"]
            for key, value in physics_params.items():
                if hasattr(self.robot_param, key):
                    setattr(self.robot_param, key, value)
                    # print(f"[{self.name}] Set physics param {key}={value}")
        
        # 1. Initialize Rover
        # [Option] Terrain Height Snap - odom 프레임을 고정하려면 false로 설정
        init_pos = np.array(config["position"])
        terrain_snap = config.get("terrain_snap", True)  # 기본값: True
        
        if terrain_snap:
            try:
                terrain_z = self.tm.get_heights(init_pos[:2].reshape(1, 2))[0]
                init_pos[2] = max(init_pos[2], terrain_z + 0.45)
                print(f"[{self.name}] Terrain snap enabled. Adjusted Z: {init_pos[2]:.3f}")
            except:
                pass
        else:
            print(f"[{self.name}] Terrain snap disabled. Using config position Z: {init_pos[2]:.3f}")
        
        self.initial_position = init_pos.copy()
        self.initial_orientation = np.array(config["orientation"]).copy()
        self._custom_reset_position = None
        self._custom_reset_orientation = None
        
        self.rover = Rover(
            prim_path=f"/World/{self.name}",
            name=self.name,
            usd_path=config.get("usd_path"),
            position=init_pos,
            orientation=np.array(config["orientation"]),
            robot_param=self.robot_param
        )

        
        # 2. Initialize Sensors & Components
        self.sensors = {}
        self.components = {}
        self._init_sensors(config.get("sensors", {}))
        self._init_components(config.get("components", {}))
            
        # 3. Initialize Native ROS 2 Manager
        # Note: rclpy.init() must be called in main.py before this
        nav2_enabled = env_config.get("nav2", {}).get("enabled", False)
        
        # publish_map_tf logic: Prioritize explicit config, fallback to !nav2_enabled
        explicit_map_tf = env_config.get("publish_map_tf", None)
        if explicit_map_tf is not None:
            should_publish_tf = explicit_map_tf
        else:
            should_publish_tf = not nav2_enabled

        self.ros2 = ROSManager(robot_name=self.name.lower(), publish_map_tf=should_publish_tf)
        self.ros2.set_target_callback(self.set_custom_reset_target)
        
        # 4. Initialize Latency Manager (통신 지연 시뮬레이션)
        self.latency_manager = None
        latency_cfg = env_config.get("latency", {})
        if latency_cfg.get("enabled", False):
            self.latency_manager = LatencyManager(
                delay_seconds=latency_cfg.get("delay_seconds", 1.3),
                dropout_rate=latency_cfg.get("dropout_rate", 0.05)
            )
            print(f"[{self.name}] LatencyManager enabled (delay={latency_cfg.get('delay_seconds', 1.3)}s)")
        
        # State
        self.nav_active = False
        
        
    def _init_sensors(self, s_cfg):
        prim_root = self.config["prim_path"]
        
        # Global Camera Effects Config
        effects_config = self.env_config.get("camera_effects", None)
        
        # [Hybrid Mode Check]
        # If 'hybrid', we assume the USD already has Sensor Graphs (Lidar/Camera).
        # If 'omnigraph' or 'native', we create them via Python Managers.
        bridge_type = self.env_config.get("ros_bridge_type", "omnigraph")
        
            
        if s_cfg.get("sun_sensor", {}).get("enabled", False):
            try:
                cfg = s_cfg["sun_sensor"]
                self.sensors["sun_sensor"] = SunSensor(
                    prim_path=f"{prim_root}/base_link/SunSensor",
                    name=f"{self.name.lower()}_sunsensor",
                    position=np.array(cfg.get("position")),
                    orientation=np.array(cfg.get("orientation"))
                )
                print(f"[{self.name}] DEBUG: SunSensor successfully created and added to self.sensors.")

            except Exception as e: print(f"Error init SunSensor: {e}")

    def _init_components(self, c_cfg):
        prim_root = self.config["prim_path"]
        
        if c_cfg.get("solar_panel", {}).get("enabled", False):
            try:
                cfg = c_cfg["solar_panel"]
                self.components["solar_panel"] = SolarPanel(
                    parent_path=f"{prim_root}/base_link",
                    name=f"{self.name.lower()}_solarpanel",
                    position=np.array(cfg.get("position"))
                )
            except Exception as e: print(f"Error init SolarPanel: {e}")
            
        if c_cfg.get("energy_manager", {}).get("enabled", False):
            try:
                self.components["energy_manager"] = EnergyManager()
            except Exception as e: print(f"Error init EnergyManager: {e}")
        
        # ThermalModel 초기화 (온도 계산 담당)
        # 달 표면 온도: 낮 +127°C, 밤 -173°C
        thermal_cfg = self.env_config.get("thermal_model", {})
        thermal_enabled = thermal_cfg.get("enabled", True)
        self.thermal_model = ThermalModel(
            enabled=thermal_enabled,
            static_temperature=thermal_cfg.get("static_temperature", 20.0),
            initial_temp=thermal_cfg.get("static_temperature", 20.0),  # Use static_temp as initial
            min_temp=thermal_cfg.get("min_temp", -173.0),
            max_temp=thermal_cfg.get("max_temp", 127.0),
            time_constant=thermal_cfg.get("time_constant", 600.0),
            measurement_noise_std=thermal_cfg.get("measurement_noise_std", 0.5)
        )
        if not thermal_enabled:
            print(f"[{self.name}] ThermalModel disabled. Using static temperature: {self.thermal_model.static_temperature}°C")
            
        # Navigator는 현재 Nav2로 대체되어 사용하지 않음
        # if c_cfg.get("navigator", {}).get("enabled", False):
        #     self.components["navigator"] = Navigator(self.tm)

    def initialize(self):
        self.rover.initialize()
        print(f"[{self.name}] Initialized Sensors: {list(self.sensors.keys())}")
        print(f"[{self.name}] Initialized Components: {list(self.components.keys())}")
    
    def reset_pose(self, position=None, orientation=None):
        """
        로버를 지정된 위치/방향으로 이동하고 모든 속도를 완전히 0으로 초기화합니다.
        
        Args:
            position: 리셋할 위치 [x, y, z]. None이면 초기 생성 위치 또는 커스텀 타겟 사용.
            orientation: 리셋할 방향 [w, x, y, z]. None이면 초기 방향 또는 커스텀 타겟 사용.
        """
        # 우선순위: 직접 전달 > 커스텀 타겟 > 초기 위치
        if position is not None:
            target_pos = np.array(position, dtype=np.float64)
        elif self._custom_reset_position is not None:
            target_pos = self._custom_reset_position.copy()
        else:
            target_pos = self.initial_position.copy()
        
        if orientation is not None:
            target_ori = np.array(orientation, dtype=np.float64)
        elif self._custom_reset_orientation is not None:
            target_ori = self._custom_reset_orientation.copy()
        else:
            target_ori = self.initial_orientation.copy()
        
        try:
            num_dofs = len(self.rover.dof_names) if self.rover.dof_names else 0
            
            # 1. 관절 토크(effort) 제거 — 구동력 즉시 차단
            if num_dofs > 0:
                self.rover.set_joint_efforts(np.zeros(num_dofs))
            
            # 2. 관절 속도를 0으로 설정
            if num_dofs > 0:
                self.rover.set_joint_velocities(np.zeros(num_dofs))
            
            # 3. ArticulationAction으로 컨트롤러 목표 속도를 0으로 오버라이드
            #    (OmniGraph velocity controller가 이전 cmd_vel을 재적용하는 것 방지)
            if num_dofs > 0:
                try:
                    from isaacsim.core.utils.types import ArticulationAction
                    zero_action = ArticulationAction(
                        joint_velocities=np.zeros(num_dofs),
                        joint_efforts=np.zeros(num_dofs)
                    )
                    self.rover.apply_action(zero_action)
                except ImportError:
                    try:
                        from omni.isaac.core.utils.types import ArticulationAction
                        zero_action = ArticulationAction(
                            joint_velocities=np.zeros(num_dofs),
                            joint_efforts=np.zeros(num_dofs)
                        )
                        self.rover.apply_action(zero_action)
                    except ImportError:
                        pass  # ArticulationAction not available
            
            # 4. 월드 포즈 설정 (위치 + 방향)
            self.rover.set_world_pose(
                position=target_pos,
                orientation=target_ori
            )
            
            # 5. 강체(RigidBody) 선속도 / 각속도 0으로 설정
            self.rover.set_linear_velocity(np.zeros(3))
            self.rover.set_angular_velocity(np.zeros(3))
            
            print(f"[{self.name}] Reset pose: pos={target_pos}, ori={target_ori}")
            return True
        except Exception as e:
            print(f"[{self.name}] Error resetting pose: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def reset_to_initial_pose(self):
        """하위 호환용 — 초기 위치로 리셋 (커스텀 타겟 무시)"""
        return self.reset_pose(self.initial_position.copy(), self.initial_orientation.copy())
    
    def set_custom_reset_target(self, position, orientation=None):
        """
        커스텀 리셋 목표 위치를 설정합니다.
        이후 reset_pose()를 인자 없이 호출하면 이 위치로 리셋됩니다.
        
        Args:
            position: [x, y, z] 목표 위치
            orientation: [w, x, y, z] 목표 방향 (None이면 초기 방향 사용)
        """
        self._custom_reset_position = np.array(position, dtype=np.float64)
        if orientation is not None:
            self._custom_reset_orientation = np.array(orientation, dtype=np.float64)
        else:
            self._custom_reset_orientation = None
        print(f"[{self.name}] Custom reset target set: pos={self._custom_reset_position}")
    
    def clear_custom_reset_target(self):
        """커스텀 리셋 목표를 해제합니다. 이후 reset_pose()는 초기 위치로 리셋됩니다."""
        self._custom_reset_position = None
        self._custom_reset_orientation = None
        print(f"[{self.name}] Custom reset target cleared. Will reset to initial position.")

    def update(self, dt, step_count):
        """
        Main update loop for the robot.
        
        Note: Robot control (cmd_vel) is handled by ROS teleop_twist_keyboard
              or Nav2, not by internal keyboard input.
        """
        # Spin ROS 2 Node to process callbacks
        self.ros2.spin_once()
                
        # 1. Energy & Solar
        if "energy_manager" in self.components:
            try:
                sun_vec = np.array([0.0, 0.0, 1.0])  # Default sun direction
                panel_norm = np.array([0.0, 0.0, 1.0])  # Default panel normal
                vel = 0.0
                try:
                    vel = np.linalg.norm(self.rover.get_linear_velocity())
                except:
                    pass
                dust_eff = 1.0
                shadow_factor = 1.0  # 1.0 = sunlight, 0.0 = shadow
                
                # Get sun vector from sensor if available
                if "sun_sensor" in self.sensors:
                    try:
                        # For energy calculation, use world frame sun direction
                        sun_vec = self.sensors["sun_sensor"].get_sun_direction_world()
                    except:
                        pass  # Use default sun_vec
                
                # Get panel info if solar panel exists
                panel_world_pos = None
                if "solar_panel" in self.components:
                    try:
                        panel = self.components["solar_panel"]
                        
                        # Apply ROS Command
                        if self.ros2:
                            cmd_angle = self.ros2.get_solar_cmd()
                            panel.set_angle(cmd_angle)
                        
                        panel_norm = panel.get_world_normal()
                        dust_eff = panel.get_efficiency_factor()
                        
                        # Get panel world position for shadow check
                        if hasattr(panel, 'get_world_position'):
                            panel_world_pos = panel.get_world_position()
                    except:
                        pass  # Use default panel_norm and dust_eff
                
                # Shadow detection using raycast
                in_shadow = False
                if "sun_sensor" in self.sensors:
                    try:
                        sun_sensor = self.sensors["sun_sensor"]
                        sun_sensor.check_shadow(panel_world_pos)
                        shadow_factor = sun_sensor.get_shadow_factor()
                        in_shadow = sun_sensor.is_in_shadow()
                    except:
                        shadow_factor = 1.0  # Assume sunlight on error
                        in_shadow = False
                
                # Apply shadow factor to dust efficiency (combined efficiency)
                combined_efficiency = dust_eff * shadow_factor
                
                # ThermalModel 업데이트 - 1초에 한 번만 계산
                interior_temp = None  # None이면 energy_manager 기존 온도 유지
                if self.thermal_model is not None:
                    # 시간 누적
                    if not hasattr(self, '_thermal_update_acc'):
                        self._thermal_update_acc = 0.0
                    self._thermal_update_acc += dt
                    
                    # 1초마다 온도 계산
                    if self._thermal_update_acc >= 1.0:
                        try:
                            # 로버 위치 및 방향 가져오기
                            rover_pos, rover_ori = self.rover.get_world_pose()
                            rover_yaw = 0.0
                            if rover_ori is not None:
                                # quaternion (w,x,y,z) to yaw
                                import math
                                w, x, y, z = rover_ori
                                rover_yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                            
                            # ThermalModel 상태 업데이트
                            self.thermal_model.set_in_shadow(in_shadow)
                            self.thermal_model.step(
                                dt=self._thermal_update_acc,
                                sun_direction=sun_vec,
                                rover_position=rover_pos,
                                rover_yaw=rover_yaw
                            )
                            self._thermal_update_acc = 0.0
                            
                            # 1초 업데이트 시점에만 온도 가져오기 (노이즈 없이)
                            interior_temp = self.thermal_model.node_temps.get("interior", 20.0)
                        except Exception as e:
                            import traceback
                            print(f"[{self.name}] ThermalModel update error: {e}")
                            traceback.print_exc()
                
                # Energy manager 업데이트 (온도는 1초에 한 번만 갱신)
                self.components["energy_manager"].update(
                    dt, sun_vec, panel_norm, vel, 
                    dust_efficiency=combined_efficiency,
                    temperature=interior_temp  # None이면 기존 온도 유지
                )
            except Exception as e:
                import traceback
                print(f"[{self.name}] Error updating energy_manager: {e}")
                traceback.print_exc()
            
        # 3. Publish Data
        self._publish_data(dt)

    def _publish_data(self, dt):
        """
        Publish all sensor data to ROS.
        """
        if not self.ros2: return
        
        # map → base_footprint TF (절대 좌표)
        try:
            pos, ori = self.rover.get_world_pose()
            # ori format from Isaac is (w, x, y, z)
            if pos is not None and ori is not None:
                self.ros2.publish_map_to_base_tf(pos, ori)
        except Exception as e:
            pass  # Silently ignore if pose not available

        # Sun Sensor (Custom)
        if "sun_sensor" in self.sensors:
            try:
                vec = self.sensors["sun_sensor"].get_sun_vector()
                self.ros2.publish_sun_vector(vec)
            except Exception as e:
                print(f"Error publishing sun vector: {e}")
            
        # Battery Data (Logic-based)
        try:
            if "energy_manager" in self.components:
                em = self.components["energy_manager"]
                pct = em.current_charge_wh / em.capacity_wh
                self.ros2.publish_battery_state(24.0, pct, 0.0)
        except Exception as e: 
            print(f"Error publishing battery state: {e}")
        
        # Sensor Temperature (for camera noise dark current)
        try:
            if "energy_manager" in self.components:
                status = self.components["energy_manager"].get_status()
                temp_c = status.get("temperature", 20.0)
                self.ros2.publish_sensor_temperature(temp_c)
        except Exception as e:
            pass  # Silently ignore if temperature not available

        # Note: Wheel Encoders (Joint States) are published by USD OmniGraph
        # No need to publish from Python code

    def shutdown(self):
        print(f"[{self.name}] Shutting down custom sensors and ROS...")
        for name, sensor in self.sensors.items():
            # 이제 커스텀 센서만 남아있으므로 로직이 간단해집니다.
            if hasattr(sensor, 'cleanup'):
                sensor.cleanup()
            
        self.ros2.shutdown()
