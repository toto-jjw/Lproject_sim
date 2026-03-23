import omni.ui as ui
import omni.kit.app
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class HUDData:
    """HUD에 표시할 데이터"""
    # 배터리
    battery_percent: float = 0.0
    battery_wh: float = 0.0
    battery_capacity: float = 100.0
    is_charging: bool = False
    
    # 태양광
    solar_power_w: float = 0.0
    solar_incidence_deg: float = 0.0
    panel_efficiency: float = 1.0
    sun_vector: tuple = (0.0, 0.0, 1.0)
    in_shadow: bool = False  # 그림자 영역 여부
    
    # 전력 소비
    power_consumption_w: float = 0.0
    net_power_w: float = 0.0
    
    # 온도
    temperature_c: float = 20.0
    
    # 로버 상태
    speed_ms: float = 0.0
    position: tuple = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    
    # 통신
    latency_ms: float = 0.0
    signal_strength: float = 1.0
    
    # 물리
    physics_info: str = ""
    
    # 시뮬레이션
    sim_time: float = 0.0
    step_count: int = 0
    
    # Stellar Engine
    stellar_time: str = ""
    sun_altitude_deg: float = 0.0
    sun_azimuth_deg: float = 0.0
    sun_visible: bool = True


class HUD:
    """
    향상된 Heads-Up Display for Lproject_sim.
    
    Features:
    - 배터리 상태 바 (색상 변화)
    - 태양광 발전 정보
    - 전력 수지 그래프
    - 온도 게이지
    - 로버 위치/속도
    - 통신 지연 표시
    """
    
    # 색상 상수 (ARGB)
    COLOR_GREEN = 0xFF00FF00
    COLOR_YELLOW = 0xFF00FFFF
    COLOR_ORANGE = 0xFF00A5FF
    COLOR_RED = 0xFF0000FF
    COLOR_BLUE = 0xFFFF8000
    COLOR_WHITE = 0xFFFFFFFF
    COLOR_GRAY = 0xFF888888
    COLOR_DARK = 0xFF333333
    
    def __init__(self, width: int = 350, height: int = 500):
        self._window = ui.Window(
            "Lproject_sim HUD", 
            width=width, 
            height=height, 
            dockPreference=ui.DockPreference.RIGHT_TOP
        )
        self._window.visible = True
        
        # UI 요소 참조
        self._labels: Dict[str, ui.Label] = {}
        self._bars: Dict[str, ui.Rectangle] = {}
        self._bar_containers: Dict[str, ui.Rectangle] = {}
        
        # 데이터
        self._data = HUDData()
        
        self._build_ui()
        
    def _build_ui(self):
        """UI 구성"""
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(height=0, spacing=8, style={"margin": 10}):
                    # 타이틀
                    ui.Label(
                        "[Lunar Rover Status]", 
                        height=25,
                        style={"font_size": 20, "color": self.COLOR_WHITE}
                    )
                    ui.Spacer(height=5)
                    
                    # === 배터리 섹션 ===
                    self._build_battery_section()
                    
                    ui.Spacer(height=10)
                    
                    # === 태양광 섹션 ===
                    self._build_solar_section()
                    
                    ui.Spacer(height=10)
                    
                    # === 온도 섹션 ===
                    self._build_temperature_section()
                    
                    ui.Spacer(height=10)
                    
                    # === 로버 상태 섹션 ===
                    self._build_rover_section()
                    
                    ui.Spacer(height=10)
                    
                    # === 통신 섹션 ===
                    self._build_comm_section()
                    
                    ui.Spacer(height=10)
                    
                    # === 물리/시뮬레이션 섹션 ===
                    self._build_physics_section()
                    
                    ui.Spacer(height=10)
                    
                    # 컨트롤 안내
                    ui.Label(
                        "Controls: teleop_twist_keyboard | P (Save DEM) | R (Reset Rover)", 
                        style={"font_size": 11, "color": self.COLOR_GRAY}
                    )
                    
    def _build_battery_section(self):
        """배터리 섹션"""
        with ui.CollapsableFrame("[Battery]", collapsed=False):
            with ui.VStack(spacing=5, style={"margin": 5}):
                # 배터리 바
                with ui.ZStack(height=25):
                    self._bar_containers["battery"] = ui.Rectangle(
                        style={"background_color": self.COLOR_DARK, "border_radius": 5}
                    )
                    with ui.HStack():
                        self._bars["battery"] = ui.Rectangle(
                            width=ui.Percent(50),
                            style={"background_color": self.COLOR_GREEN, "border_radius": 5}
                        )
                        ui.Spacer()
                    with ui.HStack():
                        ui.Spacer()
                        self._labels["battery_pct"] = ui.Label(
                            "50.0%", 
                            style={"font_size": 14, "color": self.COLOR_WHITE},
                            alignment=ui.Alignment.CENTER
                        )
                        ui.Spacer()
                        
                # 상세 정보
                with ui.HStack(height=18):
                    ui.Label("Charge:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["battery_wh"] = ui.Label("50.0 / 100.0 Wh")
                    
                with ui.HStack(height=18):
                    ui.Label("Status:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["battery_status"] = ui.Label("Discharging", style={"color": self.COLOR_YELLOW})
                    
    def _build_solar_section(self):
        """태양광 섹션"""
        with ui.CollapsableFrame("[Solar Power]", collapsed=False):
            with ui.VStack(spacing=5, style={"margin": 5}):
                # 태양광 발전 바
                with ui.ZStack(height=20):
                    self._bar_containers["solar"] = ui.Rectangle(
                        style={"background_color": self.COLOR_DARK, "border_radius": 3}
                    )
                    with ui.HStack():
                        self._bars["solar"] = ui.Rectangle(
                            width=ui.Percent(0),
                            style={"background_color": self.COLOR_YELLOW, "border_radius": 3}
                        )
                        ui.Spacer()
                        
                with ui.HStack(height=18):
                    ui.Label("Power:", width=100, style={"color": self.COLOR_GRAY})
                    self._labels["solar_power"] = ui.Label("0.0 W")
                    
                with ui.HStack(height=18):
                    ui.Label("Incidence:", width=100, style={"color": self.COLOR_GRAY})
                    self._labels["solar_angle"] = ui.Label("0.0 deg")
                    
                with ui.HStack(height=18):
                    ui.Label("Efficiency:", width=100, style={"color": self.COLOR_GRAY})
                    self._labels["panel_eff"] = ui.Label("100%")
                    
                with ui.HStack(height=18):
                    ui.Label("Sun Vec:", width=100, style={"color": self.COLOR_GRAY})
                    self._labels["sun_vec"] = ui.Label("[0.0, 0.0, 1.0]")

                with ui.HStack(height=18):
                    ui.Label("Shadow:", width=100, style={"color": self.COLOR_GRAY})
                    self._labels["shadow_status"] = ui.Label("[Sunlight]", style={"color": self.COLOR_YELLOW})
                    
                # 전력 수지
                ui.Spacer(height=5)
                with ui.HStack(height=20):
                    ui.Label("Net Power:", width=100, style={"color": self.COLOR_WHITE, "font_size": 13})
                    self._labels["net_power"] = ui.Label(
                        "+0.0 W", 
                        style={"color": self.COLOR_GREEN, "font_size": 13}
                    )
                    
    def _build_temperature_section(self):
        """온도 섹션"""
        with ui.CollapsableFrame("[Temperature]", collapsed=True):
            with ui.VStack(spacing=5, style={"margin": 5}):
                # 온도 바 (-150 ~ 150 C range)
                with ui.ZStack(height=20):
                    # 배경 (그라데이션 효과용)
                    with ui.HStack():
                        ui.Rectangle(width=ui.Percent(33), style={"background_color": 0xFFFF0000})  # Blue (cold)
                        ui.Rectangle(width=ui.Percent(34), style={"background_color": self.COLOR_GREEN})  # Green (optimal)
                        ui.Rectangle(width=ui.Percent(33), style={"background_color": 0xFF0000FF})  # Red (hot)
                    # 마커
                    with ui.HStack():
                        self._bars["temp_marker"] = ui.Spacer(width=ui.Percent(50))
                        ui.Rectangle(width=4, style={"background_color": self.COLOR_WHITE})
                        ui.Spacer()
                        
                with ui.HStack(height=18):
                    ui.Label("Current:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["temperature"] = ui.Label("20.0 C")
                    
    def _build_rover_section(self):
        """로버 상태 섹션"""
        with ui.CollapsableFrame("[Rover State]", collapsed=False):
            with ui.VStack(spacing=5, style={"margin": 5}):
                with ui.HStack(height=18):
                    ui.Label("Speed:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["speed"] = ui.Label("0.00 m/s")
                    
                with ui.HStack(height=18):
                    ui.Label("Position:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["position"] = ui.Label("[0.0, 0.0, 0.0]")
                    
                with ui.HStack(height=18):
                    ui.Label("Yaw:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["yaw"] = ui.Label("0.0 deg")

                with ui.HStack(height=18):
                    ui.Label("Roll:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["roll"] = ui.Label("0.0 deg")

                with ui.HStack(height=18):
                    ui.Label("Pitch:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["pitch"] = ui.Label("0.0 deg")
                    
    def _build_comm_section(self):
        """통신 섹션"""
        with ui.CollapsableFrame("[Communication]", collapsed=True):
            with ui.VStack(spacing=5, style={"margin": 5}):
                with ui.HStack(height=18):
                    ui.Label("Latency:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["latency"] = ui.Label("0 ms")
                    
                with ui.HStack(height=18):
                    ui.Label("Signal:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["signal"] = ui.Label("100%", style={"color": self.COLOR_GREEN})
                    
    def _build_physics_section(self):
        """물리/시뮬레이션 섹션"""
        with ui.CollapsableFrame("[Simulation]", collapsed=True):
            with ui.VStack(spacing=5, style={"margin": 5}):
                with ui.HStack(height=18):
                    ui.Label("Physics:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["physics"] = ui.Label("Init...")
                    
                with ui.HStack(height=18):
                    ui.Label("Sim Time:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["sim_time"] = ui.Label("0.0 s")
                    
                with ui.HStack(height=18):
                    ui.Label("Steps:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["steps"] = ui.Label("0")
                    
                # Stellar Engine Info
                ui.Spacer(height=5)
                ui.Label("--- Stellar Engine ---", style={"color": self.COLOR_GRAY, "font_size": 11})
                
                with ui.HStack(height=18):
                    ui.Label("Lunar Time:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["stellar_time"] = ui.Label("--")
                    
                with ui.HStack(height=18):
                    ui.Label("Sun Alt:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["sun_alt"] = ui.Label("0.0 deg")
                    
                with ui.HStack(height=18):
                    ui.Label("Sun Az:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["sun_az"] = ui.Label("0.0 deg")
                    
                with ui.HStack(height=18):
                    ui.Label("Daylight:", width=80, style={"color": self.COLOR_GRAY})
                    self._labels["sun_visible"] = ui.Label("Yes ☀️", style={"color": self.COLOR_YELLOW})
                    
    def _get_battery_color(self, percent: float) -> int:
        """배터리 잔량에 따른 색상"""
        if percent > 60:
            return self.COLOR_GREEN
        elif percent > 30:
            return self.COLOR_YELLOW
        elif percent > 15:
            return self.COLOR_ORANGE
        else:
            return self.COLOR_RED
            
    def _get_temp_color(self, temp_c: float) -> int:
        """온도에 따른 색상"""
        if -20 <= temp_c <= 40:
            return self.COLOR_GREEN
        elif -50 <= temp_c <= 70:
            return self.COLOR_YELLOW
        else:
            return self.COLOR_RED
            
    def update(self, battery_percent: float, speed: float, sun_vector, physics_info: str = ""):
        """
        기본 업데이트 (하위 호환성)
        
        Args:
            battery_percent: 배터리 퍼센트
            speed: 속도 (m/s)
            sun_vector: 태양 방향 벡터
            physics_info: 물리 정보 문자열
        """
        self._data.battery_percent = battery_percent
        self._data.speed_ms = speed
        if sun_vector is not None:
            self._data.sun_vector = tuple(sun_vector) if hasattr(sun_vector, '__iter__') else (0,0,1)
        self._data.physics_info = physics_info
        
        self._update_ui()
        
    def update_full(self, data: HUDData):
        """
        전체 데이터로 업데이트
        
        Args:
            data: HUDData 객체
        """
        self._data = data
        self._update_ui()
        
    def update_from_dict(self, data: Dict[str, Any]):
        """
        딕셔너리로 업데이트
        
        Args:
            data: 업데이트할 데이터 딕셔너리
        """
        for key, value in data.items():
            if hasattr(self._data, key):
                setattr(self._data, key, value)
        self._update_ui()
        
    def _update_ui(self):
        """UI 요소 업데이트"""
        d = self._data
        
        try:
            # === 배터리 ===
            if "battery_pct" in self._labels:
                self._labels["battery_pct"].text = f"{d.battery_percent:.1f}%"
                
            if "battery" in self._bars:
                self._bars["battery"].width = ui.Percent(max(0, min(100, d.battery_percent)))
                color = self._get_battery_color(d.battery_percent)
                self._bars["battery"].set_style({"background_color": color, "border_radius": 5})
                
            if "battery_wh" in self._labels:
                self._labels["battery_wh"].text = f"{d.battery_wh:.1f} / {d.battery_capacity:.1f} Wh"
                
            if "battery_status" in self._labels:
                if d.is_charging:
                    self._labels["battery_status"].text = ">> Charging"
                    self._labels["battery_status"].set_style({"color": self.COLOR_GREEN})
                else:
                    self._labels["battery_status"].text = "<< Discharging"
                    self._labels["battery_status"].set_style({"color": self.COLOR_YELLOW})
                    
            # === 태양광 ===
            if "solar_power" in self._labels:
                self._labels["solar_power"].text = f"{d.solar_power_w:.1f} W"
                
            if "solar" in self._bars:
                # 최대 68W 기준 (1361 W/m² * 0.25m² * 0.2 효율)
                max_solar = 68.0
                pct = min(100, (d.solar_power_w / max_solar) * 100)
                self._bars["solar"].width = ui.Percent(pct)
                
            if "solar_angle" in self._labels:
                self._labels["solar_angle"].text = f"{d.solar_incidence_deg:.1f} deg"
                
            if "panel_eff" in self._labels:
                self._labels["panel_eff"].text = f"{d.panel_efficiency * 100:.0f}%"
                
            if "sun_vec" in self._labels:
                sv = d.sun_vector
                self._labels["sun_vec"].text = f"[{sv[0]:.2f}, {sv[1]:.2f}, {sv[2]:.2f}]"

            if "shadow_status" in self._labels:
                if d.in_shadow:
                    self._labels["shadow_status"].text = "SHADOW"
                    self._labels["shadow_status"].set_style({"color": self.COLOR_RED})
                else:
                    self._labels["shadow_status"].text = "Sunlight"
                    self._labels["shadow_status"].set_style({"color": self.COLOR_YELLOW})
                
            if "net_power" in self._labels:
                net = d.net_power_w
                if net >= 0:
                    self._labels["net_power"].text = f"+{net:.1f} W"
                    self._labels["net_power"].set_style({"color": self.COLOR_GREEN, "font_size": 13})
                else:
                    self._labels["net_power"].text = f"{net:.1f} W"
                    self._labels["net_power"].set_style({"color": self.COLOR_RED, "font_size": 13})
                    
            # === 온도 ===
            if "temperature" in self._labels:
                color = self._get_temp_color(d.temperature_c)
                self._labels["temperature"].text = f"{d.temperature_c:.1f} C"
                self._labels["temperature"].set_style({"color": color})
                
            if "temp_marker" in self._bars:
                # -150 ~ 150°C 범위를 0~100%로 매핑
                pct = ((d.temperature_c + 150) / 300) * 100
                pct = max(0, min(100, pct))
                self._bars["temp_marker"].width = ui.Percent(pct)
                
            # === 로버 ===
            if "speed" in self._labels:
                self._labels["speed"].text = f"{d.speed_ms:.2f} m/s"
                
            if "position" in self._labels:
                pos = d.position
                self._labels["position"].text = f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
                
            if "yaw" in self._labels:
                self._labels["yaw"].text = f"{d.yaw_deg:.1f} deg"
                
            if "roll" in self._labels:
                self._labels["roll"].text = f"{d.roll_deg:.1f} deg"

            if "pitch" in self._labels:
                self._labels["pitch"].text = f"{d.pitch_deg:.1f} deg"

            # === 통신 ===
            if "latency" in self._labels:
                self._labels["latency"].text = f"{d.latency_ms:.0f} ms"
                
            if "signal" in self._labels:
                sig = d.signal_strength * 100
                self._labels["signal"].text = f"{sig:.0f}%"
                if sig > 70:
                    self._labels["signal"].set_style({"color": self.COLOR_GREEN})
                elif sig > 30:
                    self._labels["signal"].set_style({"color": self.COLOR_YELLOW})
                else:
                    self._labels["signal"].set_style({"color": self.COLOR_RED})
                    
            # === 물리 ===
            if "physics" in self._labels:
                self._labels["physics"].text = d.physics_info or "N/A"
                
            if "sim_time" in self._labels:
                self._labels["sim_time"].text = f"{d.sim_time:.1f} s"
                
            if "steps" in self._labels:
                self._labels["steps"].text = str(d.step_count)
            
            # === Stellar Engine ===
            stellar_time = getattr(d, 'stellar_time', "")
            sun_alt = getattr(d, 'sun_altitude_deg', 0.0)
            sun_az = getattr(d, 'sun_azimuth_deg', 0.0)
            sun_visible = getattr(d, 'sun_visible', True)
            
            if "stellar_time" in self._labels:
                if stellar_time:
                    # "2024-05-01 12:00:00 UTC" -> 간략화
                    try:
                        short_time = stellar_time[:19]  # Remove " UTC"
                    except:
                        short_time = stellar_time
                    self._labels["stellar_time"].text = short_time
                else:
                    self._labels["stellar_time"].text = "-- (disabled)"
                    
            if "sun_alt" in self._labels:
                self._labels["sun_alt"].text = f"{sun_alt:.1f} deg"
                
            if "sun_az" in self._labels:
                self._labels["sun_az"].text = f"{sun_az:.1f} deg"
                
            if "sun_visible" in self._labels:
                if sun_visible:
                    self._labels["sun_visible"].text = "Yes"
                    self._labels["sun_visible"].set_style({"color": self.COLOR_YELLOW})
                else:
                    self._labels["sun_visible"].text = "No"
                    self._labels["sun_visible"].set_style({"color": self.COLOR_GRAY})
                
        except Exception as e:
            print(f"[HUD] Error updating UI: {e}")
            
    def destroy(self):
        """리소스 정리"""
        self._window.visible = False
        self._window = None
        self._labels.clear()
        self._bars.clear()
