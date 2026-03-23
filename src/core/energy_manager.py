# Lproject_sim/src/core/energy_manager.py
"""
달 탐사 로버의 에너지 관리 시스템

주요 기능:
1. 배터리 상태 관리 (충전/방전)
2. 태양광 발전 계산 (입사각 기반)
3. 전력 소비 모델 (기본 부하 + 모터 부하)
4. 온도 기반 배터리 효율 계산

온도 모델링:
- 온도 계산은 ThermalModel에서 담당 (src/core/thermal_manager.py)
- EnergyManager는 외부에서 온도를 주입받아 배터리 효율 계산에 사용
- set_temperature() 또는 update()의 temperature 파라미터로 온도 전달

물리 모델:
- 태양광 발전: P_solar = I * A * η * cos(θ) * dust_factor
- 전력 소비: P_load = P_base + k * |v|
- 배터리 효율: η_batt = max(0.1, 1 - 0.01 * |T - T_opt|)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EnergyConfig:
    """에너지 시스템 설정"""
    # 배터리
    capacity_wh: float = 100.0          # 배터리 용량 (Wh)
    initial_charge_wh: float = 50.0     # 초기 충전량 (Wh)
    
    # 태양광 패널
    solar_efficiency: float = 0.20      # 태양전지 효율 (20%)
    solar_area: float = 0.25            # 패널 면적 (m^2) - 0.5m x 0.5m
    solar_intensity: float = 1361.0     # 태양 상수 (W/m^2)
    
    # 전력 소비
    base_load_w: float = 10.0           # 기본 소비 전력 (W) - 컴퓨터, 통신 등
    motor_load_factor: float = 5.0      # 모터 부하 계수 (W per m/s)
    
    # 온도 관련 (배터리 효율 계산용)
    initial_temp: float = 20.0          # 초기 온도 (°C)
    optimal_temp: float = 20.0          # 최적 작동 온도 (°C)


class EnergyManager:
    """
    로버 에너지 관리 시스템
    
    배터리 충/방전, 태양광 발전, 온도 관리를 담당합니다.
    """
    
    def __init__(self, config: Optional[EnergyConfig] = None, 
                 capacity_wh: float = 100.0, initial_charge_wh: float = 50.0):
        """
        Args:
            config: 에너지 시스템 설정. None이면 기본값 사용.
            capacity_wh: 배터리 용량 (config이 None일 때 사용)
            initial_charge_wh: 초기 충전량 (config이 None일 때 사용)
        """
        if config:
            self.cfg = config
        else:
            self.cfg = EnergyConfig(capacity_wh=capacity_wh, initial_charge_wh=initial_charge_wh)
        
        # 상태 변수
        self.capacity_wh = self.cfg.capacity_wh  # 하위 호환성
        self.current_charge_wh = self.cfg.initial_charge_wh
        self.temperature = self.cfg.initial_temp
        
        # 누적 통계 (디버깅/분석용)
        self._total_solar_generated_wh = 0.0
        self._total_consumed_wh = 0.0
        self._update_count = 0
        
        # 최근 업데이트 결과 캐시
        self._last_status: Dict = {}
        
    def update(self, dt: float, sun_vector: np.ndarray, panel_normal: np.ndarray, 
               robot_velocity: float, dust_efficiency: float = 1.0, 
               temperature: Optional[float] = None) -> Dict:
        """
        에너지 시스템 상태 업데이트
        
        Args:
            dt: 시간 간격 (초)
            sun_vector: 태양 방향 벡터 (센서 프레임, 정규화됨)
            panel_normal: 패널 법선 벡터 (월드 프레임, 정규화됨)
            robot_velocity: 로버 속도 (m/s)
            dust_efficiency: 먼지로 인한 효율 저하 (0~1, 1=깨끗함)
            temperature: 외부에서 주입받는 온도 (°C). None이면 기존 온도 유지.
                         ThermalModel에서 계산된 온도를 전달받습니다.
            
        Returns:
            상태 정보 딕셔너리
        """
        self._update_count += 1
        
        # 1. 태양광 발전량 계산
        solar_power_w = self._calculate_solar_power(sun_vector, panel_normal, dust_efficiency)
        
        # 2. 전력 소비량 계산
        load_w = self._calculate_load(robot_velocity)
        
        # 3. 온도 업데이트 (외부에서 주입받은 값 사용)
        # 온도 계산은 ThermalModel에서 담당, 여기서는 주입만 받음
        if temperature is not None:
            self.temperature = temperature
        
        sun_incidence = self._calculate_incidence(sun_vector, panel_normal)
        
        # 4. 배터리 효율 계산 (온도 기반)
        efficiency = self._calculate_battery_efficiency()
        
        # 5. 배터리 충전량 업데이트
        net_power_w = solar_power_w - load_w
        energy_delta_wh = self._calculate_energy_delta(net_power_w, dt, efficiency)
        
        self.current_charge_wh += energy_delta_wh
        self.current_charge_wh = np.clip(self.current_charge_wh, 0.0, self.capacity_wh)
        
        # 통계 업데이트
        if solar_power_w > 0:
            self._total_solar_generated_wh += solar_power_w * (dt / 3600.0)
        if load_w > 0:
            self._total_consumed_wh += load_w * (dt / 3600.0)
        
        # 결과 저장 및 반환
        self._last_status = {
            "solar_power_w": solar_power_w,
            "load_w": load_w,
            "net_power_w": net_power_w,
            "charge_wh": self.current_charge_wh,
            "percentage": self.get_percentage(),
            "temperature": self.temperature,
            "efficiency": efficiency,
            "sun_incidence": sun_incidence,
            "is_charging": net_power_w > 0
        }
        
        return self._last_status
    
    def _calculate_incidence(self, sun_vector: np.ndarray, panel_normal: np.ndarray) -> float:
        """태양광 입사각 계산 (코사인)"""
        sun_norm = np.linalg.norm(sun_vector)
        panel_norm = np.linalg.norm(panel_normal)
        
        if sun_norm < 1e-6 or panel_norm < 1e-6:
            return 0.0
            
        sun_dir = sun_vector / sun_norm
        panel_dir = panel_normal / panel_norm
        return max(0.0, np.dot(sun_dir, panel_dir))
    
    def _calculate_solar_power(self, sun_vector: np.ndarray, panel_normal: np.ndarray, 
                               dust_efficiency: float) -> float:
        """태양광 발전량 계산 (W)"""
        incidence = self._calculate_incidence(sun_vector, panel_normal)
        
        if incidence <= 0:
            return 0.0
            
        return (self.cfg.solar_intensity * 
                self.cfg.solar_area * 
                self.cfg.solar_efficiency * 
                incidence * 
                dust_efficiency)
    
    def _calculate_load(self, robot_velocity: float) -> float:
        """전력 소비량 계산 (W)"""
        return self.cfg.base_load_w + (abs(robot_velocity) * self.cfg.motor_load_factor)
    
    def set_temperature(self, temperature: float):
        """
        외부에서 온도 설정 (ThermalModel과 연동용)
        
        온도 계산은 ThermalModel에서 담당하고,
        계산된 온도를 이 메서드로 EnergyManager에 전달합니다.
        
        Args:
            temperature: 온도 (°C)
        """
        self.temperature = temperature
    
    def _calculate_battery_efficiency(self) -> float:
        """온도 기반 배터리 효율 계산 (0.1 ~ 1.0)"""
        temp_deviation = abs(self.temperature - self.cfg.optimal_temp)
        return max(0.1, 1.0 - 0.01 * temp_deviation)
    
    def _calculate_energy_delta(self, net_power_w: float, dt: float, 
                                efficiency: float) -> float:
        """에너지 변화량 계산 (Wh)"""
        energy_delta_wh = net_power_w * (dt / 3600.0)
        
        # 방전 시 효율 저하 적용 (저온/고온에서 더 빨리 방전)
        if energy_delta_wh < 0:
            energy_delta_wh /= efficiency
            
        return energy_delta_wh
    
    def get_status(self) -> Dict:
        """현재 상태 반환 (_last_status에는 solar_power_w, net_power_w 등 포함)"""
        if self._last_status:
            return self._last_status
        # update()가 아직 호출되지 않은 경우 기본값 반환
        return {
            "solar_power_w": 0.0,
            "load_w": 0.0,
            "net_power_w": 0.0,
            "charge_wh": self.current_charge_wh,
            "percentage": self.get_percentage(),
            "temperature": self.temperature,
            "efficiency": 1.0,
            "sun_incidence": 0.0,
            "is_charging": False
        }
    
    def get_percentage(self) -> float:
        """배터리 잔량 퍼센트 반환"""
        return (self.current_charge_wh / self.capacity_wh) * 100.0
    
    def get_statistics(self) -> Dict:
        """누적 통계 반환 (디버깅용)"""
        return {
            "total_solar_generated_wh": self._total_solar_generated_wh,
            "total_consumed_wh": self._total_consumed_wh,
            "update_count": self._update_count,
            "net_energy_wh": self._total_solar_generated_wh - self._total_consumed_wh
        }
    
    def set_charge(self, charge_wh: float):
        """배터리 충전량 직접 설정 (테스트용)"""
        self.current_charge_wh = np.clip(charge_wh, 0.0, self.capacity_wh)
    
    def is_critical(self, threshold: float = 10.0) -> bool:
        """배터리가 위험 수준인지 확인"""
        return self.get_percentage() < threshold
    
    def is_full(self) -> bool:
        """배터리가 완충인지 확인"""
        return self.current_charge_wh >= self.capacity_wh * 0.99
