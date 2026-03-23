# Lproject_sim/src/core/stellar_manager.py
"""
StellarManager - 달 표면에서 천체 위치 계산
OmniLRS의 StellarEngine을 참고하여 구현

달 표면 관측자의 위도/경도와 시간을 기반으로
태양, 지구 등 천체의 고도(altitude), 방위각(azimuth)을 계산합니다.
"""

from pxr import UsdLux, Gf, UsdGeom
from scipy.spatial.transform import Rotation as SSTR
from skyfield.api import PlanetaryConstants, load
from typing import Tuple, Optional
from dataclasses import dataclass, field
import datetime
import math
import logging
import os


@dataclass
class StellarConfig:
    """천체 엔진 설정"""
    # 관측자 위치 (달 표면 좌표)
    latitude: float = -26.3       # 위도 (도)
    longitude: float = 46.8       # 경도 (도)
    
    # 시작 시간 (UTC)
    start_year: int = 2024
    start_month: int = 5
    start_day: int = 1
    start_hour: int = 12
    start_minute: int = 0
    start_second: int = 0
    
    # 시간 설정
    time_scale: float = 1.0       # 시간 배속 (1.0 = 실시간, 60.0 = 1분/초)
    update_interval: float = 1.0  # 천체 위치 업데이트 간격 (초)
    
    # Ephemeris 파일 경로
    ephemeris_dir: str = ""       # 비어있으면 기본 경로 사용
    
    # 자동 업데이트
    auto_update: bool = True      # True면 매 프레임 자동 업데이트


class StellarManager:
    """
    달 표면에서 천체 위치를 계산하는 매니저
    
    사용법:
        stellar = StellarManager(config)
        stellar.set_lat_lon(latitude, longitude)
        stellar.update(dt)  # 매 프레임 호출
        alt, az, dist = stellar.get_sun_position()
    """
    
    def __init__(self, config: StellarConfig = None, assets_dir: str = None):
        """
        StellarManager 초기화
        
        Args:
            config: StellarConfig 설정 객체
            assets_dir: 에셋 디렉토리 경로 (Ephemeris 파일 위치)
        """
        self.config = config if config else StellarConfig()
        _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
        self.assets_dir = assets_dir or os.path.join(_project_root, "assets")
        
        # 시간 관리
        self.ts = load.timescale()
        self.current_time = self._create_start_datetime()
        self.last_update = datetime.datetime.fromtimestamp(0, datetime.timezone.utc)
        self.t = self.ts.from_datetime(self.current_time)
        
        # 천체 데이터 로드
        self._load_ephemeris()
        
        # 관측자 위치 설정
        self.observer = None
        self.set_lat_lon(self.config.latitude, self.config.longitude)
        
        # 캐시된 태양 위치
        self._cached_sun_alt = 0.0
        self._cached_sun_az = 0.0
        self._cached_sun_dist = 0.0
        
        logging.info(f"[StellarManager] Initialized at lat={self.config.latitude}°, "
                     f"lon={self.config.longitude}°, time={self.current_time}")
    
    def _create_start_datetime(self) -> datetime.datetime:
        """설정에서 시작 시간 생성"""
        return datetime.datetime(
            year=self.config.start_year,
            month=self.config.start_month,
            day=self.config.start_day,
            hour=self.config.start_hour,
            minute=self.config.start_minute,
            second=self.config.start_second,
            tzinfo=datetime.timezone.utc
        )
    
    def _load_ephemeris(self):
        """천체력(Ephemeris) 및 달 좌표계 데이터 로드"""
        eph_dir = self.config.ephemeris_dir or os.path.join(self.assets_dir, "Ephemeris")
        
        # Ephemeris 파일 경로
        ephemeris_file = os.path.join(eph_dir, "de421.bsp")
        moon_tf_file = os.path.join(eph_dir, "moon_080317.tf")
        pck_file = os.path.join(eph_dir, "pck00008.tpc")
        moon_pa_file = os.path.join(eph_dir, "moon_pa_de421_1900-2050.bpc")
        
        # 파일 존재 확인 및 다운로드
        if not os.path.exists(ephemeris_file):
            logging.warning(f"[StellarManager] Ephemeris not found at {ephemeris_file}, downloading...")
            self.eph = load("de421.bsp")
        else:
            self.eph = load(ephemeris_file)
        
        # 천체 정의
        self.earth = self.eph["earth"]
        self.moon = self.eph["moon"]
        self.sun = self.eph["sun"]
        self.venus = self.eph["venus"]
        self.bodies = {
            "earth": self.earth,
            "moon": self.moon,
            "sun": self.sun,
            "venus": self.venus
        }
        
        # 달 좌표계 (Principal Axis frame)
        self.pc = PlanetaryConstants()
        
        try:
            if os.path.exists(moon_tf_file):
                self.pc.read_text(load(moon_tf_file))
            else:
                self.pc.read_text(load("moon_080317.tf"))
            
            if os.path.exists(pck_file):
                self.pc.read_text(load(pck_file))
            else:
                self.pc.read_text(load("pck00008.tpc"))
            
            if os.path.exists(moon_pa_file):
                self.pc.read_binary(load(moon_pa_file))
            else:
                self.pc.read_binary(load("moon_pa_de421_1900-2050.bpc"))
            
            self.frame = self.pc.build_frame_named("MOON_ME_DE421")
            logging.info("[StellarManager] Ephemeris loaded successfully")
            
        except Exception as e:
            logging.error(f"[StellarManager] Failed to load planetary constants: {e}")
            logging.warning("[StellarManager] Using fallback mode without frame transformation")
            self.frame = None
    
    def set_lat_lon(self, lat: float, lon: float):
        """
        관측자의 달 표면 위치 설정
        
        Args:
            lat: 위도 (도, -90 ~ 90)
            lon: 경도 (도, -180 ~ 180)
        """
        self.config.latitude = lat
        self.config.longitude = lon
        
        if self.frame is not None:
            self.observer = self.moon + self.pc.build_latlon_degrees(self.frame, lat, lon)
        else:
            # Fallback: 달 중심 기준
            self.observer = self.moon
        
        logging.info(f"[StellarManager] Observer position set to lat={lat}°, lon={lon}°")
    
    def set_time(self, timestamp: float):
        """
        현재 시간을 Unix timestamp로 설정
        
        Args:
            timestamp: Unix timestamp (초, UTC)
        """
        self.current_time = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
        self.last_update = self.current_time
        self.t = self.ts.from_datetime(self.current_time)
        logging.debug(f"[StellarManager] Time set to {self.current_time}")
    
    def set_datetime(self, dt: datetime.datetime):
        """
        현재 시간을 datetime 객체로 설정
        
        Args:
            dt: datetime 객체 (timezone-aware 권장)
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        self.current_time = dt
        self.last_update = dt
        self.t = self.ts.from_datetime(self.current_time)
        logging.debug(f"[StellarManager] Time set to {self.current_time}")
    
    def set_time_scale(self, scale: float):
        """
        시간 배속 설정
        
        Args:
            scale: 시간 배속 (1.0 = 실시간, 60.0 = 1분/초, 3600.0 = 1시간/초)
        """
        self.config.time_scale = scale
        logging.info(f"[StellarManager] Time scale set to {scale}x")
    
    def update(self, dt: float) -> bool:
        """
        시간 업데이트 및 천체 위치 재계산
        
        Args:
            dt: 경과 시간 (초, 시뮬레이션 시간 기준)
        
        Returns:
            bool: True if celestial positions were updated
        """
        # 시간 배속 적용
        scaled_dt = dt * self.config.time_scale
        self.current_time += datetime.timedelta(seconds=scaled_dt)
        
        # 업데이트 간격 확인
        time_delta = (self.current_time - self.last_update).total_seconds()
        
        if time_delta >= self.config.update_interval:
            self.last_update = self.current_time
            self.t = self.ts.from_datetime(self.current_time)
            
            # 태양 위치 캐시 업데이트
            self._update_sun_cache()
            
            logging.debug(f"[StellarManager] Updated to {self.current_time}, "
                          f"Sun alt={self._cached_sun_alt:.2f}°, az={self._cached_sun_az:.2f}°")
            return True
        
        return False
    
    def _update_sun_cache(self):
        """태양 위치 캐시 업데이트"""
        try:
            self._cached_sun_alt, self._cached_sun_az, self._cached_sun_dist = self.get_alt_az("sun")
        except Exception as e:
            logging.warning(f"[StellarManager] Failed to update sun position: {e}")
    
    def get_alt_az(self, body: str) -> Tuple[float, float, float]:
        """
        천체의 고도, 방위각, 거리 계산
        
        Args:
            body: 천체 이름 ("sun", "earth", "venus")
        
        Returns:
            Tuple[altitude, azimuth, distance]:
                - altitude: 고도 (도, -90 ~ 90, 0=수평선)
                - azimuth: 방위각 (도, 0=북, 90=동, 180=남, 270=서)
                - distance: 거리 (m)
        """
        if self.observer is None:
            raise RuntimeError("Observer position not set. Call set_lat_lon() first.")
        
        apparent = self.observer.at(self.t).observe(self.bodies[body]).apparent()
        alt, az, distance = apparent.altaz()
        return alt.degrees, az.degrees, distance.m
    
    def get_sun_position(self) -> Tuple[float, float, float]:
        """
        태양의 현재 위치 (캐시된 값)
        
        Returns:
            Tuple[altitude, azimuth, distance]
        """
        return self._cached_sun_alt, self._cached_sun_az, self._cached_sun_dist
    
    def get_sun_direction(self) -> Tuple[float, float, float]:
        """
        태양 방향 벡터 계산 (정규화된 XYZ)
        
        Z-up 좌표계 기준:
            - X: 동쪽
            - Y: 북쪽  
            - Z: 위쪽
        
        Returns:
            Tuple[x, y, z]: 정규화된 방향 벡터
        """
        alt_rad = math.radians(self._cached_sun_alt)
        az_rad = math.radians(self._cached_sun_az)
        
        # 방위각: 0=북(+Y), 90=동(+X), 180=남(-Y), 270=서(-X)
        x = math.cos(alt_rad) * math.sin(az_rad)
        y = math.cos(alt_rad) * math.cos(az_rad)
        z = math.sin(alt_rad)
        
        return x, y, z
    
    def get_sun_rotation_quat(self) -> Tuple[float, float, float, float]:
        """
        USD DistantLight에 적용할 회전 쿼터니언 계산
        
        기본 빛 방향이 [0, 0, -1]인 DistantLight를 태양 방향으로 회전
        
        Returns:
            Tuple[w, x, y, z]: 쿼터니언 (USD 순서: w, x, y, z)
        """
        alt = self._cached_sun_alt
        az = self._cached_sun_az
        
        # OmniLRS 방식: Z축 회전(방위각) → X축 회전(고도)
        x, y, z, w = SSTR.from_euler("xyz", [0, alt, az - 90], degrees=True).as_quat()
        return (w, x, y, z)
    
    def get_sun_euler_for_distant_light(self) -> Tuple[float, float]:
        """
        USD DistantLight에 적용할 오일러 각도
        
        SceneManager의 setup_lighting과 호환되는 형식
        
        Returns:
            Tuple[rotation_z, rotation_x]: (azimuth, -(90-elevation))
        """
        rotation_z = self._cached_sun_az  # 방위각
        rotation_x = -(90.0 - self._cached_sun_alt)  # 고도 변환
        return rotation_z, rotation_x
    
    def is_sun_visible(self) -> bool:
        """태양이 수평선 위에 있는지 확인"""
        return self._cached_sun_alt > 0
    
    def get_current_time(self) -> datetime.datetime:
        """현재 시뮬레이션 시간 반환"""
        return self.current_time
    
    def get_current_time_str(self) -> str:
        """현재 시간을 문자열로 반환"""
        return self.current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    def get_info(self) -> dict:
        """현재 상태 정보 반환"""
        return {
            "time": self.get_current_time_str(),
            "latitude": self.config.latitude,
            "longitude": self.config.longitude,
            "time_scale": self.config.time_scale,
            "sun_altitude": self._cached_sun_alt,
            "sun_azimuth": self._cached_sun_az,
            "sun_visible": self.is_sun_visible()
        }


class StellarSceneUpdater:
    """
    StellarManager와 USD Scene을 연결하는 업데이터
    
    SceneManager의 태양 조명을 StellarManager의 계산 결과로 업데이트합니다.
    """
    
    def __init__(self, stage, stellar_manager: StellarManager, sun_prim_path: str = "/World/Sun"):
        """
        초기화
        
        Args:
            stage: USD Stage
            stellar_manager: StellarManager 인스턴스
            sun_prim_path: 태양 조명 Prim 경로
        """
        self.stage = stage
        self.stellar = stellar_manager
        self.sun_prim_path = sun_prim_path
        self.sun_light = None
        self._find_sun_light()
    
    def _find_sun_light(self):
        """USD Stage에서 태양 조명 찾기"""
        prim = self.stage.GetPrimAtPath(self.sun_prim_path)
        if prim.IsValid():
            self.sun_light = UsdLux.DistantLight(prim)
            print(f"[StellarSceneUpdater] ✓ Found sun light at {self.sun_prim_path}")
        else:
            print(f"[StellarSceneUpdater] ✗ Sun light NOT found at {self.sun_prim_path}")
            self.sun_light = None
    
    def update(self, dt: float) -> bool:
        """
        씬 업데이트
        
        Args:
            dt: 경과 시간 (초)
        
        Returns:
            bool: True if sun position was updated
        """
        updated = self.stellar.update(dt)
        
        if updated and self.sun_light is not None:
            self._update_sun_transform()
            # 주기적 로그 (10번에 1번)
            if hasattr(self, '_update_count'):
                self._update_count += 1
            else:
                self._update_count = 1
            
            if self._update_count % 10 == 1:
                rotation_z, rotation_x = self.stellar.get_sun_euler_for_distant_light()
        elif updated and self.sun_light is None:
            if not hasattr(self, '_warned_no_light'):
                print(f"[StellarSceneUpdater] Warning: Sun light not found, cannot update transform")
                self._warned_no_light = True
        
        return updated
    
    def _update_sun_transform(self):
        """태양 조명 Transform 업데이트"""
        rotation_z, rotation_x = self.stellar.get_sun_euler_for_distant_light()
        
        xform = UsdGeom.Xformable(self.sun_light)
        
        # 기존 xformOp 가져오기 또는 생성
        xform_ops = xform.GetOrderedXformOps()
        
        # XformOp이 이미 있으면 값만 업데이트
        updated_z = False
        updated_x = False
        if len(xform_ops) >= 2:
            for op in xform_ops:
                op_name = op.GetOpName()
                if "rotateZ" in op_name:
                    op.Set(rotation_z)
                    updated_z = True
                elif "rotateX" in op_name:
                    op.Set(rotation_x)
                    updated_x = True
        
        # 업데이트 안 된 경우 새로 생성
        if not updated_z or not updated_x:
            xform.ClearXformOpOrder()
            xform.AddRotateZOp().Set(rotation_z)
            xform.AddRotateXOp().Set(rotation_x)
            print(f"[StellarSceneUpdater] Created new xformOps for sun light")
        
        # 태양이 수평선 아래면 intensity를 0으로
        if self.stellar.is_sun_visible():
            # 고도에 따른 intensity 조절 (선택사항)
            pass
        else:
            # 밤일 때 처리 (선택사항)
            pass
    
    def force_update(self):
        """강제로 태양 위치 업데이트"""
        self.stellar._update_sun_cache()
        if self.sun_light is not None:
            self._update_sun_transform()


# 설정 파일에서 StellarConfig 생성하는 헬퍼 함수
def create_stellar_config_from_yaml(config: dict) -> StellarConfig:
    """
    YAML 설정에서 StellarConfig 생성
    
    Args:
        config: simulation_config.yaml의 stellar 섹션
    
    Returns:
        StellarConfig 인스턴스
    """
    stellar_cfg = config.get("stellar", {})
    
    # 시작 날짜 파싱
    start_date = stellar_cfg.get("start_date", {})
    
    return StellarConfig(
        latitude=stellar_cfg.get("latitude", -26.3),
        longitude=stellar_cfg.get("longitude", 46.8),
        start_year=start_date.get("year", 2024),
        start_month=start_date.get("month", 5),
        start_day=start_date.get("day", 1),
        start_hour=start_date.get("hour", 12),
        start_minute=start_date.get("minute", 0),
        start_second=start_date.get("second", 0),
        time_scale=stellar_cfg.get("time_scale", 1.0),
        update_interval=stellar_cfg.get("update_interval", 1.0),
        ephemeris_dir=stellar_cfg.get("ephemeris_dir", ""),
        auto_update=stellar_cfg.get("auto_update", True)
    )


if __name__ == "__main__":
    # 테스트 코드
    import time
    
    logging.basicConfig(level=logging.DEBUG)
    
    # 설정 생성
    config = StellarConfig(
        latitude=-26.3,  # 달 남반구
        longitude=46.8,
        start_year=2024,
        start_month=5,
        start_day=1,
        start_hour=12,
        start_minute=0,
        time_scale=3600.0,  # 1시간/초
        update_interval=0.1
    )
    
    # StellarManager 생성
    stellar = StellarManager(config)
    
    print(f"\n=== Lunar Stellar Manager Test ===")
    print(f"Location: lat={config.latitude}°, lon={config.longitude}°")
    print(f"Start time: {stellar.get_current_time_str()}")
    print(f"Time scale: {config.time_scale}x\n")
    
    # 24시간 시뮬레이션 (실제 24초)
    for i in range(24):
        stellar.update(1.0)  # 1초 (= 1시간 시뮬레이션 시간)
        
        alt, az, dist = stellar.get_sun_position()
        visible = "☀️" if stellar.is_sun_visible() else "🌑"
        
        
        time.sleep(0.1)
    
    print(f"\n=== Test Complete ===")
