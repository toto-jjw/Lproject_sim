# Lproject_sim/src/core/scene_io.py
"""
USD Scene 저장/불러오기 유틸리티

시뮬레이션 환경(지형, 바위, 조명 등)을 USD 파일로 저장하고
나중에 빠르게 불러올 수 있습니다.
"""

import os
from datetime import datetime
from pxr import Usd, UsdGeom, Sdf
from typing import Optional, Dict, Any
import os as _os

# Project root
_PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

class SceneIO:
    """USD Scene 저장/불러오기 관리자"""
    
    DEFAULT_SCENE_DIR = _os.path.join(_PROJECT_ROOT, "assets", "samplescene")
    DEFAULT_SCENE_NAME = "lunar_scene.usd"
    
    # Scene variants: 3 rock levels × 2 light conditions × 2 terrain modes = 12 scenes
    # bright = high sun altitude, dim = low sun altitude
    # s1~s6: with outer terrain, s7~s12: without outer terrain (inner terrain only)
    SCENE_VARIANTS = {
        # With outer terrain
        "s1": "lunar_scene_full_rocks_bright.usd",           # Full rocks, bright
        "s2": "lunar_scene_full_rocks_dim.usd",              # Full rocks, dim
        "s3": "lunar_scene_half_rocks_bright.usd",           # Half rocks, bright
        "s4": "lunar_scene_half_rocks_dim.usd",              # Half rocks, dim
        "s5": "lunar_scene_no_rocks_bright.usd",             # No rocks, bright
        "s6": "lunar_scene_no_rocks_dim.usd",                # No rocks, dim
        # Without outer terrain (inner terrain only)
        "s7": "lunar_scene_full_rocks_bright_nouter.usd",    # Full rocks, bright, no outer
        "s8": "lunar_scene_full_rocks_dim_nouter.usd",       # Full rocks, dim, no outer
        "s9": "lunar_scene_half_rocks_bright_nouter.usd",    # Half rocks, bright, no outer
        "s10": "lunar_scene_half_rocks_dim_nouter.usd",      # Half rocks, dim, no outer
        "s11": "lunar_scene_no_rocks_bright_nouter.usd",     # No rocks, bright, no outer
        "s12": "lunar_scene_no_rocks_dim_nouter.usd",        # No rocks, dim, no outer
    }
    
    # Light presets for scene variants
    LIGHT_PRESETS = {
        "bright": {"elevation": 11.4, "azimuth": 6.5},  # High sun, afternoon
        "dim": {"elevation": 4.3, "azimuth": 55.5},       # Low sun, sunrise/sunset
    }
    
    def __init__(self, stage: Usd.Stage):
        """
        Args:
            stage: USD Stage 객체
        """
        self.stage = stage
    
    @staticmethod
    def get_default_scene_path() -> str:
        """기본 샘플 씬 경로 반환"""
        return os.path.join(SceneIO.DEFAULT_SCENE_DIR, SceneIO.DEFAULT_SCENE_NAME)
    
    @staticmethod
    def get_scene_variant_path(variant: str) -> str:
        """씬 변형 경로 반환
        
        Args:
            variant: "s1" ~ "s6"
        
        Returns:
            Scene file path
        """
        if variant not in SceneIO.SCENE_VARIANTS:
            raise ValueError(f"Unknown variant: {variant}. Use 's1' through 's12'")
        return os.path.join(SceneIO.DEFAULT_SCENE_DIR, SceneIO.SCENE_VARIANTS[variant])
    
    def save_scene(self, 
                   output_path: Optional[str] = None,
                   include_robots: bool = False,
                   flatten: bool = False) -> str:
        """
        현재 USD Stage를 파일로 저장
        
        Args:
            output_path: 저장 경로 (None이면 기본 경로 사용)
            include_robots: 로봇 포함 여부 (False 권장 - 로봇은 동적으로 스폰)
            flatten: Stage flatten 여부 (True면 모든 레이어 병합)
            
        Returns:
            저장된 파일 경로
        """
        if output_path is None:
            output_path = self.get_default_scene_path()
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not include_robots:
            # 로봇 제외하고 저장 (임시 Stage 생성)
            output_path = self._save_without_robots(output_path, flatten)
        else:
            if flatten:
                self.stage.Flatten().Export(output_path)
            else:
                self.stage.Export(output_path)
        
        print(f"[SceneIO] Scene saved to: {output_path}")
        return output_path
    
    def _save_without_robots(self, output_path: str, flatten: bool) -> str:
        """로봇을 제외하고 씬 저장"""
        from pxr import Usd, Sdf
        
        # 저장할 Prim 경로 목록 (로봇 제외)
        robot_paths = []
        for prim in self.stage.Traverse():
            path_str = str(prim.GetPath())
            # 로봇 관련 Prim 식별 (일반적인 패턴)
            if any(keyword in path_str.lower() for keyword in 
                   ['husky', 'jackal', 'robot', 'rover']):
                # 최상위 로봇 Prim만 추가 (자식은 자동 제외됨)
                parent_path = prim.GetParent().GetPath()
                if str(parent_path) == '/World':
                    robot_paths.append(path_str)
        
        # 중복 제거
        robot_paths = list(set(robot_paths))
        print(f"[SceneIO] Found {len(robot_paths)} robot prims to exclude: {robot_paths}")
        
        if flatten:
            # Flatten은 SdfLayer를 직접 반환함
            layer = self.stage.Flatten()
            
            # 로봇 제거 - SdfLayer에서 PrimSpec 제거
            for robot_path in robot_paths:
                try:
                    prim_spec = layer.GetPrimAtPath(robot_path)
                    if prim_spec:
                        # 부모 PrimSpec에서 자식 제거
                        parent_path = Sdf.Path(robot_path).GetParentPath()
                        parent_spec = layer.GetPrimAtPath(parent_path)
                        if parent_spec:
                            child_name = robot_path.split('/')[-1]
                            del parent_spec.nameChildren[child_name]
                            print(f"[SceneIO] Removed robot: {robot_path}")
                except Exception as e:
                    print(f"[SceneIO] Warning: Could not remove {robot_path}: {e}")
            
            layer.Export(output_path)
        else:
            # 원본 Stage 복사 후 로봇 제거
            self.stage.Export(output_path)
            
            # 저장된 파일에서 로봇 제거
            temp_stage = Usd.Stage.Open(output_path)
            for robot_path in robot_paths:
                prim = temp_stage.GetPrimAtPath(robot_path)
                if prim and prim.IsValid():
                    temp_stage.RemovePrim(robot_path)
                    print(f"[SceneIO] Removed robot: {robot_path}")
            temp_stage.Save()
        
        print(f"[SceneIO] Excluded {len(robot_paths)} robot prims from saved scene")
        return output_path
    
    @staticmethod
    def load_scene(scene_path: Optional[str] = None) -> Optional[str]:
        """
        저장된 USD Scene 경로 확인
        
        Args:
            scene_path: 씬 파일 경로 (None이면 기본 경로)
            
        Returns:
            유효한 씬 파일 경로 또는 None
        """
        if scene_path is None:
            scene_path = SceneIO.get_default_scene_path()
        
        if os.path.exists(scene_path):
            print(f"[SceneIO] Found saved scene: {scene_path}")
            return scene_path
        else:
            print(f"[SceneIO] Scene not found: {scene_path}")
            return None
    
    @staticmethod
    def scene_exists(scene_path: Optional[str] = None) -> bool:
        """저장된 씬 파일이 존재하는지 확인"""
        if scene_path is None:
            scene_path = SceneIO.get_default_scene_path()
        return os.path.exists(scene_path)


def save_current_scene(stage: Usd.Stage, 
                       output_path: Optional[str] = None,
                       include_robots: bool = False) -> str:
    """
    편의 함수: 현재 씬을 저장
    
    사용법 (Isaac Sim 내에서):
        from src.core.scene_io import save_current_scene
        save_current_scene(world.stage)
    """
    io = SceneIO(stage)
    return io.save_scene(output_path, include_robots=include_robots, flatten=True)
