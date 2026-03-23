# Lproject_sim/src/sdg/annotator.py
"""
SDG Annotator - 자동 라벨링 및 데이터 저장

Omni Replicator를 사용한 합성 데이터 생성:
- RGB 이미지
- Depth 맵
- Semantic Segmentation
- Instance Segmentation
- 2D/3D Bounding Box
- Surface Normals

출력 구조:
data/
├── rgb/                 # RGB 이미지
├── depth/               # Depth 맵 (EXR)
├── semantic/            # Semantic Segmentation
├── instance/            # Instance Segmentation
├── bbox_2d/             # 2D Bounding Box (JSON)
├── bbox_3d/             # 3D Bounding Box (JSON)
├── normals/             # Surface Normals
└── metadata.json        # 전체 메타데이터
"""
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry
import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Any

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_SDG_OUTPUT = os.path.join(_PROJECT_ROOT, "data", "sdg_output")


class AnnotatorConfig:
    """어노테이터 설정"""
    def __init__(
        self,
        output_dir: str = None,
        rgb: bool = True,
        depth: bool = True,
        semantic_segmentation: bool = True,
        instance_segmentation: bool = False,
        bounding_box_2d_tight: bool = True,
        bounding_box_2d_loose: bool = False,
        bounding_box_3d: bool = False,
        normals: bool = False,
        distance_to_camera: bool = False,
        image_format: str = "png",          # png, jpg, exr
        depth_format: str = "npy",          # npy, exr
        resolution: tuple = (1280, 720),
        colorize_semantic: bool = True,
        colorize_instance: bool = True,
    ):
        self.output_dir = output_dir or _DEFAULT_SDG_OUTPUT
        self.rgb = rgb
        self.depth = depth
        self.semantic_segmentation = semantic_segmentation
        self.instance_segmentation = instance_segmentation
        self.bounding_box_2d_tight = bounding_box_2d_tight
        self.bounding_box_2d_loose = bounding_box_2d_loose
        self.bounding_box_3d = bounding_box_3d
        self.normals = normals
        self.distance_to_camera = distance_to_camera
        self.image_format = image_format
        self.depth_format = depth_format
        self.resolution = resolution
        self.colorize_semantic = colorize_semantic
        self.colorize_instance = colorize_instance


class Annotator:
    """
    SDG용 자동 라벨링 및 데이터 저장 시스템
    
    Features:
    - 다양한 출력 형식 지원 (RGB, Depth, Segmentation, BBox)
    - Replicator Writer를 통한 자동 저장
    - 메타데이터 기록
    - 트리거 기반 캡처
    """
    
    def __init__(self, output_dir: str = None, 
                 config: AnnotatorConfig = None):
        """
        Args:
            output_dir: 출력 디렉토리 경로 (None이면 기본값 사용)
            config: AnnotatorConfig 객체 (None이면 기본값 사용)
        """
        output_dir = output_dir or _DEFAULT_SDG_OUTPUT
        if config:
            self.config = config
            self.config.output_dir = output_dir
        else:
            self.config = AnnotatorConfig(output_dir=output_dir)
            
        self.output_dir = output_dir
        self.render_product = None
        self.writer = None
        self.annotators: Dict[str, Any] = {}
        
        self._frame_count = 0
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
    def setup(self, camera_prim_path: str, resolution: tuple = None, interval: int = 100):
        """
        Replicator 그래프 및 Writer 설정
        
        Args:
            camera_prim_path: 카메라 prim 경로
            resolution: 해상도 (width, height). None이면 config 값 사용
            interval: 캡처 간격 (프레임 수)
        """
        if resolution:
            self.config.resolution = resolution
            
        try:
            # Render Product 생성
            self.render_product = rep.create.render_product(
                camera_prim_path, 
                self.config.resolution
            )
            
            # Writer 초기화
            self._setup_writer(interval)
            
            # 메타데이터 저장
            self._save_session_metadata(camera_prim_path, interval)
            
            print(f"[Annotator] Setup complete:")
            print(f"  - Camera: {camera_prim_path}")
            print(f"  - Resolution: {self.config.resolution}")
            print(f"  - Interval: {interval} frames")
            print(f"  - Output: {self.output_dir}")
            
        except Exception as e:
            print(f"[Annotator] Error during setup: {e}")
            import traceback
            traceback.print_exc()
            
    def _setup_writer(self, interval: int):
        """
        BasicWriter 설정
        """
        # 세션별 출력 디렉토리
        session_dir = os.path.join(self.output_dir, self._session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Writer 초기화
        self.writer = rep.WriterRegistry.get("BasicWriter")
        
        # Writer 파라미터 설정
        writer_params = {
            "output_dir": session_dir,
            "rgb": self.config.rgb,
        }
        
        # 선택적 출력 추가
        if self.config.depth:
            writer_params["distance_to_image_plane"] = True
            
        if self.config.semantic_segmentation:
            writer_params["semantic_segmentation"] = True
            writer_params["colorize_semantic_segmentation"] = self.config.colorize_semantic
            
        if self.config.instance_segmentation:
            writer_params["instance_segmentation"] = True
            writer_params["colorize_instance_segmentation"] = self.config.colorize_instance
            
        if self.config.bounding_box_2d_tight:
            writer_params["bounding_box_2d_tight"] = True
            
        if self.config.bounding_box_2d_loose:
            writer_params["bounding_box_2d_loose"] = True
            
        if self.config.bounding_box_3d:
            writer_params["bounding_box_3d"] = True
            
        if self.config.normals:
            writer_params["normals"] = True
            
        if self.config.distance_to_camera:
            writer_params["distance_to_camera"] = True
            
        self.writer.initialize(**writer_params)
        
        # Trigger와 함께 Render Product 연결
        self.writer.attach(
            [self.render_product], 
            trigger=rep.trigger.on_frame(interval=interval)
        )
        
        print(f"[Annotator] Writer configured with outputs: {list(writer_params.keys())}")
        
    def _save_session_metadata(self, camera_path: str, interval: int):
        """
        세션 메타데이터 저장
        """
        metadata = {
            "session_id": self._session_id,
            "timestamp": datetime.now().isoformat(),
            "camera_path": camera_path,
            "resolution": self.config.resolution,
            "interval": interval,
            "outputs": {
                "rgb": self.config.rgb,
                "depth": self.config.depth,
                "semantic_segmentation": self.config.semantic_segmentation,
                "instance_segmentation": self.config.instance_segmentation,
                "bounding_box_2d_tight": self.config.bounding_box_2d_tight,
                "bounding_box_2d_loose": self.config.bounding_box_2d_loose,
                "bounding_box_3d": self.config.bounding_box_3d,
                "normals": self.config.normals,
            }
        }
        
        session_dir = os.path.join(self.output_dir, self._session_id)
        metadata_path = os.path.join(session_dir, "metadata.json")
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
    def setup_semantic_labels(self, label_mapping: Dict[str, str] = None):
        """
        Semantic Segmentation 라벨 설정 (PrimSemanticData API 사용)
        
        Replicator의 semantic_segmentation annotator가 인식할 수 있도록
        semantics.schema.editor.PrimSemanticData를 사용합니다.
        
        Args:
            label_mapping: prim 경로 패턴 -> 클래스 이름 매핑
                          예: {"/World/Rocks/**": "rock", "/World/Terrain": "terrain"}
        """
        if label_mapping is None:
            # 기본 라벨 매핑 — ** 패턴으로 자식 prims도 매칭
            label_mapping = {
                "/World/Terrain": "terrain",
                "/World/Terrain/**": "terrain",
                "/World/Rocks": "rock",
                "/World/Rocks/**": "rock",
                "/World/Husky_*": "rover",
                "/World/husky_*": "rover",
                "/World/OuterTerrain": "terrain",
                "/World/OuterTerrain/**": "terrain",
            }
            
        try:
            from semantics.schema.editor import PrimSemanticData
            
            try:
                from isaacsim.core.utils.stage import get_current_stage
            except ImportError:
                from omni.isaac.core.utils.stage import get_current_stage
            
            stage = get_current_stage()
            if not stage:
                print("[Annotator] Warning: Stage not available for semantic setup")
                return
            
            labeled_count = 0
            
            for pattern, class_name in label_mapping.items():
                try:
                    from pxr import Usd
                    
                    # 패턴에 맞는 prims 찾기
                    for prim in stage.Traverse():
                        prim_path = str(prim.GetPath())
                        
                        if self._match_pattern(prim_path, pattern):
                            try:
                                prim_sd = PrimSemanticData(prim)
                                prim_sd.add_entry("class", class_name)
                                labeled_count += 1
                            except Exception as e:
                                # Fallback: pxr Semantics API
                                try:
                                    from pxr import Semantics
                                    if not prim.HasAPI(Semantics.SemanticsAPI):
                                        Semantics.SemanticsAPI.Apply(prim, "Semantics")
                                    sem_api = Semantics.SemanticsAPI.Get(prim, "Semantics")
                                    sem_api.CreateSemanticTypeAttr().Set("class")
                                    sem_api.CreateSemanticDataAttr().Set(class_name)
                                    labeled_count += 1
                                except:
                                    pass
                                    
                except Exception as e:
                    print(f"[Annotator] Warning: Could not apply semantic label for {pattern}: {e}")
                    
            print(f"[Annotator] Semantic labels configured: {labeled_count} prims labeled")
            print(f"[Annotator] Label mapping: {label_mapping}")
            
        except ImportError:
            print("[Annotator] Warning: semantics.schema.editor not available, trying fallback")
            self._setup_semantic_labels_fallback(label_mapping)
        except Exception as e:
            print(f"[Annotator] Error setting up semantic labels: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_semantic_labels_fallback(self, label_mapping: Dict[str, str]):
        """add_update_semantics 기반 fallback"""
        try:
            try:
                from isaacsim.core.utils.stage import get_current_stage
                from isaacsim.core.utils.semantics import add_update_semantics
            except ImportError:
                from omni.isaac.core.utils.stage import get_current_stage
                from omni.isaac.core.utils.semantics import add_update_semantics
            
            stage = get_current_stage()
            if not stage:
                return
            
            labeled_count = 0
            for pattern, class_name in label_mapping.items():
                for prim in stage.Traverse():
                    prim_path = str(prim.GetPath())
                    if self._match_pattern(prim_path, pattern):
                        try:
                            add_update_semantics(
                                prim=prim,
                                semantic_label=class_name,
                                type_label="class"
                            )
                            labeled_count += 1
                        except:
                            pass
            print(f"[Annotator] Fallback semantic labels: {labeled_count} prims labeled")
        except Exception as e:
            print(f"[Annotator] Fallback also failed: {e}")
            
    def _match_pattern(self, path: str, pattern: str) -> bool:
        """
        간단한 패턴 매칭 (*, ** 지원)
        
        *  → 한 레벨의 경로 세그먼트 매칭 (/ 제외)
        ** → 모든 하위 경로 매칭
        
        예:
        - "/World/Rocks" → 정확히 /World/Rocks 만 매칭
        - "/World/Rocks/**" → /World/Rocks 하위 모든 prims
        - "/World/Husky_*" → /World/Husky_1, /World/Husky_2 등
        """
        import fnmatch
        
        # ** 패턴 처리: 하위 모든 경로 매칭
        if pattern.endswith("/**"):
            prefix = pattern[:-3]  # "/**" 제거
            return path.startswith(prefix + "/")
        
        return fnmatch.fnmatch(path, pattern)
        
    def capture_frame(self) -> bool:
        """
        수동으로 한 프레임 캡처
        
        Returns:
            성공 여부
        """
        try:
            if self.writer:
                rep.orchestrator.step()
                self._frame_count += 1
                print(f"[Annotator] Captured frame {self._frame_count}")
                return True
        except Exception as e:
            print(f"[Annotator] Error capturing frame: {e}")
        return False
        
    def get_frame_count(self) -> int:
        """캡처된 프레임 수 반환"""
        return self._frame_count
        
    def get_output_dir(self) -> str:
        """현재 세션의 출력 디렉토리 반환"""
        return os.path.join(self.output_dir, self._session_id)
        
    def cleanup(self):
        """
        리소스 정리
        """
        try:
            if self.writer:
                # Writer detach
                self.writer = None
            if self.render_product:
                self.render_product = None
            print(f"[Annotator] Cleanup complete. Total frames: {self._frame_count}")
        except Exception as e:
            print(f"[Annotator] Error during cleanup: {e}")
