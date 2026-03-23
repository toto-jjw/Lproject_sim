# Lproject_sim/src/physics/wheel_track_renderer.py
"""
실시간 바퀴자국 렌더링 시스템 (Groove Track)
- 지형 메시 변형 없이 바퀴자국을 시각적으로 표현
- U자형 홈(groove) 메시로 실제 침하된 모양을 재현
- 바퀴당 max ~2000 vertices → 전체 ~8000 vertices (4M 대비 500배 가벼움)
"""

import numpy as np
from collections import deque
from pxr import UsdGeom, Gf, Sdf, UsdShade


class WheelTrackRenderer:
    """
    바퀴자국을 U-groove 메시로 렌더링하는 클래스
    
    Cross-section (단면):
        outer_left   inner_left ─────── inner_right   outer_right
            \            ╲___________________╱            /
             (terrain)       (sinkage depth)       (terrain)
    
    4 vertices per cross-section:
      outer_left  → terrain level (berm rim)
      inner_left  → lowered by groove_depth
      inner_right → lowered by groove_depth
      outer_right → terrain level (berm rim)
    """
    
    VERTS_PER_SECTION = 4
    
    def __init__(self, stage, terrain_manager, num_wheels: int = 4,
                 track_width: float = 0.1,
                 max_points_per_wheel: int = 500,
                 min_distance: float = 0.05,
                 track_height_offset: float = 0.001):
        self.stage = stage
        self.tm = terrain_manager
        self.num_wheels = num_wheels
        self.track_width = track_width
        self.max_points = max_points_per_wheel
        self.min_distance = min_distance
        self.height_offset = track_height_offset
        
        # 각 바퀴의 경로 히스토리 — 4 vertices per section
        self.wheel_paths = [deque(maxlen=max_points_per_wheel) for _ in range(num_wheels)]
        
        # 마지막 위치 (최소 거리 체크용)
        self.last_positions = [None] * num_wheels
        
        # Cached index/facecount arrays (rebuilt only when section count changes)
        self._cached_counts = [0] * num_wheels
        self._cached_indices = [None] * num_wheels
        self._cached_face_counts = [None] * num_wheels
        
        # USD 메시 프림
        self.track_meshes = []
        self._create_track_prims()
        
        # 업데이트 카운터
        self.update_counter = 0
        self.mesh_update_interval = 5
        
        max_verts = max_points_per_wheel * self.VERTS_PER_SECTION * num_wheels
        print(f"[WheelTrackRenderer] Groove mode: {num_wheels} wheels, "
              f"max {max_points_per_wheel} pts/wheel, ~{max_verts} max verts")
    
    def _create_track_prims(self):
        """바퀴자국 메시 프림 생성"""
        tracks_path = "/World/WheelTracks"
        UsdGeom.Xform.Define(self.stage, tracks_path)
        
        # 머티리얼 (어두운 색상 = 다져진 레골리스)
        material_path = "/World/Looks/WheelTrackMaterial"
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, f"{material_path}/Shader")
        shader.CreateIdAttr().Set("UsdPreviewSurface")
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader_output)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.06, 0.06, 0.08))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.98)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        
        for i in range(self.num_wheels):
            mesh_path = f"{tracks_path}/Track_{i}"
            mesh = UsdGeom.Mesh.Define(self.stage, mesh_path)
            mesh.GetPointsAttr().Set([])
            mesh.GetFaceVertexIndicesAttr().Set([])
            mesh.GetFaceVertexCountsAttr().Set([])
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
            mesh.GetDoubleSidedAttr().Set(True)
            self.track_meshes.append(mesh)
    
    def update(self, wheel_positions: np.ndarray, forward_dir: np.ndarray,
               terrain_z_values: np.ndarray, is_moving: bool,
               sinkages: np.ndarray = None):
        """
        Args:
            wheel_positions: (4, 2) [x, y]
            forward_dir: [x, y] normalised rover heading
            terrain_z_values: (4,) terrain height per wheel
            is_moving: whether rover is moving
            sinkages: (4,) Bekker sinkage per wheel [m] (used for groove depth)
        """
        if not is_moving:
            return
        
        if sinkages is None:
            sinkages = np.full(self.num_wheels, 0.005)
        
        # 바퀴 폭 방향 (전진 방향의 수직)
        right_dir = np.array([forward_dir[1], -forward_dir[0]])
        half_w = self.track_width / 2.0
        outer_w = half_w * 1.3  # berm rim slightly wider
        
        dirty = False
        for i in range(min(len(wheel_positions), self.num_wheels)):
            pos_xy = wheel_positions[i]
            
            # 최소 거리 체크
            if self.last_positions[i] is not None:
                dist = np.linalg.norm(pos_xy - self.last_positions[i])
                if dist < self.min_distance:
                    continue
            
            self.last_positions[i] = pos_xy.copy()
            terrain_z = float(terrain_z_values[i])
            groove_depth = max(0.001, min(float(sinkages[i]), 0.05))
            
            z_top = terrain_z + self.height_offset
            z_bot = terrain_z - groove_depth + self.height_offset
            
            # 4 cross-section vertices
            ol = (pos_xy[0] - right_dir[0] * outer_w,
                  pos_xy[1] - right_dir[1] * outer_w,
                  z_top)
            il = (pos_xy[0] - right_dir[0] * half_w,
                  pos_xy[1] - right_dir[1] * half_w,
                  z_bot)
            ir = (pos_xy[0] + right_dir[0] * half_w,
                  pos_xy[1] + right_dir[1] * half_w,
                  z_bot)
            outr = (pos_xy[0] + right_dir[0] * outer_w,
                    pos_xy[1] + right_dir[1] * outer_w,
                    z_top)
            
            self.wheel_paths[i].append((ol, il, ir, outr))
            dirty = True
        
        # 메시 업데이트 (일정 간격)
        self.update_counter += 1
        if dirty and self.update_counter >= self.mesh_update_interval:
            self.update_counter = 0
            self._update_meshes()
    
    def _build_indices(self, num_sections: int):
        """
        Build triangle indices for groove strip (3 quads per segment).
        
        Between consecutive sections:
          quad 0: outer_left  → inner_left   (left berm slope)
          quad 1: inner_left  → inner_right  (track bottom)
          quad 2: inner_right → outer_right  (right berm slope)
        """
        V = self.VERTS_PER_SECTION  # 4
        n = num_sections - 1
        # 3 quads × 2 triangles × 3 indices = 18 per segment
        indices = np.empty(n * 18, dtype=np.int32)
        
        seg = np.arange(n, dtype=np.int32)
        base = seg * V
        nxt = base + V
        
        # Quad 0 (outer_left → inner_left)
        indices[0::18] = base
        indices[1::18] = base + 1
        indices[2::18] = nxt
        indices[3::18] = base + 1
        indices[4::18] = nxt + 1
        indices[5::18] = nxt
        
        # Quad 1 (inner_left → inner_right = bottom)
        indices[6::18]  = base + 1
        indices[7::18]  = base + 2
        indices[8::18]  = nxt + 1
        indices[9::18]  = base + 2
        indices[10::18] = nxt + 2
        indices[11::18] = nxt + 1
        
        # Quad 2 (inner_right → outer_right)
        indices[12::18] = base + 2
        indices[13::18] = base + 3
        indices[14::18] = nxt + 2
        indices[15::18] = base + 3
        indices[16::18] = nxt + 3
        indices[17::18] = nxt + 2
        
        face_counts = np.full(n * 6, 3, dtype=np.int32)
        return indices.tolist(), face_counts.tolist()
    
    def _update_meshes(self):
        """Update USD meshes from path data."""
        for wheel_idx in range(self.num_wheels):
            path = self.wheel_paths[wheel_idx]
            mesh = self.track_meshes[wheel_idx]
            n = len(path)
            if n < 2:
                continue
            
            # Build vertex array (n × 4 = lightweight)
            vertices = []
            for ol, il, ir, outr in path:
                vertices.append(Gf.Vec3f(float(ol[0]), float(ol[1]), float(ol[2])))
                vertices.append(Gf.Vec3f(float(il[0]), float(il[1]), float(il[2])))
                vertices.append(Gf.Vec3f(float(ir[0]), float(ir[1]), float(ir[2])))
                vertices.append(Gf.Vec3f(float(outr[0]), float(outr[1]), float(outr[2])))
            
            # Rebuild indices only when section count changes
            if n != self._cached_counts[wheel_idx]:
                idx_list, fc_list = self._build_indices(n)
                self._cached_indices[wheel_idx] = idx_list
                self._cached_face_counts[wheel_idx] = fc_list
                self._cached_counts[wheel_idx] = n
                mesh.GetFaceVertexIndicesAttr().Set(idx_list)
                mesh.GetFaceVertexCountsAttr().Set(fc_list)
            
            # Only vertices change each update (~2000 max per wheel)
            mesh.GetPointsAttr().Set(vertices)
    
    def clear(self):
        """바퀴자국 초기화"""
        for path in self.wheel_paths:
            path.clear()
        for mesh in self.track_meshes:
            mesh.GetPointsAttr().Set([])
            mesh.GetFaceVertexIndicesAttr().Set([])
            mesh.GetFaceVertexCountsAttr().Set([])
        self.last_positions = [None] * self.num_wheels
        self._cached_counts = [0] * self.num_wheels
    
    def cleanup(self):
        """리소스 정리"""
        self.clear()
