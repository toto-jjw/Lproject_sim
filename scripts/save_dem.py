#!/usr/bin/env python3
"""
시뮬레이션 환경의 전체 DEM 데이터를 저장하는 스크립트 (Ground Truth)

저장되는 데이터:
1. terrain_dem.npy: 기본 지형 DEM (craters, noise 포함)
2. rock_heightmap.npy: Rock 높이 맵 (실제 Mesh geometry 기반)
3. combined_dem.npy: 지형 + Rock 결합 DEM (Ground Truth)
4. dem_metadata.json: 메타데이터 (해상도, 크기, 오프셋 등)
5. pointcloud.ply: 전체 환경 포인트클라우드 (지형 + Rock)
6. pointcloud.npy: 전체 환경 포인트클라우드 (numpy 배열)

사용법:
    시뮬레이션 시작 시 자동 저장
    또는 SimulationRunner에서 save_dem() 호출
"""

import numpy as np
import json
import os
from datetime import datetime
from pxr import Usd, UsdGeom, Gf


def _is_prim_visible(prim) -> bool:
    """Check if a USD prim is visible (not hidden via visibility attribute)."""
    vis_attr = prim.GetAttribute("visibility")
    if vis_attr and vis_attr.IsAuthored():
        return vis_attr.Get() != "invisible"
    return True


def get_mesh_world_points(prim) -> np.ndarray:
    """
    Mesh prim의 모든 정점을 월드 좌표로 변환하여 반환합니다.
    
    Args:
        prim: USD Mesh prim
        
    Returns:
        np.ndarray: (N, 3) 월드 좌표 배열
    """
    # Mesh 찾기 (prim 자체 또는 자식에서)
    mesh = None
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
    else:
        # 자식에서 Mesh 찾기
        for child in prim.GetAllChildren():
            if child.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(child)
                break
            # 더 깊이 탐색
            for grandchild in child.GetAllChildren():
                if grandchild.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(grandchild)
                    break
            if mesh:
                break
    
    if not mesh:
        return np.array([])
    
    # 로컬 좌표의 정점들
    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        return np.array([])
    
    local_points = points_attr.Get()
    if local_points is None or len(local_points) == 0:
        return np.array([])
    
    # 월드 변환 행렬 가져오기
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    
    # 각 정점을 월드 좌표로 변환
    world_points = []
    for pt in local_points:
        world_pt = world_transform.Transform(Gf.Vec3d(pt[0], pt[1], pt[2]))
        world_points.append([world_pt[0], world_pt[1], world_pt[2]])
    
    return np.array(world_points)


def get_rock_heights_from_geometry(stage, terrain_manager) -> np.ndarray:
    """
    Stage에서 모든 Rock의 실제 Mesh geometry를 읽어와 DEM에 반영합니다.
    Ground Truth 높이맵을 생성합니다.
    
    Args:
        stage: USD Stage
        terrain_manager: TerrainManager 인스턴스
        
    Returns:
        np.ndarray: Rock이 포함된 높이 맵 (Ground Truth)
    """
    rock_heightmap = np.zeros_like(terrain_manager.current_dem)
    
    rocks_prim = stage.GetPrimAtPath("/World/Rocks")
    if not rocks_prim or not rocks_prim.IsValid():
        print("[DEM Export] No rocks found in scene")
        return rock_heightmap
    
    resolution = terrain_manager.cfg.resolution
    x_offset = terrain_manager.x_offset
    y_offset = terrain_manager.y_offset
    grid_width = terrain_manager.grid_width
    grid_height = terrain_manager.grid_height
    
    rock_count = 0
    total_points = 0
    
    skipped_invisible = 0
    for prim in rocks_prim.GetChildren():
        if not prim.IsValid():
            continue
        
        # Skip invisible rocks (e.g. hidden in scene variants like s3)
        if not _is_prim_visible(prim):
            skipped_invisible += 1
            continue
        
        # Mesh의 실제 정점들을 월드 좌표로 가져오기
        world_points = get_mesh_world_points(prim)
        
        if len(world_points) == 0:
            continue
        
        # 각 정점을 DEM 그리드에 매핑
        for pt in world_points:
            x, y, z = pt
            
            # 그리드 인덱스로 변환
            grid_x = int((x - x_offset) / resolution)
            grid_y = int((y - y_offset) / resolution)
            
            # 범위 체크
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                # 지형 높이 위의 Rock 높이만 기록 (최대값 유지)
                terrain_z = terrain_manager.current_dem[grid_y, grid_x]
                if z > terrain_z:
                    rock_height_above_terrain = z - terrain_z
                    rock_heightmap[grid_y, grid_x] = max(
                        rock_heightmap[grid_y, grid_x], 
                        rock_height_above_terrain
                    )
        
        rock_count += 1
        total_points += len(world_points)
    
    if skipped_invisible > 0:
        print(f"[DEM Export] Skipped {skipped_invisible} invisible rocks")
    print(f"[DEM Export] Processed {rock_count} visible rocks ({total_points} mesh vertices)")
    print(f"[DEM Export] Rock heightmap max: {rock_heightmap.max():.3f}m")
    
    return rock_heightmap


def generate_terrain_pointcloud(terrain_manager, subsample: int = 1) -> np.ndarray:
    """
    지형 DEM에서 포인트클라우드를 생성합니다.
    
    [핵심] USD 메시와 동일한 좌표계를 사용합니다:
    - 메시는 np.flip(dem, 0)된 DEM을 사용
    - 여기서도 동일하게 flip 적용
    
    Args:
        terrain_manager: TerrainManager 인스턴스
        subsample: 서브샘플링 배율 (1=전체, 2=1/4, 4=1/16 점)
        
    Returns:
        np.ndarray: (N, 3) 포인트클라우드 [x, y, z]
    """
    dem = terrain_manager.current_dem
    resolution = terrain_manager.cfg.resolution
    x_offset = terrain_manager.x_offset
    y_offset = terrain_manager.y_offset
    
    height, width = dem.shape
    
    # 서브샘플링 적용
    subsample = max(1, int(subsample))
    
    # [핵심 수정] USD 메시와 동일하게 Y축 flip 적용
    # terrain_manager._create_mesh_from_dem에서 flipped_dem = np.flip(dem, 0)을 사용함
    flipped_dem = np.flip(dem, 0)
    
    # 그리드 생성 (subsample 간격으로)
    x = np.arange(0, width, subsample) * resolution + x_offset
    y = np.arange(0, height, subsample) * resolution + y_offset
    
    X, Y = np.meshgrid(x, y)
    Z = flipped_dem[::subsample, ::subsample]
    
    # (N, 3) 형태로 변환
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    return points


def get_all_rock_points(stage, terrain_manager, max_points_per_rock: int = None) -> np.ndarray:
    """
    Stage에서 모든 Rock의 mesh 정점들을 수집합니다.
    
    Args:
        stage: USD Stage
        terrain_manager: TerrainManager 인스턴스
        max_points_per_rock: Rock당 최대 점 개수 (None=전체)
        
    Returns:
        np.ndarray: (N, 3) Rock 포인트클라우드
    """
    all_points = []
    
    rocks_prim = stage.GetPrimAtPath("/World/Rocks")
    if not rocks_prim or not rocks_prim.IsValid():
        return np.array([]).reshape(0, 3)
    
    for prim in rocks_prim.GetChildren():
        if not prim.IsValid():
            continue
        
        # Skip invisible rocks (e.g. hidden in scene variants like s3)
        if not _is_prim_visible(prim):
            continue
        
        world_points = get_mesh_world_points(prim)
        if len(world_points) > 0:
            # 다운샘플링 적용
            if max_points_per_rock is not None and len(world_points) > max_points_per_rock:
                indices = np.random.choice(len(world_points), max_points_per_rock, replace=False)
                world_points = world_points[indices]
            all_points.append(world_points)
    
    if all_points:
        return np.vstack(all_points)
    return np.array([]).reshape(0, 3)


def save_pointcloud_ply(points: np.ndarray, filepath: str, colors: np.ndarray = None):
    """
    포인트클라우드를 PLY 형식으로 저장합니다.
    
    Args:
        points: (N, 3) 포인트 배열 [x, y, z]
        filepath: 저장 경로
        colors: (N, 3) 색상 배열 [r, g, b] (0-255), optional
    """
    n_points = len(points)
    
    with open(filepath, 'w') as f:
        # PLY 헤더
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # 포인트 데이터
        for i in range(n_points):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i].astype(int)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"[DEM Export] Saved PLY pointcloud: {filepath} ({n_points} points)")


# 저장 디렉토리 목록 (여러 위치에 동시 저장)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIRS = [
    os.path.join(_PROJECT_ROOT, "data", "dem_exports"),
    "/shared_data/dem"
]


def save_simulation_dem(terrain_manager, stage, output_dirs: list = None, prefix: str = "sim",
                        pointcloud_subsample: int = 1, max_rock_points_per_rock: int = None,
                        robot_start_position: tuple = None):
    """
    시뮬레이션 환경의 전체 Ground Truth DEM을 저장합니다.
    latest 파일만 저장하여 덮어씁니다 (타임스탬프 파일 생성 안함).
    
    좌표계 규칙:
    - robot_start_position이 주어지면 map 좌표계 (로봇 시작 위치가 원점)
    - robot_start_position이 None이면 시뮬레이션 월드 좌표계 (지형 중심이 원점)
    - DEM[row, col] = DEM[y_index, x_index]
    - map 좌표: (x, y, z) = (world_x - robot_x, world_y - robot_y, world_z - robot_z)
    
    Args:
        terrain_manager: TerrainManager 인스턴스
        stage: USD Stage
        output_dirs: 저장 디렉토리 목록 (None이면 기본 위치들)
        prefix: 파일명 접두사
        pointcloud_subsample: 지형 포인트클라우드 서브샘플링 배율
                              1=전체, 2=1/4 점, 4=1/16 점, 10=1/100 점
        max_rock_points_per_rock: Rock당 최대 점 개수 (None=전체)
        robot_start_position: 로봇 시작 위치 (x, y, z) - map 좌표계 원점
                              None이면 시뮬레이션 좌표계 그대로 저장
        
    Returns:
        dict: 저장된 파일 경로들 (첫 번째 디렉토리 기준)
    """
    if output_dirs is None:
        output_dirs = OUTPUT_DIRS
    
    # 모든 출력 디렉토리 생성
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
    
    # 첫 번째 디렉토리를 기본으로 사용 (반환값용)
    primary_output_dir = output_dirs[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 기본 지형 DEM (deformation 포함된 현재 상태) - 원본 좌표계 유지
    terrain_dem = terrain_manager.current_dem.copy()
    print(f"[DEM Export] Terrain DEM - Shape: {terrain_dem.shape}, Range: [{terrain_dem.min():.3f}, {terrain_dem.max():.3f}]")
    
    # 2. Rock 높이 맵 (실제 Mesh geometry 기반 Ground Truth) - 원본 좌표계 유지
    rock_heightmap = get_rock_heights_from_geometry(stage, terrain_manager)
    
    # 3. 결합 DEM (지형 + Rock) - Ground Truth
    combined_dem = terrain_dem + rock_heightmap
    print(f"[DEM Export] Combined DEM - Shape: {combined_dem.shape}, Range: [{combined_dem.min():.3f}, {combined_dem.max():.3f}]")
    
    # 4. 포인트클라우드 생성 (해상도 조절 가능)
    print(f"[DEM Export] Generating pointcloud (subsample={pointcloud_subsample})...")
    terrain_points = generate_terrain_pointcloud(terrain_manager, subsample=pointcloud_subsample)
    rock_points = get_all_rock_points(stage, terrain_manager, max_points_per_rock=max_rock_points_per_rock)
    
    # [수정] generate_terrain_pointcloud에서 이미 flipped DEM을 사용하므로
    # 추가적인 Y축 반전 불필요 (이전: terrain_points[:, 1] = -terrain_points[:, 1])
    
    if len(rock_points) > 0:
        all_points = np.vstack([terrain_points, rock_points])
    else:
        all_points = terrain_points
    
    print(f"  Total points: {len(all_points)} (terrain: {len(terrain_points)}, rocks: {len(rock_points)})")
    
    # 5. map 좌표계로 변환 (로봇 시작 위치가 원점)
    coordinate_frame = "world"
    map_origin = [0.0, 0.0, 0.0]
    if robot_start_position is not None:
        # 로봇 시작 위치를 원점으로 변환
        robot_x, robot_y, robot_z = robot_start_position[:3]
        all_points[:, 0] -= robot_x
        all_points[:, 1] -= robot_y
        all_points[:, 2] -= robot_z
        coordinate_frame = "map"
        map_origin = [float(robot_x), float(robot_y), float(robot_z)]
        print(f"[DEM Export] Converted to map frame (origin: robot start position [{robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f}])")
    
    # 6. 색상 생성 (지형: 갈색, Rock: 회색)
    terrain_colors = np.tile([139, 119, 101], (len(terrain_points), 1))
    if len(rock_points) > 0:
        rock_colors = np.tile([128, 128, 128], (len(rock_points), 1))
        all_colors = np.vstack([terrain_colors, rock_colors])
    else:
        all_colors = terrain_colors
    
    # 7. 메타데이터 생성
    metadata = {
        "timestamp": timestamp,
        "type": "ground_truth",
        "resolution": terrain_manager.cfg.resolution,
        "x_size": terrain_manager.cfg.x_size,
        "y_size": terrain_manager.cfg.y_size,
        "grid_width": terrain_manager.grid_width,
        "grid_height": terrain_manager.grid_height,
        "x_offset": terrain_manager.x_offset,
        "y_offset": terrain_manager.y_offset,
        "z_scale": terrain_manager.cfg.z_scale,
        "num_rocks": len(terrain_manager.rock_paths),
        "terrain_dem_range": [float(terrain_dem.min()), float(terrain_dem.max())],
        "rock_heightmap_range": [float(rock_heightmap.min()), float(rock_heightmap.max())],
        "combined_dem_range": [float(combined_dem.min()), float(combined_dem.max())],
        "pointcloud": {
            "total_points": len(all_points),
            "terrain_points": len(terrain_points),
            "rock_points": len(rock_points),
            "subsample_factor": pointcloud_subsample,
            "max_rock_points_per_rock": max_rock_points_per_rock,
            "coordinate_frame": coordinate_frame,
            "map_origin_in_world": map_origin
        },
        "files": {
            "terrain_dem": f"{prefix}_terrain_dem_latest.npy",
            "rock_heightmap": f"{prefix}_rock_heightmap_latest.npy",
            "combined_dem": f"{prefix}_combined_dem_latest.npy",
            "pointcloud_ply": f"{prefix}_pointcloud_latest.ply",
            "pointcloud_npy": f"{prefix}_pointcloud_latest.npy"
        },
        "coordinate_system": {
            "frame": coordinate_frame,
            "origin": "robot start position" if robot_start_position else "center of terrain",
            "map_origin_in_world": map_origin,
            "x_range": [terrain_manager.x_offset - map_origin[0], terrain_manager.x_offset + terrain_manager.cfg.x_size - map_origin[0]],
            "y_range": [terrain_manager.y_offset - map_origin[1], terrain_manager.y_offset + terrain_manager.cfg.y_size - map_origin[1]],
            "units": "meters"
        },
        "output_directories": output_dirs
    }
    
    # 8. 모든 디렉토리에 latest 파일만 저장 (덮어쓰기)
    for output_dir in output_dirs:
        # DEM 파일들
        np.save(os.path.join(output_dir, f"{prefix}_terrain_dem_latest.npy"), terrain_dem)
        np.save(os.path.join(output_dir, f"{prefix}_rock_heightmap_latest.npy"), rock_heightmap)
        np.save(os.path.join(output_dir, f"{prefix}_combined_dem_latest.npy"), combined_dem)
        
        # Pointcloud 파일들
        np.save(os.path.join(output_dir, f"{prefix}_pointcloud_latest.npy"), all_points)
        save_pointcloud_ply(all_points, os.path.join(output_dir, f"{prefix}_pointcloud_latest.ply"), all_colors)
        
        # 메타데이터
        with open(os.path.join(output_dir, f"{prefix}_dem_metadata_latest.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"[DEM Export] Saved to {len(output_dirs)} directories (latest files only, overwritten)")
    for output_dir in output_dirs:
        print(f"  - {output_dir}")
    
    # Return primary directory paths
    return {
        "terrain_dem": os.path.join(primary_output_dir, f"{prefix}_terrain_dem_latest.npy"),
        "rock_heightmap": os.path.join(primary_output_dir, f"{prefix}_rock_heightmap_latest.npy"),
        "combined_dem": os.path.join(primary_output_dir, f"{prefix}_combined_dem_latest.npy"),
        "pointcloud_ply": os.path.join(primary_output_dir, f"{prefix}_pointcloud_latest.ply"),
        "pointcloud_npy": os.path.join(primary_output_dir, f"{prefix}_pointcloud_latest.npy"),
        "metadata": os.path.join(primary_output_dir, f"{prefix}_dem_metadata_latest.json"),
        "output_directories": output_dirs
    }


def load_dem_with_metadata(dem_path: str, metadata_path: str = None):
    """
    저장된 DEM과 메타데이터를 로드합니다.
    
    Args:
        dem_path: DEM .npy 파일 경로
        metadata_path: 메타데이터 .json 파일 경로 (None이면 자동 탐색)
        
    Returns:
        tuple: (dem_array, metadata_dict)
    """
    dem = np.load(dem_path)
    
    if metadata_path is None:
        # 같은 디렉토리에서 메타데이터 찾기
        dir_path = os.path.dirname(dem_path)
        possible_metadata = [
            dem_path.replace('.npy', '_metadata.json'),
            dem_path.replace('combined_dem', 'dem_metadata'),
            dem_path.replace('terrain_dem', 'dem_metadata'),
            os.path.join(dir_path, 'sim_dem_metadata_latest.json')
        ]
        
        for mp in possible_metadata:
            if os.path.exists(mp):
                metadata_path = mp
                break
    
    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return dem, metadata


def world_to_grid(x, y, metadata):
    """월드 좌표를 그리드 인덱스로 변환"""
    x_offset = metadata.get('x_offset', 0)
    y_offset = metadata.get('y_offset', 0)
    resolution = metadata.get('resolution', 0.05)
    
    grid_x = int((x - x_offset) / resolution)
    grid_y = int((y - y_offset) / resolution)
    
    return grid_x, grid_y


def grid_to_world(grid_x, grid_y, metadata):
    """그리드 인덱스를 월드 좌표로 변환"""
    x_offset = metadata.get('x_offset', 0)
    y_offset = metadata.get('y_offset', 0)
    resolution = metadata.get('resolution', 0.05)
    
    x = grid_x * resolution + x_offset
    y = grid_y * resolution + y_offset
    
    return x, y


# 테스트용 메인
if __name__ == "__main__":
    print("DEM Export Utility (Ground Truth)")
    print("=" * 50)
    print("This module should be imported and used with TerrainManager.")
    print("\nUsage in simulation:")
    print("  from scripts.save_dem import save_simulation_dem")
    print("  save_simulation_dem(terrain_manager, stage)")
    print("\nUsage to load:")
    print("  from scripts.save_dem import load_dem_with_metadata")
    print("  dem, metadata = load_dem_with_metadata('path/to/dem.npy')")
