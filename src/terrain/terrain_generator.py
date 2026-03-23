# Lproject_sim/src/terrain/terrain_generator.py
import numpy as np
import pickle
import cv2
try:
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import rotate
except ImportError:
    print("Warning: scipy not found. Terrain generation features will be limited.")
    CubicSpline = None
    rotate = None
from typing import List, Tuple, Optional
import dataclasses
import os

@dataclasses.dataclass
class TerrainConfig:
    type: str = "procedural"
    x_size: float = 100.0
    y_size: float = 100.0
    resolution: float = 0.05
    src_resolution: float = 5.0
    min_elevation: float = -0.5
    max_elevation: float = 0.5
    crop_center_meters: Tuple[float, float] = (0.0, 0.0)
    z_scale: float = 1.0
    seed: int = 42
    dem_path: str = ""
    mask_path: str = ""

    # --- [핵심 수정] 아래 두 줄을 추가합니다. ---
    rock_assets_dir: str = ""
    num_rocks: int = 0
    # ---------------------------------------------
    # Hybrid Mode Parameters
    detail_strength: float = 0.1
    detail_scale: float = 10.0

@dataclasses.dataclass
class CraterConfig:
    profiles_path: str
    min_xy_ratio: float = 0.85
    max_xy_ratio: float = 1.0
    z_scale: float = 1.0
    resolution: float = 0.1
    pad_size: int = 500
    seed: int = 42

@dataclasses.dataclass
class CraterData:
    deformation_spline: Optional[CubicSpline] = None
    marks_spline: Optional[CubicSpline] = None
    marks_intensity: float = 0
    size: int = 0
    crater_profile_id: int = 0
    xy_deformation_factor: Tuple[float, float] = (0, 0)
    rotation: float = 0
    coord: Tuple[int, int] = (0, 0)

class CraterGenerator:
    def __init__(self, cfg: CraterConfig):
        self.cfg = cfg
        self._profiles = None
        self._rng = np.random.default_rng(cfg.seed)
        self.load_profiles()

    def load_profiles(self):
        with open(self.cfg.profiles_path, "rb") as handle:
            self._profiles = pickle.load(handle)

    def sat_gaussian(self, x: np.ndarray, mu1: float, mu2: float, std: float) -> np.ndarray:
        shape = x.shape
        x = x.flatten()
        x[x < mu1] = np.exp(-0.5 * ((x[x < mu1] - mu1) / std) ** 2)
        x[x > mu2] = np.exp(-0.5 * ((x[x > mu2] - mu2) / std) ** 2)
        x[(x >= mu1) & (x <= mu2)] = 1.0
        x = x / (std * np.sqrt(2 * np.pi))
        return x.reshape(shape)

    def centered_distance_matrix(self, crater_data: CraterData) -> np.ndarray:
        m = np.zeros([crater_data.size, crater_data.size])
        x, y = np.meshgrid(np.linspace(-1, 1, crater_data.size), np.linspace(-1, 1, crater_data.size))
        theta = np.arctan2(y, x)
        fac = crater_data.deformation_spline(theta / (2 * np.pi) + 0.5)
        marks = crater_data.marks_spline(theta / (2 * np.pi) + 0.5) * crater_data.size / 2 * crater_data.marks_intensity

        x, y = np.meshgrid(range(crater_data.size), range(crater_data.size))
        m = np.sqrt(
            ((x - (crater_data.size / 2) + 1) * 1 / crater_data.xy_deformation_factor[0]) ** 2
            + ((y - (crater_data.size / 2) + 1) * 1 / crater_data.xy_deformation_factor[1]) ** 2
        )
        m = m * fac
        sat = self.sat_gaussian(m, 0.15 * crater_data.size / 2, 0.45 * crater_data.size / 2, 0.05 * crater_data.size / 2)
        sat = (sat - sat.min()) / (sat.max() - sat.min())
        m = m + marks * sat
        m = rotate(m, crater_data.rotation, reshape=False, cval=crater_data.size / 2)
        m[m > crater_data.size / 2] = crater_data.size / 2
        return m

    def apply_profile(self, distance: np.ndarray, crater_data: CraterData) -> np.ndarray:
        return self._profiles[crater_data.crater_profile_id](2 * distance / crater_data.size)

    def randomize_crater_parameters(self, size: int) -> CraterData:
        crater_data = CraterData()
        size = size + ((size % 2) == 0)
        crater_data.size = size

        deformation_profile = self._rng.uniform(0.95, 1, 9)
        deformation_profile = np.concatenate([deformation_profile, [deformation_profile[0]]], axis=0)
        tmp_x = np.linspace(0, 1, deformation_profile.shape[0])
        crater_data.deformation_spline = CubicSpline(tmp_x, deformation_profile, bc_type=((1, 0.0), (1, 0.0)))

        marks_profile = self._rng.uniform(0.0, 0.01, 45)
        marks_profile = np.concatenate([marks_profile, [marks_profile[0]]], axis=0)
        tmp_x = np.linspace(0, 1, marks_profile.shape[0])
        crater_data.marks_spline = CubicSpline(tmp_x, marks_profile, bc_type=((1, 0.0), (1, 0.0)))
        crater_data.marks_intensity = self._rng.uniform(0, 1)

        crater_data.xy_deformation_factor = (self._rng.uniform(self.cfg.min_xy_ratio, self.cfg.max_xy_ratio), 1.0)
        crater_data.rotation = int(self._rng.uniform(0, 360))
        crater_data.crater_profile_id = self._rng.integers(0, len(self._profiles), 1)[0]
        return crater_data

    def generate_crater(self, size: int) -> Tuple[np.ndarray, CraterData]:
        crater_data = self.randomize_crater_parameters(size)
        distance = self.centered_distance_matrix(crater_data)
        crater = self.apply_profile(distance, crater_data) * crater_data.size / 2.0 * self.cfg.z_scale * self.cfg.resolution
        return crater, crater_data

class TerrainGenerator:
    def __init__(self, cfg: TerrainConfig, crater_cfg: Optional[CraterConfig] = None):
        self.cfg = cfg
        self.crater_cfg = crater_cfg
        self._rng = np.random.default_rng(cfg.seed)
        self.crater_gen = CraterGenerator(crater_cfg) if crater_cfg else None

    def generate(self) -> np.ndarray:
        if self.cfg.type == "real_data":
            return self.load_from_file()
        elif self.cfg.type == "hybrid":
            return self.generate_hybrid()
        else:
            return self.generate_procedural()


# #평지로 사용하고 싶을때 이용##
#         x_pixels = int(self.cfg.x_size / self.cfg.resolution)
#         y_pixels = int(self.cfg.y_size / self.cfg.resolution)
        
#         # 2. 콘솔에 평지를 생성한다고 알립니다.
#         print(f"--- DEBUG: Generating a completely flat terrain of size {y_pixels}x{x_pixels} pixels. ---")
        
#         # 3. 모든 값이 0인 numpy 배열을 생성하여 반환합니다.
#         #    shape는 (y_pixels, x_pixels) 순서가 맞습니다. (행, 열)
#         return np.zeros((y_pixels, x_pixels), dtype=np.float32)

    def generate_hybrid(self) -> np.ndarray:
        """
        Hybrid Mode: Real DEM + High Frequency Procedural Noise.
        Simulates OmniLRS 'LargeScale' generation logic.
        """
        print(f"Generating Hybrid Terrain (Base: {self.cfg.dem_path})")
        
        # 1. Load Base DEM
        base_dem = self.load_from_file()
        # Note: load_from_file already handles loading, resizing to (y,x), and scaling by z_scale.
        
        # 2. Generate Detail Noise (High Frequency)
        x_pixels = base_dem.shape[1]
        y_pixels = base_dem.shape[0]
        
        noise_layer = np.zeros((y_pixels, x_pixels))
        
        octaves = 4 # Fewer octaves for just detail
        persistence = 0.5
        lacunarity = 2.0
        scale = self.cfg.detail_scale # Start with higher scale (smaller features)
        amplitude = self.cfg.detail_strength # Controls how rough the details are
        
        res_correction = 0.1 / self.cfg.resolution 
        # e.g. if res=0.05, correction=2.0. We want sub_x = 100 (same as res 0.1).
        # x_pixels is 1000. 1000 / (scale * 2.0) = 500 / scale.
        # if scale=5.0, sub_x=100. Correct.
        
        for i in range(octaves):
            current_scale = scale * res_correction
            sub_y = max(1, int(y_pixels / current_scale))
            sub_x = max(1, int(x_pixels / current_scale))
            
            # Generate random noise
            layer = self._rng.uniform(-1.0, 1.0, (sub_y, sub_x))
            
            if sub_x > 5 and sub_y > 5:
                # Gaussian blur to smooth the noise grid
                layer = cv2.GaussianBlur(layer, (3, 3), 0)
                
            layer = cv2.resize(layer, (x_pixels, y_pixels), interpolation=cv2.INTER_CUBIC)
            
            noise_layer += layer * amplitude
            
            amplitude *= persistence
            scale *= lacunarity # Note: We update 'scale' for next octave, correction applies each time
            
        # 3. Blend
        # Base DEM (already z-scaled) + Noise (scaled by amplitude)
        # Note: We do NOT re-normalize here because that would squash the DEM structure.
        # We just add surface roughness.
        
        final_dem = base_dem + noise_layer
        
        return final_dem


    def load_from_file(self) -> np.ndarray:
        """
        Load DEM from .npy file.
        [Optimized] Crops the required region from the low-res source DEM FIRST,
        then resamples only the small cropped area to the target resolution.
        This prevents Out Of Memory errors.
        """
        if not self.cfg.dem_path or not os.path.exists(self.cfg.dem_path):
            print(f"Error: DEM path not found: {self.cfg.dem_path}. Falling back to procedural.")
            return self.generate_procedural()

        try:
            print(f"Loading Real Lunar DEM from: {self.cfg.dem_path}")
            dem = np.load(self.cfg.dem_path)
            src_h, src_w = dem.shape
            

            # 1. 원본 DEM에서 잘라낼 영역의 '픽셀 크기'를 계산합니다.
            crop_w_in_src_pixels = int(self.cfg.x_size / self.cfg.src_resolution)
            crop_h_in_src_pixels = int(self.cfg.y_size / self.cfg.src_resolution)

            # 2. [수정] 원본 DEM의 '픽셀 중심'을 계산합니다.
            src_center_x_px = src_w // 2
            src_center_y_px = src_h // 2
            
            # 3. [추가] YAML에서 지정한 '미터 단위 오프셋'을 '픽셀 단위 오프셋'으로 변환합니다.
            #    crop_center_meters[0]은 X축(동쪽), [1]은 Y축(북쪽) 오프셋을 의미한다고 가정합니다.
            offset_x_m, offset_y_m = self.cfg.crop_center_meters
            offset_x_px = int(offset_x_m / self.cfg.src_resolution)
            offset_y_px = int(offset_y_m / self.cfg.src_resolution)
            
            # 4. [수정] 잘라낼 영역의 최종 '픽셀 중심'을 계산합니다.
            #    (Numpy 배열 인덱싱에서 Y축은 아래로 갈수록 증가하므로, Y 오프셋에 -를 붙입니다)
            crop_center_x_px = src_center_x_px + offset_x_px
            crop_center_y_px = src_center_y_px - offset_y_px # Y축 방향 반전
            
            # 5. [수정] 잘라낼 영역의 시작/끝 픽셀 인덱스를 계산합니다.
            start_x = crop_center_x_px - (crop_w_in_src_pixels // 2)
            end_x = start_x + crop_w_in_src_pixels
            start_y = crop_center_y_px - (crop_h_in_src_pixels // 2)
            end_y = start_y + crop_h_in_src_pixels

            # 6. [추가] 계산된 영역이 원본 DEM의 경계를 벗어나지 않도록 클램핑합니다.
            if start_x < 0 or end_x > src_w or start_y < 0 or end_y > src_h:
                print(f"Warning: Requested crop region is out of source DEM bounds. Clamping to valid area.")
                start_x = max(0, start_x)
                end_x = min(src_w, end_x)
                start_y = max(0, start_y)
                end_y = min(src_h, end_y)

            # 7. [Crop] 원본 DEM에서 최종 계산된 영역을 잘라냅니다.
            print(f"Cropping source DEM region from Y:[{start_y}:{end_y}], X:[{start_x}:{end_x}]")
            cropped_dem = dem[start_y:end_y, start_x:end_x]

            # 3. [Resample] 이제 작아진 'cropped_dem'만 최종 해상도로 리샘플링합니다.
            target_pixels_x = int(self.cfg.x_size / self.cfg.resolution)
            target_pixels_y = int(self.cfg.y_size / self.cfg.resolution)
            
            print(f"Resampling cropped DEM to target resolution ({target_pixels_y}x{target_pixels_x}) pixels.")
            final_dem = cv2.resize(cropped_dem, (target_pixels_x, target_pixels_y), interpolation=cv2.INTER_CUBIC)

            # --------------------------------------------------------

            # 4. 정규화 및 Z-스케일 적용
            # 중심점(로봇 시작 위치)을 Z=0으로 맞추기 위해 중앙값 기준 정규화
            min_val = final_dem.min()
            max_val = final_dem.max()
            if max_val - min_val > 1e-6:
                # -0.5 ~ +0.5 범위로 정규화 (중심=0)
                final_dem = (final_dem - min_val) / (max_val - min_val) - 0.5
            
            final_dem *= self.cfg.z_scale
            
            return final_dem

        except Exception as e:
            print(f"Failed to load DEM: {e}")
            return self.generate_procedural()

# ... (이하 코드는 그대로)
    def generate_procedural(self) -> np.ndarray:
        """
        Procedural generation with adjusted parameters for smoother, drivable terrain.
        - Reduced persistence to weaken high-frequency noise.
        - Increased initial scale for gentler slopes.
        - Added a final Gaussian blur to smooth out sharp edges.
        """
        x_pixels = int(self.cfg.x_size / self.cfg.resolution)
        y_pixels = int(self.cfg.y_size / self.cfg.resolution)
        
        dem = np.zeros((y_pixels, x_pixels), dtype=np.float32)
        
        # --- [수정 1] 파라미터 조정으로 근본적인 노이즈 감소 ---
        octaves = 4
        persistence = 0.4      # 값을 낮춰 고주파 노이즈의 영향력을 줄임
        lacunarity = 2.0
        scale = 20.0           # 초기 스케일을 키워 전체적으로 완만한 지형 생성
        amplitude = 0.0        # 기본 진폭 (활성화)
        


        for i in range(octaves):
            sub_y = max(1, int(y_pixels / scale))
            sub_x = max(1, int(x_pixels / scale))
            
            noise_layer = self._rng.uniform(-1.0, 1.0, (sub_y, sub_x))
            if sub_x > 5 and sub_y > 5:
                # 저해상도 노이즈를 좀 더 부드럽게 만듦
                noise_layer = cv2.GaussianBlur(noise_layer, (5, 5), 0)
            
            noise_layer = cv2.resize(noise_layer, (x_pixels, y_pixels), interpolation=cv2.INTER_CUBIC)
            
            dem += noise_layer * amplitude
            
            amplitude *= persistence
            scale *= lacunarity

        # --- [수정 2] 최종 지형을 부드럽게 만드는 후처리 과정 추가 ---
        # 모든 노이즈가 합쳐진 후, 가장 날카로운 부분을 한번 더 부드럽게 다듬습니다.
        # 마치 사포질을 하는 것과 같은 효과입니다.
        dem = cv2.GaussianBlur(dem, (5, 5), 0)

        # 크레이터 추가 (정규화 전)
        if self.crater_gen:
            num_craters = 15
            for _ in range(num_craters):
                radius_m = self._rng.uniform(2.0, 20.0)
                radius_px = int(radius_m / self.cfg.resolution)

                if x_pixels <= radius_px * 2 or y_pixels <= radius_px * 2:
                    continue

                cx = self._rng.integers(radius_px, x_pixels - radius_px)
                cy = self._rng.integers(radius_px, y_pixels - radius_px)
                
                crater, _ = self.crater_gen.generate_crater(radius_px * 2)
                
                x_start = max(0, cx - crater.shape[1] // 2)
                x_end = min(x_pixels, x_start + crater.shape[1])
                y_start = max(0, cy - crater.shape[0] // 2)
                y_end = min(y_pixels, y_start + crater.shape[0])
                
                c_x_start = max(0, crater.shape[1] // 2 - cx)
                c_x_end = c_x_start + (x_end - x_start)
                c_y_start = max(0, crater.shape[0] // 2 - cy)
                c_y_end = c_y_start + (y_end - y_start)

                dem_slice = dem[y_start:y_end, x_start:x_end]
                crater_slice = crater[c_y_start:c_y_end, c_x_start:c_x_end]

                if dem_slice.shape == crater_slice.shape:
                    dem[y_start:y_end, x_start:x_end] += crater_slice
                else:
                    print(f"Warning: Mismatched slice shapes. DEM: {dem_slice.shape}, Crater: {crater_slice.shape}. Skipping.")

        # 최종 정규화
        # 중심점(로봇 시작 위치)을 Z=0으로 맞추기 위해 중앙값 기준 정규화
        min_val = dem.min()
        max_val = dem.max()
        print(f"[TerrainGenerator] Before normalization: min={min_val:.3f}, max={max_val:.3f}")
        
        if max_val - min_val > 1e-6:
            # -0.5 ~ +0.5 범위로 정규화 (중심=0)
            dem = (dem - min_val) / (max_val - min_val) - 0.5
        
        dem *= self.cfg.z_scale
        
        # 중심점(로봇 시작 위치) 높이 확인
        center_y, center_x = dem.shape[0] // 2, dem.shape[1] // 2
        center_height = dem[center_y, center_x]
        print(f"[TerrainGenerator] After normalization: min={dem.min():.3f}, max={dem.max():.3f}")
        print(f"[TerrainGenerator] Center point height (robot start): {center_height:.3f}m")

        return dem
