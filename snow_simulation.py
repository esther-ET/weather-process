"""
snow_simulation.py - 雪天点云模拟（支持随机参数）
输出: 4维 (x, y, z, intensity)
"""

import numpy as np
import os
import warnings
import importlib
import sys
import re
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import (load_kitti_points, save_kitti_points,
                   get_lidar_distance, sample_snowfall_rate)


class SnowSimulation:
    _LISA_IMPORT_CACHE = {}
    _LIDAR_SNOW_IMPORT_CACHE = {}
    _PARTICLE_PROFILE_CACHE = {}

    def __init__(self, snowfall_rate=2.5, terminal_velocity=1.0, snow_density=0.1,
                 backend='auto', lidar_snow_sim_path=None,
                 lisa_path=None,
                 particle_file_prefix=None, beam_divergence=0.35,
                 only_camera_fov=False, noise_floor=0.7, root_path=None,
                 lidar_parallel_backend='thread',
                 particle_model='gunn',
                 rainfall_rate_levels=None,
                 rainfall_level_sampling='nearest',
                 lidar_intensity_mode='kitti_01',
                 channel_mode='infer', num_lasers=64,
                 fov_down_deg=-24.8, fov_up_deg=2.0):
        """snowfall_rate: mm/h 水当量, 建议范围 [0.5, 10]"""
        self.snowfall_rate = snowfall_rate
        self.terminal_velocity = terminal_velocity
        self.snow_density = snow_density
        self.backend = backend
        self.lidar_snow_sim_path = lidar_snow_sim_path
        self.lisa_path = lisa_path
        self.particle_file_prefix = particle_file_prefix
        self.beam_divergence = beam_divergence
        self.only_camera_fov = only_camera_fov
        self.noise_floor = noise_floor
        self.root_path = root_path
        self.lidar_parallel_backend = str(lidar_parallel_backend)
        self.particle_model = str(particle_model).lower()
        if self.particle_model not in ('gunn', 'sekhon'):
            raise ValueError(f"particle_model must be 'gunn' or 'sekhon', got {particle_model}")
        if rainfall_rate_levels is None:
            self.rainfall_rate_levels = [2.0, 8.0, 17.0, 34.0, 70.0]
        else:
            self.rainfall_rate_levels = [float(x) for x in rainfall_rate_levels]
        self.rainfall_level_sampling = str(rainfall_level_sampling).lower()
        if self.rainfall_level_sampling not in ('nearest', 'balanced'):
            raise ValueError(
                f"rainfall_level_sampling must be 'nearest' or 'balanced', got {rainfall_level_sampling}"
            )
        self.lidar_intensity_mode = str(lidar_intensity_mode).lower()
        if self.lidar_intensity_mode not in ('kitti_01', 'raw_255', 'auto'):
            raise ValueError(
                f"lidar_intensity_mode must be 'kitti_01', 'raw_255' or 'auto', got {lidar_intensity_mode}"
            )
        self.channel_mode = channel_mode
        self.num_lasers = int(num_lasers)
        self.fov_down_deg = float(fov_down_deg)
        self.fov_up_deg = float(fov_up_deg)
        self.lidar_range = 120.0
        self.d_snow = 2.0 + 0.5 * np.log(1 + self.snowfall_rate)
        self.snow_conc = self._concentration()
        self._snow_module = None
        self._lisa = None
        self._lisa_impl = None
        self.last_particle_file_prefix = None
        self.last_particle_rainfall_rate = None
        self.last_target_rainfall_rate = None
        self.last_target_rainfall_level = None
        self._resolved_backend = self._resolve_backend()

    def set_snowfall_rate(self, snowfall_rate):
        """更新降雪率并刷新与启发式模型相关的派生参数。"""
        self.snowfall_rate = float(snowfall_rate)
        self.d_snow = 2.0 + 0.5 * np.log(1 + self.snowfall_rate)
        self.snow_conc = self._concentration()
        return self

    def _resolve_backend(self):
        if self.backend == 'heuristic':
            return 'heuristic'

        if self.backend in ('auto', 'lidar_snow_sim'):
            module, debug_msg = self._import_lidar_snow_sim()
            if module is not None:
                self._snow_module = module
                return 'lidar_snow_sim'
        else:
            debug_msg = 'lidar_snow_sim skipped by backend setting'

        if self.backend == 'lidar_snow_sim':
            raise ImportError(
                "backend='lidar_snow_sim' but simulation module is not importable. "
                "Set --lidar_snow_sim_path or export LIDAR_SNOW_SIM_PATH "
                "to the directory containing simulation.py. "
                f"{debug_msg}"
            )

        lisa_handle, lisa_debug = self._import_lisa()
        if lisa_handle is not None:
            impl, lisa_cls = lisa_handle
            self._lisa_impl = impl
            if impl == 'legacy':
                self._lisa = lisa_cls(atm_model='snow')
            elif impl == 'python':
                self._lisa = lisa_cls(mode='snow')
            elif impl == 'pylisa':
                self._lisa = lisa_cls(atm_model='snow')
            return 'lisa'

        if self.backend == 'lisa':
            raise ImportError(
                "backend='lisa' but LISA is not importable. "
                "Set --lisa_path or export LISA_PATH to the LISA repository root. "
                f"{lisa_debug}"
            )

        warnings.warn(
            "Physical snow backends not found, falling back to heuristic snow simulation. "
            f"lidar_snow_sim={debug_msg}; lisa={lisa_debug}",
            RuntimeWarning
        )
        return 'heuristic'

    def _import_lisa(self):
        search_paths = []
        tried = []
        errors = []

        if self.lisa_path:
            base = Path(self.lisa_path).expanduser()
            search_paths.append(str(base))
            search_paths.append(str(base / 'python_old'))

        env_lisa_path = os.getenv('LISA_PATH')
        if env_lisa_path:
            base = Path(env_lisa_path).expanduser()
            search_paths.append(str(base))
            search_paths.append(str(base / 'python_old'))

        repo_root = Path(__file__).resolve().parent
        search_paths.extend([
            str(repo_root / 'thirdparty' / 'LISA'),
            str(repo_root / 'thirdparty' / 'LISA' / 'python_old'),
            str(repo_root.parent / 'LiDAR_snow_sim' / 'lib' / 'LISA'),
            str(repo_root.parent / 'LISA'),
            str(Path.home() / 'SWW' / 'code' / 'weather-process' / 'thirdparty' / 'LISA'),
            str(Path.home() / 'SWW' / 'code' / 'weather-process' / 'thirdparty' / 'LISA' / 'python_old'),
            str(Path.home() / 'SWW' / 'code' / 'LiDAR_snow_sim' / 'lib' / 'LISA'),
            str(Path.home() / 'SWW' / 'code' / 'LISA'),
        ])

        for p in search_paths:
            if p and p not in sys.path:
                sys.path.insert(0, p)
            tried.append(p)

        cache_key = tuple(tried)
        if cache_key in self._LISA_IMPORT_CACHE:
            return self._LISA_IMPORT_CACHE[cache_key]

        try:
            module = importlib.import_module('atmos_models')
            cls = getattr(module, 'LISA', None)
            if cls is not None:
                result = (('legacy', cls), f"loaded atmos_models from {module.__file__}")
                self._LISA_IMPORT_CACHE[cache_key] = result
                return result
        except Exception as e:
            errors.append(f"atmos_models: {e}")

        try:
            module = importlib.import_module('lisa')
            cls = getattr(module, 'LISA', None)
            if cls is not None:
                result = (('python', cls), f"loaded lisa from {module.__file__}")
                self._LISA_IMPORT_CACHE[cache_key] = result
                return result
        except Exception as e:
            errors.append(f"lisa: {e}")

        try:
            module = importlib.import_module('pylisa.lisa')
            cls = getattr(module, 'Lisa', None)
            if cls is not None:
                result = (('pylisa', cls), f"loaded pylisa.lisa from {module.__file__}")
                self._LISA_IMPORT_CACHE[cache_key] = result
                return result
        except Exception as e:
            errors.append(f"pylisa.lisa: {e}")

        debug_msg = f"searched_paths={tried}; import_errors={errors}"
        result = (None, debug_msg)
        self._LISA_IMPORT_CACHE[cache_key] = result
        return result

    def _import_lidar_snow_sim(self):
        raw_inputs = []
        tried = []
        errors = []
        added_paths = []
        import_ok = False
        if self.lidar_snow_sim_path:
            raw_inputs.append(self.lidar_snow_sim_path)
        env_path = os.getenv('LIDAR_SNOW_SIM_PATH')
        if env_path:
            raw_inputs.append(env_path)
        repo_root = Path(__file__).resolve().parent
        raw_inputs.extend([
            str(repo_root.parent / 'LiDAR_snow_sim'),
            str(Path.home() / 'SWW' / 'code' / 'LiDAR_snow_sim'),
        ])

        search_paths = []
        for raw in raw_inputs:
            if not raw:
                continue
            p = Path(raw).expanduser()
            if p.name == 'snowfall' and p.parent.name == 'tools':
                repo = p.parent.parent
                snowfall_dir = p
            elif p.name == 'tools':
                repo = p.parent
                snowfall_dir = p / 'snowfall'
            elif (p / 'tools' / 'snowfall').exists():
                repo = p
                snowfall_dir = p / 'tools' / 'snowfall'
            else:
                repo = p
                snowfall_dir = p
            search_paths.extend([
                str(snowfall_dir),
                str(repo),
                str(repo / 'tools' / 'wet_ground'),
                str(repo / 'lib'),
            ])

        # unique preserve order
        seen = set()
        ordered_paths = []
        for p in search_paths:
            if p not in seen:
                seen.add(p)
                ordered_paths.append(p)

        for p in ordered_paths:
            if p and p not in sys.path:
                sys.path.insert(0, p)
                added_paths.append(p)
            tried.append(p)

        cache_key = tuple(tried)
        if cache_key in self._LIDAR_SNOW_IMPORT_CACHE:
            return self._LIDAR_SNOW_IMPORT_CACHE[cache_key]

        # Avoid local module-name collisions (e.g., weather-process/utils.py) while
        # importing LiDAR_snow_sim modules that use top-level imports like
        # `from utils import ...`.
        shadow_names = ('utils', 'planes', 'phy_equations')
        shadow_backup = {}
        for name in shadow_names:
            if name in sys.modules:
                shadow_backup[name] = sys.modules.pop(name)

        try:
            module = importlib.import_module('simulation')
            import_ok = True
            result = (module, f"loaded simulation from {module.__file__}")
            self._LIDAR_SNOW_IMPORT_CACHE[cache_key] = result
            return result
        except Exception as e:
            errors.append(str(e))
            debug_msg = f"searched_paths={tried}; import_errors={errors}"
            result = (None, debug_msg)
            self._LIDAR_SNOW_IMPORT_CACHE[cache_key] = result
            return result
        finally:
            # Restore original modules to keep this project import behavior stable.
            for name in shadow_names:
                if name in shadow_backup:
                    sys.modules[name] = shadow_backup[name]

            # If import failed, clean up paths we injected to avoid polluting sys.path.
            if not import_ok:
                for p in added_paths:
                    try:
                        sys.path.remove(p)
                    except ValueError:
                        pass

    def _concentration(self):
        d_m = self.d_snow * 1e-3
        v = (4 / 3) * np.pi * (d_m / 2) ** 3
        R = self.snowfall_rate * 1e-3 / 3600.0
        return min(R / (self.terminal_velocity * v * self.snow_density), 5000)

    def _extinction(self):
        d_m = self.d_snow * 1e-3
        return 2.0 * np.pi * (d_m / 2) ** 2 * self.snow_conc

    def _attenuate(self, pts):
        dist = get_lidar_distance(pts)
        att = np.exp(-2 * self._extinction() * dist)
        pts[:, 3] *= att
        keep = np.random.uniform(0, 1, len(pts)) < np.clip(att ** 0.5, 0.2, 1.0)
        return pts[keep]

    def _snow_noise(self, n_orig):
        n = int(n_orig * min(0.01 * self.snowfall_rate, 0.2))
        if n == 0:
            return np.empty((0, 4), dtype=np.float32)
        n_near, n_far = int(n * 0.6), n - int(n * 0.6)
        parts = []
        if n_near > 0:
            r = np.clip(np.random.exponential(5.0, n_near), 0.5, 15.0)
            az = np.random.uniform(-np.pi, np.pi, n_near)
            el = np.random.uniform(np.radians(-24.8), np.radians(2.0), n_near)
            parts.append(np.stack([
                r * np.cos(el) * np.cos(az),
                r * np.cos(el) * np.sin(az),
                r * np.sin(el),
                np.random.uniform(0, 0.25, n_near)
            ], axis=1))
        if n_far > 0:
            r = np.random.uniform(15.0, 60.0, n_far)
            az = np.random.uniform(-np.pi, np.pi, n_far)
            el = np.random.uniform(np.radians(-24.8), np.radians(2.0), n_far)
            x = r * np.cos(el) * np.cos(az) + np.random.normal(0, 0.1, n_far)
            y = r * np.cos(el) * np.sin(az) + np.random.normal(0, 0.1, n_far)
            z = r * np.sin(el)
            parts.append(np.stack([x, y, z, np.random.uniform(0, 0.1, n_far)], axis=1))
        return np.vstack(parts).astype(np.float32)

    def _clutter(self):
        if self.snowfall_rate < 2.0:
            return np.empty((0, 4), dtype=np.float32)
        nc = int(self.snowfall_rate * 2)
        npc = int(self.snowfall_rate * 5)
        total_points = nc * npc
        # Generate all cluster centers at once (vectorized)
        cr = np.random.uniform(1, 10, nc)
        caz = np.random.uniform(-np.pi, np.pi, nc)
        cel = np.random.uniform(np.radians(-15), np.radians(2), nc)
        cx = cr * np.cos(cel) * np.cos(caz)
        cy = cr * np.cos(cel) * np.sin(caz)
        cz = cr * np.sin(cel)
        # Repeat each center npc times, then add per-point scatter
        cx_rep = np.repeat(cx, npc)
        cy_rep = np.repeat(cy, npc)
        cz_rep = np.repeat(cz, npc)
        x = cx_rep + np.random.normal(0, 0.15, total_points)
        y = cy_rep + np.random.normal(0, 0.15, total_points)
        z = cz_rep + np.random.normal(0, 0.10, total_points)
        intensity = np.random.uniform(0.05, 0.3, total_points)
        return np.stack([x, y, z, intensity], axis=1).astype(np.float32)

    def _accumulation(self, pts):
        mask = np.abs(pts[:, 2] + 1.73) < 0.3
        if np.any(mask):
            pts[mask, 2] += min(0.02 * self.snowfall_rate, 0.15)
            mix = min(0.3 * self.snowfall_rate, 0.8)
            snow_ref = np.random.uniform(0.6, 0.9, np.sum(mask))
            pts[mask, 3] = np.clip((1 - mix) * pts[mask, 3] + mix * snow_ref, 0, 1)
        return pts

    def _simulate_lidar_snow_sim(self, points):
        fn = getattr(self._snow_module, 'augment', None)
        if fn is None:
            raise RuntimeError("LiDAR_snow_sim tools/snowfall/simulation.py missing augment(...) entrypoint.")

        pc5 = self._ensure_channel(points)
        in_max_i = float(np.max(points[:, 3])) if len(points) else 1.0
        input_is_normalized = in_max_i <= 1.0 + 1e-6

        # LiDAR_snow_sim snowfall model operates in a 0-255 intensity domain.
        if self.lidar_intensity_mode == 'kitti_01' or (
            self.lidar_intensity_mode == 'auto' and input_is_normalized
        ):
            pc5 = pc5.copy()
            pc5[:, 3] = np.clip(pc5[:, 3], 0.0, 1.0) * 255.0

        particle_file_prefix = self._resolve_particle_file_prefix()

        stats, aug_pc = fn(
            pc=pc5,
            particle_file_prefix=particle_file_prefix,
            beam_divergence=self.beam_divergence,
            shuffle=True,
            show_progressbar=False,
            only_camera_fov=self.only_camera_fov,
            noise_floor=self.noise_floor,
            root_path=self.root_path,
            parallel_backend=self.lidar_parallel_backend
        )
        _ = stats
        out = np.asarray(aug_pc[:, :4], dtype=np.float32)

        if self.lidar_intensity_mode == 'kitti_01':
            out[:, 3] = np.clip(out[:, 3] / 255.0, 0.0, 1.0)
        elif self.lidar_intensity_mode == 'raw_255':
            out[:, 3] = np.clip(out[:, 3], 0.0, 255.0)
        else:
            if input_is_normalized and np.max(out[:, 3]) > 1.0 + 1e-6:
                out[:, 3] = np.clip(out[:, 3] / 255.0, 0.0, 1.0)
            else:
                out[:, 3] = np.clip(out[:, 3], 0.0, 255.0 if np.max(out[:, 3]) > 1.0 else 1.0)
        return out

    def _snowfall_to_rainfall_rate(self, snowfall_rate):
        # Match LiDAR_snow_sim tools/snowfall/sampling.py conversion.
        denom = 487.0 * float(self.snow_density) * 0.003 * float(self.terminal_velocity)
        if denom <= 0:
            raise ValueError("snow_density and terminal_velocity must be positive")
        return float(np.sqrt((float(snowfall_rate) / denom) ** 3))

    def _particle_npy_dir(self):
        if self.root_path:
            return Path(self.root_path).expanduser() / 'training' / 'snowflakes' / 'npy'
        if self._snow_module is None:
            return None
        module_path = Path(getattr(self._snow_module, '__file__', '')).resolve()
        return module_path.parent.parent.parent / 'npy'

    def _load_particle_profiles(self):
        npy_dir = self._particle_npy_dir()
        if npy_dir is None:
            return []

        cache_key = str(npy_dir)
        if cache_key in self._PARTICLE_PROFILE_CACHE:
            return self._PARTICLE_PROFILE_CACHE[cache_key]

        if not npy_dir.exists():
            self._PARTICLE_PROFILE_CACHE[cache_key] = []
            return []

        pattern = re.compile(r'^(gunn|sekhon)_([\deE+\-.]+)_([\deE+\-.]+)_(\d+)\.npy$')
        grouped = {}
        for f in npy_dir.glob('*.npy'):
            m = pattern.match(f.name)
            if not m:
                continue
            model = m.group(1)
            rainfall_rate = float(m.group(2))
            occupancy = float(m.group(3))
            channel = int(m.group(4))
            key = (model, rainfall_rate, occupancy)
            if key not in grouped:
                grouped[key] = set()
            grouped[key].add(channel)

        profiles = []
        for (model, rainfall_rate, occupancy), channels in grouped.items():
            profiles.append({
                'model': model,
                'rainfall_rate': rainfall_rate,
                'occupancy': occupancy,
                'num_channels': len(channels),
            })

        profiles.sort(key=lambda x: (x['model'], x['rainfall_rate']))
        self._PARTICLE_PROFILE_CACHE[cache_key] = profiles
        return profiles

    def _resolve_particle_file_prefix(self):
        if self.particle_file_prefix:
            self.last_particle_file_prefix = self.particle_file_prefix
            try:
                self.last_particle_rainfall_rate = float(str(self.particle_file_prefix).split('_')[1])
            except Exception:
                self.last_particle_rainfall_rate = None
            self.last_target_rainfall_rate = None
            self.last_target_rainfall_level = None
            return self.particle_file_prefix

        profiles = [p for p in self._load_particle_profiles() if p['model'] == self.particle_model]
        if not profiles:
            raise ValueError(
                "LiDAR_snow_sim backend requires particle_file_prefix, or available precomputed particle files. "
                f"No {self.particle_model} profiles found under {self._particle_npy_dir()}."
            )

        rainfall_target = self._snowfall_to_rainfall_rate(self.snowfall_rate)
        if self.rainfall_level_sampling == 'balanced':
            levels = sorted(float(r) for r in self.rainfall_rate_levels)
            nominal = float(np.random.choice(levels))
            idx = levels.index(nominal)
            if len(levels) == 1:
                lo, hi = nominal, nominal
            elif idx == 0:
                right_mid = 0.5 * (levels[idx] + levels[idx + 1])
                lo, hi = max(0.0, nominal - (right_mid - nominal)), right_mid
            elif idx == len(levels) - 1:
                left_mid = 0.5 * (levels[idx - 1] + levels[idx])
                lo, hi = left_mid, nominal + (nominal - left_mid)
            else:
                left_mid = 0.5 * (levels[idx - 1] + levels[idx])
                right_mid = 0.5 * (levels[idx] + levels[idx + 1])
                lo, hi = left_mid, right_mid
            rainfall_target = float(np.random.uniform(lo, hi)) if hi > lo else nominal
        else:
            nominal = min(self.rainfall_rate_levels, key=lambda r: abs(float(r) - rainfall_target))

        selected = min(profiles, key=lambda p: abs(p['rainfall_rate'] - rainfall_target))

        prefix = f"{selected['model']}_{selected['rainfall_rate']}_{selected['occupancy']}"
        self.last_particle_file_prefix = prefix
        self.last_particle_rainfall_rate = selected['rainfall_rate']
        self.last_target_rainfall_rate = rainfall_target
        self.last_target_rainfall_level = float(nominal)

        if selected['num_channels'] < self.num_lasers:
            warnings.warn(
                f"Particle prefix {prefix} has {selected['num_channels']} channels, fewer than num_lasers={self.num_lasers}.",
                RuntimeWarning
            )

        return prefix

    def _infer_channels_from_xyz(self, points):
        xy_norm = np.sqrt(np.maximum(points[:, 0] ** 2 + points[:, 1] ** 2, 1e-12))
        elev = np.degrees(np.arctan2(points[:, 2], xy_norm))
        span = max(self.fov_up_deg - self.fov_down_deg, 1e-6)
        frac = (elev - self.fov_down_deg) / span
        ring = np.floor(frac * self.num_lasers).astype(np.int32)
        ring = np.clip(ring, 0, max(self.num_lasers - 1, 0))
        return ring.astype(np.float32)

    def _ensure_channel(self, points):
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError(f"Expected Nx4/Nx5 point cloud, got shape={points.shape}")
        if points.shape[1] >= 5:
            return np.asarray(points[:, :5], dtype=np.float32)

        if self.channel_mode == 'require':
            raise ValueError(
                "LiDAR_snow_sim backend requires channel dimension (Nx5), "
                "but received Nx4 and channel_mode=require."
            )
        if self.channel_mode == 'zero':
            channel = np.zeros((points.shape[0],), dtype=np.float32)
        else:
            channel = self._infer_channels_from_xyz(points)

        return np.concatenate([points[:, :4].astype(np.float32), channel[:, None]], axis=1)

    def simulate(self, points):
        if self._resolved_backend == 'lidar_snow_sim':
            try:
                return self._simulate_lidar_snow_sim(points)
            except Exception as e:
                if self.backend == 'lidar_snow_sim':
                    raise
                warnings.warn(
                    f"LiDAR_snow_sim call failed ({e}), fallback to heuristic snow simulation.",
                    RuntimeWarning
                )

        if self._resolved_backend == 'lisa':
            rain_points = self._lisa.augment(points, self.snowfall_rate)
            rain_points = np.asarray(rain_points, dtype=np.float32)
            out = rain_points[:, :4]
            out[:, 3] = np.clip(out[:, 3], 0, 1)
            return out

        pts = points.copy()
        pts = self._attenuate(pts)
        pts = self._accumulation(pts)
        for arr in [self._snow_noise(len(pts)), self._clutter()]:
            if len(arr) > 0:
                pts = np.vstack([pts, arr])
        pts[:, 3] = np.clip(pts[:, 3], 0, 1)
        return pts.astype(np.float32)


def _save_param_log(output_dir, param_log, weather_type):
    log_dir = os.path.dirname(output_dir.rstrip('/'))
    txt_path = os.path.join(log_dir, f'{weather_type}_params.txt')
    with open(txt_path, 'w') as f:
        f.write("filename,param_name,param_value\n")
        for fname, params in param_log.items():
            for k, v in params.items():
                f.write(f"{fname},{k},{v}\n")
    npy_path = os.path.join(log_dir, f'{weather_type}_params.npy')
    np.save(npy_path, param_log, allow_pickle=True)
    print(f"  Param log saved to {txt_path}")


def _process_snow_batch(worker_payload):
    """Process a batch of frames in one worker process, reusing simulator instance."""
    (batch_files, input_dir, output_dir,
     random_params, sample_mode, snowfall_rate, seed,
     backend, lidar_snow_sim_path, lisa_path,
     particle_file_prefix, beam_divergence, only_camera_fov,
     noise_floor, root_path, lidar_parallel_backend,
    particle_model, rainfall_rate_levels,
    rainfall_level_sampling, lidar_intensity_mode,
     channel_mode, num_lasers, fov_down_deg, fov_up_deg,
     skip_existing) = worker_payload

    if seed is not None:
        np.random.seed(seed)

    sim = SnowSimulation(
        snowfall_rate=(snowfall_rate if snowfall_rate is not None else 2.5),
        backend=backend,
        lidar_snow_sim_path=lidar_snow_sim_path,
        lisa_path=lisa_path,
        particle_file_prefix=particle_file_prefix,
        beam_divergence=beam_divergence,
        only_camera_fov=only_camera_fov,
        noise_floor=noise_floor,
        root_path=root_path,
        lidar_parallel_backend=lidar_parallel_backend,
        particle_model=particle_model,
        rainfall_rate_levels=rainfall_rate_levels,
        rainfall_level_sampling=rainfall_level_sampling,
        lidar_intensity_mode=lidar_intensity_mode,
        channel_mode=channel_mode,
        num_lasers=num_lasers,
        fov_down_deg=fov_down_deg,
        fov_up_deg=fov_up_deg
    )

    batch_log = {}
    for fname in batch_files:
        out_path = os.path.join(output_dir, fname)
        if skip_existing and os.path.isfile(out_path):
            continue
        sr = sample_snowfall_rate(mode=sample_mode) if random_params else snowfall_rate
        sim.set_snowfall_rate(sr)
        points = load_kitti_points(os.path.join(input_dir, fname))
        result = sim.simulate(points)
        save_kitti_points(result, out_path)
        batch_log[fname] = {
            'snowfall_rate': round(sr, 2),
            'particle_file_prefix': sim.last_particle_file_prefix,
            'particle_rainfall_rate': sim.last_particle_rainfall_rate,
            'target_rainfall_rate': sim.last_target_rainfall_rate,
            'target_rainfall_level': sim.last_target_rainfall_level,
        }
    return batch_log


def process_kitti_snow(input_dir, output_dir, snowfall_rate=None,
                       random_params=False, sample_mode='log', seed=None,
                       backend='auto', lidar_snow_sim_path=None,
                       lisa_path=None,
                       particle_file_prefix=None, beam_divergence=0.35,
                       only_camera_fov=False, noise_floor=0.7, root_path=None,
                       lidar_parallel_backend='thread',
                       particle_model='gunn',
                       rainfall_rate_levels=None,
                       rainfall_level_sampling='nearest',
                       lidar_intensity_mode='kitti_01',
                       num_workers=1,
                       skip_existing=False,
                       channel_mode='infer', num_lasers=64,
                       fov_down_deg=-24.8, fov_up_deg=2.0):
    if seed is not None:
        np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    bin_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.bin')])
    param_log = {}

    if random_params:
        print(f"[Snow] {len(bin_files)} files, RANDOM snowfall_rate, mode={sample_mode}")
    else:
        snowfall_rate = snowfall_rate or 2.5
        print(f"[Snow] {len(bin_files)} files, snowfall_rate={snowfall_rate} mm/h")

    if skip_existing:
        pending = [f for f in bin_files if not os.path.isfile(os.path.join(output_dir, f))]
        skipped = len(bin_files) - len(pending)
        if skipped:
            print(f"[Snow] Skipping {skipped} already-processed files, {len(pending)} remaining")
        bin_files = pending

    if not bin_files:
        return param_log

    if num_workers <= 1:
        sim = SnowSimulation(
            snowfall_rate=(snowfall_rate if snowfall_rate is not None else 2.5),
            backend=backend,
            lidar_snow_sim_path=lidar_snow_sim_path,
            lisa_path=lisa_path,
            particle_file_prefix=particle_file_prefix,
            beam_divergence=beam_divergence,
            only_camera_fov=only_camera_fov,
            noise_floor=noise_floor,
            root_path=root_path,
            lidar_parallel_backend=lidar_parallel_backend,
            particle_model=particle_model,
            rainfall_rate_levels=rainfall_rate_levels,
            rainfall_level_sampling=rainfall_level_sampling,
            lidar_intensity_mode=lidar_intensity_mode,
            channel_mode=channel_mode,
            num_lasers=num_lasers,
            fov_down_deg=fov_down_deg,
            fov_up_deg=fov_up_deg
        )
        for fname in tqdm(bin_files, desc="Snow"):
            sr = sample_snowfall_rate(mode=sample_mode) if random_params else snowfall_rate
            sim.set_snowfall_rate(sr)
            points = load_kitti_points(os.path.join(input_dir, fname))
            result = sim.simulate(points)
            save_kitti_points(result, os.path.join(output_dir, fname))
            param_log[fname] = {
                'snowfall_rate': round(sr, 2),
                'particle_file_prefix': sim.last_particle_file_prefix,
                'particle_rainfall_rate': sim.last_particle_rainfall_rate,
                'target_rainfall_rate': sim.last_target_rainfall_rate,
                'target_rainfall_level': sim.last_target_rainfall_level,
            }
    else:
        num_workers = min(int(num_workers), len(bin_files))
        chunk_size = (len(bin_files) + num_workers - 1) // num_workers
        batches = [bin_files[i:i + chunk_size] for i in range(0, len(bin_files), chunk_size)]

        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for worker_id, batch_files in enumerate(batches):
                worker_seed = None if seed is None else int(seed) + worker_id
                payload = (
                    batch_files, input_dir, output_dir,
                    random_params, sample_mode, snowfall_rate, worker_seed,
                    backend, lidar_snow_sim_path, lisa_path,
                    particle_file_prefix, beam_divergence, only_camera_fov,
                    noise_floor, root_path, lidar_parallel_backend,
                    particle_model, rainfall_rate_levels,
                    rainfall_level_sampling, lidar_intensity_mode,
                    channel_mode, num_lasers, fov_down_deg, fov_up_deg,
                    skip_existing
                )
                futures[executor.submit(_process_snow_batch, payload)] = batch_files

            for future in tqdm(as_completed(futures), total=len(futures), desc="Snow workers"):
                param_log.update(future.result())

    _save_param_log(output_dir, param_log, 'snow')
    return param_log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--snowfall_rate", type=float, default=2.5)
    parser.add_argument("--random_params", action='store_true')
    parser.add_argument("--sample_mode", type=str, default='log',
                        choices=['uniform', 'log', 'category'])
    parser.add_argument("--backend", type=str, default='auto',
                        choices=['auto', 'heuristic', 'lidar_snow_sim', 'lisa'],
                        help="snow backend: auto(优先LiDAR_snow_sim, 次选LISA), heuristic, lidar_snow_sim, lisa")
    parser.add_argument("--lisa_path", type=str, default=None,
                        help="LISA仓库路径(支持 thirdparty/LISA 与 python_old 布局)")
    parser.add_argument("--lidar_snow_sim_path", type=str, default=None,
                        help="LiDAR_snow_sim 中 simulation.py 所在目录")
    parser.add_argument("--particle_file_prefix", type=str, default=None,
                        help="LiDAR_snow_sim augment 必需参数，如 gunn_4.816236598076465_1.1574074074074074")
    parser.add_argument("--beam_divergence", type=float, default=0.35,
                        help="LiDAR_snow_sim beam_divergence (degree)")
    parser.add_argument("--only_camera_fov", action='store_true',
                        help="LiDAR_snow_sim only_camera_fov")
    parser.add_argument("--noise_floor", type=float, default=0.7,
                        help="LiDAR_snow_sim noise_floor")
    parser.add_argument("--root_path", type=str, default=None,
                        help="LiDAR_snow_sim root_path (如STF root)")
    parser.add_argument("--lidar_parallel_backend", type=str, default='thread',
                        choices=['thread', 'process'],
                        help="LiDAR_snow_sim 通道并行后端: thread 或 process")
    parser.add_argument("--particle_model", type=str, default='gunn',
                        choices=['gunn', 'sekhon'],
                        help="LiDAR_snow_sim 自动选前缀时使用的粒径分布模型")
    parser.add_argument("--rainfall_rate_levels", nargs='+', type=float,
                        default=[2.0, 8.0, 17.0, 34.0, 70.0],
                        help="自动选粒子前缀时使用的降雨率档位(mm/h)，默认 2 8 17 34 70")
    parser.add_argument("--rainfall_level_sampling", type=str, default='nearest',
                        choices=['nearest', 'balanced'],
                        help="nearest: 按 snowfall_rate 最近档映射; balanced: 先等概率选档位再在档位内均匀采样")
    parser.add_argument("--lidar_intensity_mode", type=str, default='kitti_01',
                        choices=['kitti_01', 'raw_255', 'auto'],
                        help="LiDAR_snow_sim 强度域处理: kitti_01(输入0~1转255并在输出除255), raw_255(保持0~255), auto(兼容启发式)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="并行处理帧数的worker数(>1启用多进程)")
    parser.add_argument("--skip_existing", action='store_true',
                        help="跳过输出目录中已存在的文件，支持断点续处理")
    parser.add_argument("--channel_mode", type=str, default='infer',
                        choices=['infer', 'zero', 'require'],
                        help="Nx4输入时如何处理channel：infer(按仰角估计)/zero(全0)/require(强制必须Nx5)")
    parser.add_argument("--num_lasers", type=int, default=64,
                        help="channel_mode=infer 时使用的线束数量")
    parser.add_argument("--fov_down_deg", type=float, default=-24.8,
                        help="channel_mode=infer 时垂直FOV下界")
    parser.add_argument("--fov_up_deg", type=float, default=2.0,
                        help="channel_mode=infer 时垂直FOV上界")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    process_kitti_snow(args.input_dir, args.output_dir,
                       snowfall_rate=args.snowfall_rate,
                       random_params=args.random_params,
                       sample_mode=args.sample_mode,
                       backend=args.backend,
                       lidar_snow_sim_path=args.lidar_snow_sim_path,
                       lisa_path=args.lisa_path,
                       particle_file_prefix=args.particle_file_prefix,
                       beam_divergence=args.beam_divergence,
                       only_camera_fov=args.only_camera_fov,
                       noise_floor=args.noise_floor,
                       root_path=args.root_path,
                       lidar_parallel_backend=args.lidar_parallel_backend,
                       particle_model=args.particle_model,
                       rainfall_rate_levels=args.rainfall_rate_levels,
                       rainfall_level_sampling=args.rainfall_level_sampling,
                       lidar_intensity_mode=args.lidar_intensity_mode,
                       num_workers=args.num_workers,
                       skip_existing=args.skip_existing,
                       channel_mode=args.channel_mode,
                       num_lasers=args.num_lasers,
                       fov_down_deg=args.fov_down_deg,
                       fov_up_deg=args.fov_up_deg,
                       seed=args.seed)
