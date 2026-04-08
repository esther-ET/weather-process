"""
snow_simulation.py - 雪天点云模拟（支持随机参数）
输出: 4维 (x, y, z, intensity)
"""

import numpy as np
import os
import warnings
import importlib
import inspect
import sys
from pathlib import Path
from tqdm import tqdm
from utils import (load_kitti_points, save_kitti_points,
                   get_lidar_distance, sample_snowfall_rate)


class SnowSimulation:
    def __init__(self, snowfall_rate=2.5, terminal_velocity=1.0, snow_density=0.1,
                 backend='auto', lidar_snow_sim_path=None):
        """snowfall_rate: mm/h 水当量, 建议范围 [0.5, 10]"""
        self.snowfall_rate = snowfall_rate
        self.terminal_velocity = terminal_velocity
        self.snow_density = snow_density
        self.backend = backend
        self.lidar_snow_sim_path = lidar_snow_sim_path
        self.lidar_range = 120.0
        self.d_snow = 2.0 + 0.5 * np.log(1 + self.snowfall_rate)
        self.snow_conc = self._concentration()
        self._snow_module = None
        self._resolved_backend = self._resolve_backend()

    def _resolve_backend(self):
        if self.backend == 'heuristic':
            return 'heuristic'

        module = self._import_lidar_snow_sim()
        if module is not None:
            self._snow_module = module
            return 'lidar_snow_sim'

        if self.backend == 'lidar_snow_sim':
            raise ImportError(
                "backend='lidar_snow_sim' but simulation module is not importable. "
                "Set --lidar_snow_sim_path or export LIDAR_SNOW_SIM_PATH "
                "to the directory containing simulation.py."
            )

        warnings.warn(
            "LiDAR_snow_sim backend not found, falling back to heuristic snow simulation.",
            RuntimeWarning
        )
        return 'heuristic'

    def _import_lidar_snow_sim(self):
        search_paths = []
        if self.lidar_snow_sim_path:
            search_paths.append(self.lidar_snow_sim_path)
        env_path = os.getenv('LIDAR_SNOW_SIM_PATH')
        if env_path:
            search_paths.append(env_path)

        for p in search_paths:
            if p and p not in sys.path:
                sys.path.insert(0, str(Path(p).expanduser()))

        try:
            return importlib.import_module('simulation')
        except Exception:
            return None

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
        parts = []
        for _ in range(nc):
            cr = np.random.uniform(1, 10)
            caz = np.random.uniform(-np.pi, np.pi)
            cel = np.random.uniform(np.radians(-15), np.radians(2))
            cx = cr * np.cos(cel) * np.cos(caz)
            cy = cr * np.cos(cel) * np.sin(caz)
            cz = cr * np.sin(cel)
            parts.append(np.stack([
                cx + np.random.normal(0, 0.15, npc),
                cy + np.random.normal(0, 0.15, npc),
                cz + np.random.normal(0, 0.10, npc),
                np.random.uniform(0.05, 0.3, npc)
            ], axis=1))
        return np.vstack(parts).astype(np.float32) if parts else np.empty((0, 4), dtype=np.float32)

    def _accumulation(self, pts):
        mask = np.abs(pts[:, 2] + 1.73) < 0.3
        if np.any(mask):
            pts[mask, 2] += min(0.02 * self.snowfall_rate, 0.15)
            mix = min(0.3 * self.snowfall_rate, 0.8)
            snow_ref = np.random.uniform(0.6, 0.9, np.sum(mask))
            pts[mask, 3] = np.clip((1 - mix) * pts[mask, 3] + mix * snow_ref, 0, 1)
        return pts

    def _simulate_lidar_snow_sim(self, points):
        candidates = ('augment', 'simulate', 'simulate_snow')
        for name in candidates:
            fn = getattr(self._snow_module, name, None)
            if fn is None:
                continue
            try:
                sig = inspect.signature(fn)
                kwargs = {}
                for p_name in sig.parameters:
                    if p_name in ('points', 'pc', 'point_cloud', 'pointcloud'):
                        kwargs[p_name] = points
                    elif p_name in ('snowfall_rate', 'rate'):
                        kwargs[p_name] = self.snowfall_rate
                out = fn(**kwargs) if kwargs else fn(points)
                if isinstance(out, tuple):
                    out = out[0]
                out = np.asarray(out, dtype=np.float32)
                if out.ndim == 2 and out.shape[1] >= 4:
                    out = out[:, :4]
                    out[:, 3] = np.clip(out[:, 3], 0, 1)
                    return out
            except Exception:
                continue
        raise RuntimeError(
            "Could not call LiDAR_snow_sim simulation entrypoint. "
            "Expected one of: augment/simulate/simulate_snow."
        )

    def simulate(self, points):
        if self._resolved_backend == 'lidar_snow_sim':
            return self._simulate_lidar_snow_sim(points)

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


def process_kitti_snow(input_dir, output_dir, snowfall_rate=None,
                       random_params=False, sample_mode='log', seed=None,
                       backend='auto', lidar_snow_sim_path=None):
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

    for fname in tqdm(bin_files, desc="Snow"):
        sr = sample_snowfall_rate(mode=sample_mode) if random_params else snowfall_rate
        param_log[fname] = {'snowfall_rate': round(sr, 2)}
        sim = SnowSimulation(
            snowfall_rate=sr,
            backend=backend,
            lidar_snow_sim_path=lidar_snow_sim_path
        )
        points = load_kitti_points(os.path.join(input_dir, fname))
        result = sim.simulate(points)
        save_kitti_points(result, os.path.join(output_dir, fname))

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
                        choices=['auto', 'heuristic', 'lidar_snow_sim'],
                        help="snow backend: auto(优先LiDAR_snow_sim), heuristic(当前启发式), lidar_snow_sim(强制)")
    parser.add_argument("--lidar_snow_sim_path", type=str, default=None,
                        help="LiDAR_snow_sim 中 simulation.py 所在目录")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    process_kitti_snow(args.input_dir, args.output_dir,
                       snowfall_rate=args.snowfall_rate,
                       random_params=args.random_params,
                       sample_mode=args.sample_mode,
                       backend=args.backend,
                       lidar_snow_sim_path=args.lidar_snow_sim_path,
                       seed=args.seed)
