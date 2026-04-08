"""
rain_simulation.py - 雨天点云模拟（支持随机参数）
输出: 4维 (x, y, z, intensity)
"""

import numpy as np
import os
import warnings
import importlib
import sys
from pathlib import Path
from tqdm import tqdm
from utils import (load_kitti_points, save_kitti_points,
                   get_lidar_distance, sample_rain_rate)


class RainSimulation:
    def __init__(self, rain_rate=10.0, backend='auto', lisa_path=None):
        """rain_rate: mm/h, 建议范围 [1, 80]"""
        self.rain_rate = rain_rate
        self.backend = backend
        self.lisa_path = lisa_path
        self.lidar_range = 120.0
        self.d0 = 1.238 * self.rain_rate ** 0.182
        self.lambda_mp = 4.1 * self.rain_rate ** (-0.21)
        self._lisa = None
        self._lisa_impl = None
        self._resolved_backend = self._resolve_backend()

    def _resolve_backend(self):
        if self.backend == 'heuristic':
            return 'heuristic'

        lisa_handle = self._import_lisa()
        if lisa_handle is not None:
            impl, lisa_cls = lisa_handle
            self._lisa_impl = impl
            if impl == 'legacy':
                self._lisa = lisa_cls(atm_model='rain')
            elif impl == 'python':
                self._lisa = lisa_cls(mode='rain')
            elif impl == 'pylisa':
                lidar = importlib.import_module('pylisa').Lidar()
                water = importlib.import_module('pylisa').Water()
                rain = importlib.import_module('pylisa').MarshallPalmerRain()
                self._lisa = lisa_cls(lidar, water, rain)
            return 'lisa'

        if self.backend == 'lisa':
            raise ImportError(
                "backend='lisa' but LISA is not importable. "
                "Set --lisa_path or export LISA_PATH to the LISA repository root."
            )

        warnings.warn(
            "LISA backend not found, falling back to heuristic rain simulation.",
            RuntimeWarning
        )
        return 'heuristic'

    def _import_lisa(self):
        search_paths = []
        if self.lisa_path:
            search_paths.append(self.lisa_path)
            search_paths.append(str(Path(self.lisa_path).expanduser() / 'python'))
        env_lisa_path = os.getenv('LISA_PATH')
        if env_lisa_path:
            search_paths.append(env_lisa_path)
            search_paths.append(str(Path(env_lisa_path).expanduser() / 'python'))

        for p in search_paths:
            if p and p not in sys.path:
                sys.path.insert(0, str(Path(p).expanduser()))

        # legacy python version: from atmos_models import LISA
        try:
            module = importlib.import_module('atmos_models')
            cls = getattr(module, 'LISA', None)
            if cls is not None:
                return ('legacy', cls)
        except Exception:
            pass

        # MartinHahner/LISA@76cdb86 python implementation: from lisa import LISA
        try:
            module = importlib.import_module('lisa')
            cls = getattr(module, 'LISA', None)
            if cls is not None:
                return ('python', cls)
        except Exception:
            pass

        # pip package variant
        try:
            module = importlib.import_module('pylisa')
            cls = getattr(module, 'Lisa', None)
            if cls is not None:
                return ('pylisa', cls)
        except Exception:
            pass

        return None

    def _extinction(self):
        k, a = 0.01, 0.65
        ext_db_km = k * (self.rain_rate ** a)
        return ext_db_km / (10 * np.log10(np.e)) / 1000.0

    def _attenuate(self, pts):
        dist = get_lidar_distance(pts)
        att = np.exp(-2 * self._extinction() * dist)
        pts[:, 3] *= att
        keep = np.random.uniform(0, 1, len(pts)) < np.clip(att, 0.3, 1.0)
        return pts[keep]

    def _rain_noise(self, n_orig):
        n = int(n_orig * min(0.005 * self.rain_rate, 0.15))
        if n == 0:
            return np.empty((0, 4), dtype=np.float32)
        r = np.clip(np.random.exponential(15.0, n), 0.5, 50.0)
        # KITTI Velodyne bin是360°扫描，噪声方位角应覆盖全周
        az = np.random.uniform(-np.pi, np.pi, n)
        el = np.random.uniform(np.radians(-24.8), np.radians(2.0), n)
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        i = np.random.uniform(0.0, 0.15, n)
        return np.stack([x, y, z, i], axis=1).astype(np.float32)

    def _wet_ground(self, pts):
        mask = np.abs(pts[:, 2] + 1.73) < 0.3
        if np.any(mask):
            n = np.sum(mask)
            up = np.random.uniform(0, 1, n) > 0.5
            s = np.where(up,
                         np.random.uniform(1.1, 1.5, n),
                         np.random.uniform(0.5, 0.9, n))
            pts[mask, 3] = np.clip(pts[mask, 3] * s, 0, 1)
        return pts

    def _simulate_lisa(self, points, labels=None):
        if self._lisa_impl == 'legacy':
            if labels is None:
                labels = np.zeros((points.shape[0], 1), dtype=np.int32)
            rain_points, _ = self._lisa.augment_mc(points, labels, self.rain_rate)
            out = np.asarray(rain_points[:, :4], dtype=np.float32)
        else:
            rain_points = self._lisa.augment(points, self.rain_rate)
            rain_points = np.asarray(rain_points, dtype=np.float32)
            out = rain_points[:, :4]

        out[:, 3] = np.clip(out[:, 3], 0, 1)
        return out

    def simulate(self, points, labels=None):
        if self._resolved_backend == 'lisa':
            return self._simulate_lisa(points, labels=labels)

        pts = points.copy()
        pts = self._attenuate(pts)
        pts = self._wet_ground(pts)
        noise = self._rain_noise(len(pts))
        if len(noise) > 0:
            pts = np.vstack([pts, noise])
        pts[:, 3] = np.clip(pts[:, 3], 0, 1)
        return pts.astype(np.float32)


def process_kitti_rain(input_dir, output_dir, rain_rate=None,
                       random_params=False, sample_mode='log', seed=None,
                       backend='auto', lisa_path=None):
    """
    Args:
        rain_rate: 固定降雨率 (random_params=False时使用)
        random_params: True则每帧随机采样降雨率
        sample_mode: 'uniform' | 'log' | 'category'
        seed: 随机种子 (可复现)
    """
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    bin_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.bin')])

    # 记录每帧使用的参数（方便复现和分析）
    param_log = {}

    if random_params:
        print(f"[Rain] {len(bin_files)} files, RANDOM rain_rate, mode={sample_mode}")
    else:
        rain_rate = rain_rate or 10.0
        print(f"[Rain] {len(bin_files)} files, rain_rate={rain_rate} mm/h")

    for fname in tqdm(bin_files, desc="Rain"):
        if random_params:
            rr = sample_rain_rate(mode=sample_mode)
        else:
            rr = rain_rate

        param_log[fname] = {'rain_rate': round(rr, 2)}

        sim = RainSimulation(rain_rate=rr, backend=backend, lisa_path=lisa_path)
        points = load_kitti_points(os.path.join(input_dir, fname))
        result = sim.simulate(points)
        save_kitti_points(result, os.path.join(output_dir, fname))

    # 保存参数日志
    _save_param_log(output_dir, param_log, 'rain')
    return param_log


def _save_param_log(output_dir, param_log, weather_type):
    """保存每帧的参数到txt和npy"""
    log_dir = os.path.dirname(output_dir.rstrip('/'))
    # txt日志
    txt_path = os.path.join(log_dir, f'{weather_type}_params.txt')
    with open(txt_path, 'w') as f:
        f.write("filename,param_name,param_value\n")
        for fname, params in param_log.items():
            for k, v in params.items():
                f.write(f"{fname},{k},{v}\n")
    # npy日志
    npy_path = os.path.join(log_dir, f'{weather_type}_params.npy')
    np.save(npy_path, param_log, allow_pickle=True)
    print(f"  Param log saved to {txt_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rain_rate", type=float, default=10.0)
    parser.add_argument("--random_params", action='store_true',
                        help="每帧随机采样降雨率")
    parser.add_argument("--sample_mode", type=str, default='log',
                        choices=['uniform', 'log', 'category'])
    parser.add_argument("--backend", type=str, default='auto',
                        choices=['auto', 'heuristic', 'lisa'],
                        help="rain backend: auto(优先LISA), heuristic(当前启发式), lisa(强制LISA)")
    parser.add_argument("--lisa_path", type=str, default=None,
                        help="LISA仓库路径(应包含 atmos_models.py)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    process_kitti_rain(args.input_dir, args.output_dir,
                       rain_rate=args.rain_rate,
                       random_params=args.random_params,
                       sample_mode=args.sample_mode,
                       backend=args.backend,
                       lisa_path=args.lisa_path,
                       seed=args.seed)
