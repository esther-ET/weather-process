"""
fog_simulation.py - 雾天点云模拟（支持随机参数）
输出: 4维 (x, y, z, intensity)
"""

import numpy as np
import os
from tqdm import tqdm
from utils import (load_kitti_points, save_kitti_points,
                   get_lidar_distance, sample_visibility)


class FogSimulation:
    def __init__(self, visibility=500.0, fog_type='uniform'):
        """visibility: m, 建议范围 [30, 2000]"""
        self.visibility = visibility
        self.fog_type = fog_type
        self.lidar_range = 120.0
        self.alpha = 3.912 / self.visibility

    def _attenuate(self, pts):
        dist = get_lidar_distance(pts)
        if self.fog_type == 'inhomogeneous':
            a = self.alpha * np.clip(1 + 0.3 * np.random.randn(len(dist)),
                                     0.3, 2.0)
        else:
            a = self.alpha
        att = np.exp(-2 * a * dist)
        orig_int = pts[:, 3].copy()
        pts[:, 3] *= att

        rho = np.clip(orig_int, 0.01, 1.0)
        recv = rho / (dist ** 2 + 1e-6) * att
        recv_n = recv / (np.max(recv) + 1e-10)
        keep = (recv_n > 0.005) & (np.random.uniform(0, 1, len(pts)) < 0.95)
        return pts[keep], dist[keep], att[keep]

    def _fog_noise(self, pts, dist, att):
        n = int(len(pts) * min(0.3 / self.visibility * 100, 0.15))
        if n == 0:
            return np.empty((0, 4), dtype=np.float32)
        n_ray, n_rand = int(n * 0.7), n - int(n * 0.7)
        parts = []

        if n_ray > 0 and len(pts) > 0:
            idx = np.random.choice(len(pts), min(n_ray, len(pts)),
                                   replace=n_ray > len(pts))
            ratio = np.random.beta(2, 5, len(idx))
            fd = dist[idx] * ratio
            d = pts[idx, :3] / (dist[idx, None] + 1e-6)
            xyz = d * fd[:, None]
            fi = self.alpha * np.exp(-2 * self.alpha * fd)
            fi = np.clip(fi / (np.max(fi) + 1e-10) * 0.3, 0, 0.4)
            parts.append(np.column_stack([xyz, fi]))

        if n_rand > 0:
            r = np.clip(np.random.exponential(self.visibility / 5, n_rand),
                        0.5, min(self.visibility, self.lidar_range))
            az = np.random.uniform(-np.pi / 2, np.pi / 2, n_rand)
            el = np.random.uniform(np.radians(-24.8), np.radians(2.0), n_rand)
            parts.append(np.stack([
                r * np.cos(el) * np.cos(az),
                r * np.cos(el) * np.sin(az),
                r * np.sin(el),
                np.clip(np.random.exponential(0.05, n_rand), 0, 0.2)
            ], axis=1))

        return np.vstack(parts).astype(np.float32) if parts else np.empty((0, 4), dtype=np.float32)

    def _curtain(self):
        if self.visibility > 200:
            return np.empty((0, 4), dtype=np.float32)
        n = int(200 * (200 / self.visibility))
        cd = self.visibility * np.random.uniform(0.5, 1.0)
        az = np.random.uniform(-np.pi / 4, np.pi / 4, n)
        el = np.random.uniform(np.radians(-20), np.radians(2.0), n)
        r = np.clip(cd + np.random.normal(0, 2.0, n), cd * 0.8, cd * 1.2)
        return np.stack([
            r * np.cos(el) * np.cos(az),
            r * np.cos(el) * np.sin(az),
            r * np.sin(el),
            np.random.uniform(0.02, 0.15, n)
        ], axis=1).astype(np.float32)

    def simulate(self, points):
        pts = points.copy()
        pts, dist, att = self._attenuate(pts)
        for arr in [self._fog_noise(pts, dist, att), self._curtain()]:
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


def process_kitti_fog(input_dir, output_dir, visibility=None, fog_type='uniform',
                      random_params=False, sample_mode='log', seed=None):
    if seed is not None:
        np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    bin_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.bin')])
    param_log = {}

    if random_params:
        print(f"[Fog] {len(bin_files)} files, RANDOM visibility, mode={sample_mode}")
    else:
        visibility = visibility or 500.0
        print(f"[Fog] {len(bin_files)} files, visibility={visibility}m")

    for fname in tqdm(bin_files, desc="Fog"):
        if random_params:
            vis = sample_visibility(mode=sample_mode)
            # 低能见度时自动用非均匀雾
            ft = 'inhomogeneous' if vis < 200 else fog_type
        else:
            vis = visibility
            ft = fog_type

        param_log[fname] = {'visibility': round(vis, 1), 'fog_type': ft}
        sim = FogSimulation(visibility=vis, fog_type=ft)
        points = load_kitti_points(os.path.join(input_dir, fname))
        result = sim.simulate(points)
        save_kitti_points(result, os.path.join(output_dir, fname))

    _save_param_log(output_dir, param_log, 'fog')
    return param_log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--visibility", type=float, default=500.0)
    parser.add_argument("--fog_type", type=str, default='uniform',
                        choices=['uniform', 'inhomogeneous'])
    parser.add_argument("--random_params", action='store_true')
    parser.add_argument("--sample_mode", type=str, default='log',
                        choices=['uniform', 'log', 'category'])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    process_kitti_fog(args.input_dir, args.output_dir,
                      visibility=args.visibility,
                      fog_type=args.fog_type,
                      random_params=args.random_params,
                      sample_mode=args.sample_mode,
                      seed=args.seed)