"""
rain_simulation.py - 雨天点云模拟（支持随机参数）
输出: 4维 (x, y, z, intensity)
"""

import numpy as np
import os
from tqdm import tqdm
from utils import (load_kitti_points, save_kitti_points,
                   get_lidar_distance, sample_rain_rate)


class RainSimulation:
    def __init__(self, rain_rate=10.0):
        """rain_rate: mm/h, 建议范围 [1, 80]"""
        self.rain_rate = rain_rate
        self.lidar_range = 120.0
        self.d0 = 1.238 * self.rain_rate ** 0.182
        self.lambda_mp = 4.1 * self.rain_rate ** (-0.21)

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

    def simulate(self, points):
        pts = points.copy()
        pts = self._attenuate(pts)
        pts = self._wet_ground(pts)
        noise = self._rain_noise(len(pts))
        if len(noise) > 0:
            pts = np.vstack([pts, noise])
        pts[:, 3] = np.clip(pts[:, 3], 0, 1)
        return pts.astype(np.float32)


def process_kitti_rain(input_dir, output_dir, rain_rate=None,
                       random_params=False, sample_mode='log', seed=None):
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

        sim = RainSimulation(rain_rate=rr)
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
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    process_kitti_rain(args.input_dir, args.output_dir,
                       rain_rate=args.rain_rate,
                       random_params=args.random_params,
                       sample_mode=args.sample_mode,
                       seed=args.seed)
