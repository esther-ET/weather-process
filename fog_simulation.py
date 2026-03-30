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
        # Koschmieder近似（5%对比阈值）: MOR = ln(20)/alpha
        # 使用ln(20)可与常见LiDAR雾仿真设置保持一致
        self.alpha = np.log(20.0) / self.visibility
        # TripleMixer风格soft target中的后向散射系数beta与MOR成反比
        # 这里用保守默认值（对应论文/代码中的beta_min量级）
        self.beta = 0.023 / max(self.visibility, 1.0)

    def _attenuate(self, pts):
        dist = get_lidar_distance(pts)
        if self.fog_type == 'inhomogeneous':
            alpha = self.alpha * np.clip(1 + 0.3 * np.random.randn(len(dist)),
                                         0.3, 2.0)
        else:
            alpha = np.full(len(dist), self.alpha, dtype=np.float32)

        # hard target: 与TripleMixer中的 P_R_fog_hard 一致（I <- I * exp(-2αr)）
        att = np.exp(-2 * alpha * dist)
        orig_int = pts[:, 3].copy()
        pts[:, 3] = np.clip(pts[:, 3] * att, 0, 1)
        return pts, dist, orig_int

    def _soft_backscatter(self, pts, dist, orig_int):
        """
        TripleMixer中的soft target核心行为：
        - 若后向散射回波强于hard target回波，则把点拉回更近距离并替换强度
        """
        if len(pts) == 0:
            return pts

        # 用Beta分布近似积分表给出的fog_distance/r_0
        ratio = np.random.beta(2, 5, len(pts))
        fog_dist = np.clip(dist * ratio, 0.5, np.minimum(dist, self.lidar_range))

        # surrogate fog response（原实现依赖预计算积分表，这里用解析近似）
        fog_resp = orig_int * (dist ** 2) * self.beta * np.exp(-2 * self.alpha * fog_dist)
        fog_resp = np.clip(fog_resp, 0, 1)

        replace = fog_resp > pts[:, 3]
        if np.any(replace):
            direction = pts[:, :3] / (dist[:, None] + 1e-6)
            pts[replace, :3] = direction[replace] * fog_dist[replace, None]
            pts[replace, 3] = fog_resp[replace]
        return pts

    def simulate(self, points):
        pts = points.copy()
        pts, dist, orig_int = self._attenuate(pts)
        pts = self._soft_backscatter(pts, dist, orig_int)
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
