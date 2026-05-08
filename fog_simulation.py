"""
fog_simulation.py - 雾天点云模拟（与 LiDAR_fog_sim 对齐）
输出: 4维 (x, y, z, intensity)
"""

import copy
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils import load_kitti_points, save_kitti_points, sample_visibility


RNG = np.random.default_rng(seed=42)
AVAILABLE_TAU_HS = [20]


class ParameterSet:
    """Parameter defaults mirrored from MartinHahner/LiDAR_fog_sim.

    The upstream implementation computes dependent values (``mor``, ``beta``
    and ``beta_0``) before applying ``kwargs``.  Keeping that order is
    important for bit-level compatibility with the released lookup tables: in
    upstream, overriding ``alpha`` does not automatically recompute ``beta``.
    """

    def __init__(self, **kwargs):
        self.n = 500
        self.n_min = 100
        self.n_max = 1000

        self.r_range = 100
        self.r_range_min = 50
        self.r_range_max = 250

        # soft target / fog
        self.alpha = 0.06
        self.alpha_min = 0.003
        self.alpha_max = 0.5
        self.alpha_scale = 1000
        self.mor = np.log(20) / self.alpha
        self.beta = 0.046 / self.mor
        self.beta_min = 0.023 / self.mor
        self.beta_max = 0.092 / self.mor
        self.beta_scale = 1000 * self.mor

        # sensor params
        self.p_0 = 80
        self.p_0_min = 60
        self.p_0_max = 100
        self.tau_h = 2e-8
        self.tau_h_min = 5e-9
        self.tau_h_max = 8e-8
        self.tau_h_scale = 1e9
        self.e_p = self.p_0 * self.tau_h
        self.a_r = 0.25
        self.a_r_min = 0.01
        self.a_r_max = 0.1
        self.a_r_scale = 1000
        self.l_r = 0.05
        self.l_r_min = 0.01
        self.l_r_max = 0.10
        self.l_r_scale = 100
        self.c_a = 299792458.0 * self.l_r * self.a_r / 2
        self.linear_xsi = True
        self.D = 0.1
        self.ROH_T = 0.01
        self.ROH_R = 0.01
        self.GAMMA_T_DEG = 2
        self.GAMMA_R_DEG = 3.5
        self.GAMMA_T = np.deg2rad(self.GAMMA_T_DEG)
        self.GAMMA_R = np.deg2rad(self.GAMMA_R_DEG)
        self.r_1 = 0.9
        self.r_1_min = 0
        self.r_1_max = 10
        self.r_1_scale = 10
        self.r_2 = 1.0
        self.r_2_min = 0
        self.r_2_max = 10
        self.r_2_scale = 10

        # hard target
        self.r_0 = 30
        self.r_0_min = 1
        self.r_0_max = 200
        self.gamma = 0.000001
        self.gamma_min = 0.0000001
        self.gamma_max = 0.00001
        self.gamma_scale = 10000000
        self.beta_0 = self.gamma / np.pi

        self.__dict__.update(kwargs)


class FogSimulation:
    def __init__(self, visibility=500.0, fog_type='uniform',
                 integral_root=None, noise=10, noise_variant='v1',
                 hard=True, soft=True, gain=False):
        """
        Args:
            visibility: 能见度（米）
            fog_type: uniform/inhomogeneous
            integral_root: 积分查表目录，默认使用 weather-process/integral_lookup_tables/original
            noise/noise_variant/hard/soft/gain: 与 LiDAR_fog_sim 的 simulate_fog 参数语义一致
        """
        self.visibility = max(float(visibility), 1e-6)
        self.fog_type = fog_type
        self.noise = int(noise)
        self.noise_variant = noise_variant
        self.hard = hard
        self.soft = soft
        self.gain = gain
        self._warned_missing_integral = False

        alpha = np.log(20.0) / self.visibility
        # Match upstream: ParameterSet(alpha=...) changes attenuation for hard
        # targets and lookup-table selection, but keeps beta at the default value.
        self.param_set = ParameterSet(alpha=alpha, gamma=0.000001)

        if integral_root is None:
            integral_root = (
                Path(__file__).resolve().parent
                / 'integral_lookup_tables'
                / 'original'
            )
        self.integral_path = Path(integral_root)
        self._integral_dict = None

    def _get_available_alphas(self):
        alphas = []
        if not self.integral_path.exists():
            return alphas
        for file in os.listdir(self.integral_path):
            if file.endswith('.pickle'):
                alpha = file.split('_')[-1].replace('.pickle', '')
                try:
                    alphas.append(float(alpha))
                except ValueError:
                    continue
        return sorted(alphas)

    def _get_integral_dict(self):
        if self._integral_dict is not None:
            return self._integral_dict

        alphas = self._get_available_alphas()
        if not alphas:
            if not self._warned_missing_integral:
                print(f"[FogSimulation] WARNING: integral table not found in {self.integral_path}; soft fog disabled.")
                self._warned_missing_integral = True
            return None

        p = self.param_set
        alpha = min(alphas, key=lambda x: abs(x - p.alpha))
        tau_h = min(AVAILABLE_TAU_HS, key=lambda x: abs(x - int(p.tau_h * 1e9)))

        filename = self.integral_path / (
            f'integral_0m_to_200m_stepsize_0.1m_tau_h_{tau_h}ns_alpha_{alpha}.pickle'
        )
        if not filename.exists():
            return None

        with open(filename, 'rb') as f:
            self._integral_dict = pickle.load(f)
        return self._integral_dict

    def _prepare_intensity(self, points):
        """兼容 [0,1] 与 [0,255] 强度输入，内部统一到 [0,255]。"""
        pts = points.copy().astype(np.float32)
        max_i = float(np.max(pts[:, 3])) if len(pts) > 0 else 1.0
        use_unit_scale = max_i <= 1.0 + 1e-6
        if use_unit_scale:
            pts[:, 3] *= 255.0
        return pts, use_unit_scale

    def _recover_intensity(self, points, use_unit_scale):
        pts = points.copy().astype(np.float32)
        if use_unit_scale:
            pts[:, 3] = np.clip(pts[:, 3] / 255.0, 0.0, 1.0)
        return pts

    def _p_r_fog_hard(self, points):
        p = self.param_set
        pc = points.copy()
        r_0 = np.linalg.norm(pc[:, 0:3], axis=1)

        if self.fog_type == 'inhomogeneous':
            alpha_vec = p.alpha * np.clip(1.0 + 0.3 * np.random.randn(len(r_0)), 0.3, 2.0)
            pc[:, 3] = np.round(np.exp(-2.0 * alpha_vec * r_0) * pc[:, 3])
        else:
            pc[:, 3] = np.round(np.exp(-2.0 * p.alpha * r_0) * pc[:, 3])
        return pc

    def _p_r_fog_soft(self, pc_hard, original_intensity):
        p = self.param_set
        integral_dict = self._get_integral_dict()
        if integral_dict is None:
            return pc_hard

        augmented_pc = np.zeros(pc_hard.shape, dtype=np.float32)
        r_zeros = np.linalg.norm(pc_hard[:, 0:3], axis=1)
        r_noise = 10

        for i, r_0 in enumerate(r_zeros):
            key = float(str(round(float(r_0), 1)))
            fog_distance, fog_response = integral_dict[min(key, 200)]

            fog_response = fog_response * original_intensity[i] * (r_0 ** 2) * p.beta / p.beta_0
            fog_response = min(fog_response, 255)

            if fog_response > pc_hard[i, 3]:
                scaling_factor = fog_distance / max(r_0, 1e-6)
                augmented_pc[i, 0] = pc_hard[i, 0] * scaling_factor
                augmented_pc[i, 1] = pc_hard[i, 1] * scaling_factor
                augmented_pc[i, 2] = pc_hard[i, 2] * scaling_factor
                augmented_pc[i, 3] = fog_response
                if pc_hard.shape[1] > 4:
                    augmented_pc[i, 4:] = pc_hard[i, 4:]

                if self.noise > 0:
                    if self.noise_variant == 'v1':
                        distance_noise = RNG.uniform(low=r_0 - self.noise, high=r_0 + self.noise, size=1)[0]
                        noise_factor = r_0 / max(distance_noise, 1e-6)
                    elif self.noise_variant == 'v2':
                        power = RNG.uniform(low=-1, high=1, size=1)[0]
                        noise_factor = max(1.0, self.noise / 5) ** power
                    elif self.noise_variant == 'v3':
                        power = RNG.uniform(low=-0.5, high=1, size=1)[0]
                        noise_factor = max(1.0, self.noise * 4 / 10) ** power
                    elif self.noise_variant == 'v4':
                        additive = r_noise * RNG.beta(a=2, b=20, size=1)[0]
                        new_dist = fog_distance + additive
                        noise_factor = new_dist / max(fog_distance, 1e-6)
                    else:
                        raise NotImplementedError(
                            f"noise variant '{self.noise_variant}' is not implemented"
                        )

                    augmented_pc[i, 0] *= noise_factor
                    augmented_pc[i, 1] *= noise_factor
                    augmented_pc[i, 2] *= noise_factor
            else:
                augmented_pc[i] = pc_hard[i]

        if self.gain:
            max_intensity = np.ceil(max(augmented_pc[:, 3]))
            if max_intensity > 0:
                augmented_pc[:, 3] *= 255 / max_intensity

        return augmented_pc

    def simulate(self, points):
        if len(points) == 0:
            return points.astype(np.float32)

        pts_255, use_unit_scale = self._prepare_intensity(points)
        original_intensity = copy.deepcopy(pts_255[:, 3])
        augmented_pc = copy.deepcopy(pts_255)

        if self.hard:
            augmented_pc = self._p_r_fog_hard(augmented_pc)
        if self.soft:
            augmented_pc = self._p_r_fog_soft(augmented_pc, original_intensity)

        augmented_pc[:, 3] = np.clip(augmented_pc[:, 3], 0.0, 255.0)
        out = self._recover_intensity(augmented_pc, use_unit_scale)
        return out.astype(np.float32)


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
            # Upstream LiDAR_fog_sim uses a spatially uniform attenuation
            # coefficient; keep random sampling on visibility only unless the
            # caller explicitly requests the local inhomogeneous extension.
            ft = fog_type
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
