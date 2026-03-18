"""
vis_and_diff.py - 天气数据可视化与差异分析
功能:
1. 点云3D可视化对比 (BEV + 侧视图)
2. 强度分布对比
3. 点密度分布对比
4. 距离-强度关系对比
5. 差异热力图
6. 统计指标量化
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 服务器无显示器时使用
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from collections import OrderedDict
import json


# ============ 基础工具 ============

def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def get_distance(pts):
    return np.sqrt(np.sum(pts[:, :3] ** 2, axis=1))


# ============ 1. 点云BEV可视化 ============

def plot_bev_comparison(clean_pts, weather_pts_dict, save_path,
                        xlim=(-40, 40), ylim=(0, 70), title_prefix=""):
    """
    鸟瞰图(BEV)对比: 清晰 vs 各天气
    颜色编码: 强度值
    """
    n_weather = len(weather_pts_dict)
    fig, axes = plt.subplots(1, n_weather + 1, figsize=(6 * (n_weather + 1), 6))
    if n_weather == 0:
        axes = [axes]

    def draw_bev(ax, pts, title):
        mask = (pts[:, 0] > xlim[0]) & (pts[:, 0] < xlim[1]) & \
               (pts[:, 1] > ylim[0]) & (pts[:, 1] < ylim[1])
        p = pts[mask]
        sc = ax.scatter(p[:, 1], p[:, 0], c=p[:, 3], cmap='viridis',
                        s=0.1, vmin=0, vmax=1, alpha=0.8)
        ax.set_xlim(ylim)
        ax.set_ylim(xlim)
        ax.set_title(f"{title}\n({len(p):,} pts)", fontsize=12)
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('X (m)')
        ax.set_aspect('equal')
        return sc

    sc = draw_bev(axes[0], clean_pts, f"{title_prefix}Clean")
    for i, (name, pts) in enumerate(weather_pts_dict.items()):
        draw_bev(axes[i + 1], pts, f"{title_prefix}{name}")

    fig.colorbar(sc, ax=axes, label='Intensity', shrink=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved BEV: {save_path}")


def plot_side_view_comparison(clean_pts, weather_pts_dict, save_path,
                              xlim=(0, 70), zlim=(-3, 3)):
    """
    侧视图对比: X-Z平面
    """
    n = len(weather_pts_dict)
    fig, axes = plt.subplots(n + 1, 1, figsize=(14, 3 * (n + 1)))
    if n == 0:
        axes = [axes]

    def draw_side(ax, pts, title):
        mask = (pts[:, 0] > xlim[0]) & (pts[:, 0] < xlim[1]) & \
               (pts[:, 2] > zlim[0]) & (pts[:, 2] < zlim[1])
        p = pts[mask]
        ax.scatter(p[:, 0], p[:, 2], c=p[:, 3], cmap='viridis',
                   s=0.1, vmin=0, vmax=1, alpha=0.6)
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)
        ax.set_title(f"{title} ({len(p):,} pts)", fontsize=11)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')

    draw_side(axes[0], clean_pts, "Clean")
    for i, (name, pts) in enumerate(weather_pts_dict.items()):
        draw_side(axes[i + 1], pts, name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved Side View: {save_path}")


# ============ 2. 强度分布对比 ============

def plot_intensity_distribution(clean_pts, weather_pts_dict, save_path):
    """
    强度值直方图叠加对比
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bins = np.linspace(0, 1, 100)
    ax.hist(clean_pts[:, 3], bins=bins, density=True, alpha=0.6,
            label=f'Clean (μ={clean_pts[:,3].mean():.3f})', color='black')

    colors = plt.cm.Set1(np.linspace(0, 1, len(weather_pts_dict)))
    for (name, pts), c in zip(weather_pts_dict.items(), colors):
        ax.hist(pts[:, 3], bins=bins, density=True, alpha=0.4,
                label=f'{name} (μ={pts[:,3].mean():.3f})', color=c)

    ax.set_xlabel('Intensity', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Intensity Distribution Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved Intensity Dist: {save_path}")


# ============ 3. 距离-强度关系 ============

def plot_distance_intensity(clean_pts, weather_pts_dict, save_path):
    """
    距离 vs 平均强度曲线 (展示衰减效果)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    dist_bins = np.arange(0, 80, 2)

    def compute_curve(pts):
        d = get_distance(pts)
        means, edges = [], dist_bins
        for i in range(len(edges) - 1):
            mask = (d >= edges[i]) & (d < edges[i + 1])
            means.append(pts[mask, 3].mean() if np.any(mask) else np.nan)
        return (edges[:-1] + edges[1:]) / 2, np.array(means)

    x, y = compute_curve(clean_pts)
    ax.plot(x, y, 'k-', linewidth=2, label='Clean')

    colors = plt.cm.Set1(np.linspace(0, 1, len(weather_pts_dict)))
    for (name, pts), c in zip(weather_pts_dict.items(), colors):
        x, y = compute_curve(pts)
        ax.plot(x, y, '-', color=c, linewidth=1.5, label=name)

    ax.set_xlabel('Distance (m)', fontsize=13)
    ax.set_ylabel('Mean Intensity', fontsize=13)
    ax.set_title('Distance vs Intensity (Signal Attenuation)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved Dist-Int Curve: {save_path}")


# ============ 4. 点密度对比 ============

def plot_point_density(clean_pts, weather_pts_dict, save_path):
    """
    距离-点密度曲线 (展示远处丢点)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    dist_bins = np.arange(0, 80, 2)

    def compute_density(pts):
        d = get_distance(pts)
        counts = np.histogram(d, bins=dist_bins)[0]
        return (dist_bins[:-1] + dist_bins[1:]) / 2, counts

    # 绝对密度
    ax = axes[0]
    x, y = compute_density(clean_pts)
    ax.plot(x, y, 'k-', linewidth=2, label='Clean')
    colors = plt.cm.Set1(np.linspace(0, 1, len(weather_pts_dict)))
    for (name, pts), c in zip(weather_pts_dict.items(), colors):
        x2, y2 = compute_density(pts)
        ax.plot(x2, y2, '-', color=c, linewidth=1.5, label=name)
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Point Count', fontsize=12)
    ax.set_title('Point Density vs Distance', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 相对密度 (相对clean的保留率)
    ax = axes[1]
    x_clean, y_clean = compute_density(clean_pts)
    for (name, pts), c in zip(weather_pts_dict.items(), colors):
        x2, y2 = compute_density(pts)
        # 对齐长度
        min_len = min(len(y_clean), len(y2))
        ratio = y2[:min_len] / (y_clean[:min_len] + 1e-6)
        ax.plot(x_clean[:min_len], ratio, '-', color=c, linewidth=1.5, label=name)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Clean baseline')
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Point Retention Ratio', fontsize=12)
    ax.set_title('Point Retention vs Distance', fontsize=13)
    ax.set_ylim(0, 2.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved Point Density: {save_path}")


# ============ 5. BEV 差异热力图 ============

def plot_bev_diff_heatmap(clean_pts, weather_pts_dict, save_path,
                          xlim=(-40, 40), ylim=(0, 70), resolution=0.5):
    """
    BEV 点密度差异热力图: weather - clean
    红色=噪声点增多, 蓝色=点丢失
    """
    nx = int((xlim[1] - xlim[0]) / resolution)
    ny = int((ylim[1] - ylim[0]) / resolution)

    def pts_to_density(pts):
        mask = (pts[:, 0] > xlim[0]) & (pts[:, 0] < xlim[1]) & \
               (pts[:, 1] > ylim[0]) & (pts[:, 1] < ylim[1])
        p = pts[mask]
        grid = np.zeros((nx, ny))
        xi = ((p[:, 0] - xlim[0]) / resolution).astype(int)
        yi = ((p[:, 1] - ylim[0]) / resolution).astype(int)
        xi = np.clip(xi, 0, nx - 1)
        yi = np.clip(yi, 0, ny - 1)
        for x, y in zip(xi, yi):
            grid[x, y] += 1
        return grid

    clean_grid = pts_to_density(clean_pts)

    n = len(weather_pts_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, pts) in zip(axes, weather_pts_dict.items()):
        weather_grid = pts_to_density(pts)
        diff = weather_grid - clean_grid

        vmax = max(abs(diff.min()), abs(diff.max()), 1)
        im = ax.imshow(diff.T, origin='lower', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='auto',
                       extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
        ax.set_title(f'{name} - Clean\n(red=noise added, blue=points lost)', fontsize=11)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        plt.colorbar(im, ax=ax, label='Point count diff')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved Diff Heatmap: {save_path}")


# ============ 6. 统计指标量化 ============

def compute_statistics(clean_pts, weather_pts):
    """
    计算clean与weather点云之间的多种统计差异指标
    """
    stats_dict = OrderedDict()

    # 基本统计
    stats_dict['clean_num_points'] = len(clean_pts)
    stats_dict['weather_num_points'] = len(weather_pts)
    stats_dict['point_ratio'] = len(weather_pts) / len(clean_pts)

    # 强度统计
    stats_dict['clean_intensity_mean'] = float(clean_pts[:, 3].mean())
    stats_dict['clean_intensity_std'] = float(clean_pts[:, 3].std())
    stats_dict['weather_intensity_mean'] = float(weather_pts[:, 3].mean())
    stats_dict['weather_intensity_std'] = float(weather_pts[:, 3].std())
    stats_dict['intensity_mean_diff'] = float(
        weather_pts[:, 3].mean() - clean_pts[:, 3].mean()
    )

    # 距离统计
    clean_dist = get_distance(clean_pts)
    weather_dist = get_distance(weather_pts)
    stats_dict['clean_mean_distance'] = float(clean_dist.mean())
    stats_dict['weather_mean_distance'] = float(weather_dist.mean())

    # KL散度 (强度分布)
    bins = np.linspace(0, 1, 50)
    p_clean = np.histogram(clean_pts[:, 3], bins=bins, density=True)[0] + 1e-10
    p_weather = np.histogram(weather_pts[:, 3], bins=bins, density=True)[0] + 1e-10
    p_clean /= p_clean.sum()
    p_weather /= p_weather.sum()
    kl_div = float(np.sum(p_clean * np.log(p_clean / p_weather)))
    stats_dict['intensity_KL_divergence'] = kl_div

    # JS散度
    m = 0.5 * (p_clean + p_weather)
    js_div = 0.5 * np.sum(p_clean * np.log(p_clean / m)) + \
             0.5 * np.sum(p_weather * np.log(p_weather / m))
    stats_dict['intensity_JS_divergence'] = float(js_div)

    # Wasserstein距离 (Earth Mover's Distance)
    from scipy.stats import wasserstein_distance
    stats_dict['intensity_wasserstein'] = float(
        wasserstein_distance(clean_pts[:, 3], weather_pts[:, 3])
    )

    # KS检验 (两样本)
    ks_stat, ks_pval = stats.ks_2samp(clean_pts[:, 3], weather_pts[:, 3])
    stats_dict['intensity_KS_statistic'] = float(ks_stat)
    stats_dict['intensity_KS_pvalue'] = float(ks_pval)

    # MMD (Maximum Mean Discrepancy) - 使用采样近似
    stats_dict['MMD_xyz'] = float(_compute_mmd(
        clean_pts[:, :3], weather_pts[:, :3], n_samples=5000
    ))
    stats_dict['MMD_intensity'] = float(_compute_mmd(
        clean_pts[:, 3:4], weather_pts[:, 3:4], n_samples=5000
    ))

    # 距离分布KS检验
    ks_d, ks_d_p = stats.ks_2samp(clean_dist, weather_dist)
    stats_dict['distance_KS_statistic'] = float(ks_d)
    stats_dict['distance_KS_pvalue'] = float(ks_d_p)

    # 近距离噪声比例 (0-5m，正常场景中极少有点)
    near_clean = np.sum(clean_dist < 5) / len(clean_pts)
    near_weather = np.sum(weather_dist < 5) / len(weather_pts)
    stats_dict['near_point_ratio_clean'] = float(near_clean)
    stats_dict['near_point_ratio_weather'] = float(near_weather)

    # 远距离保留率 (>50m)
    far_clean = np.sum(clean_dist > 50)
    far_weather = np.sum(weather_dist > 50)
    stats_dict['far_point_retention'] = float(far_weather / (far_clean + 1e-6))

    return stats_dict


def _compute_mmd(X, Y, n_samples=5000, gamma=None):
    """
    Maximum Mean Discrepancy (RBF kernel)
    用于衡量两个分布之间的差异
    """
    n_x = min(len(X), n_samples)
    n_y = min(len(Y), n_samples)
    X_s = X[np.random.choice(len(X), n_x, replace=False)]
    Y_s = Y[np.random.choice(len(Y), n_y, replace=False)]

    if gamma is None:
        gamma = 1.0 / X_s.shape[1]

    def rbf_kernel(A, B):
        A2 = np.sum(A ** 2, axis=1, keepdims=True)
        B2 = np.sum(B ** 2, axis=1, keepdims=True)
        dist2 = A2 + B2.T - 2 * A @ B.T
        return np.exp(-gamma * dist2)

    K_xx = rbf_kernel(X_s, X_s)
    K_yy = rbf_kernel(Y_s, Y_s)
    K_xy = rbf_kernel(X_s, Y_s)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return max(mmd, 0)


def plot_statistics_table(all_stats, save_path):
    """
    将多种天气的统计指标绘制为表格图
    """
    # 选择关键指标
    key_metrics = [
        'point_ratio', 'intensity_mean_diff',
        'intensity_KL_divergence', 'intensity_JS_divergence',
        'intensity_wasserstein', 'intensity_KS_statistic',
        'MMD_xyz', 'MMD_intensity',
        'near_point_ratio_weather', 'far_point_retention'
    ]

    weather_names = list(all_stats.keys())
    table_data = []
    for metric in key_metrics:
        row = [metric]
        for wname in weather_names:
            val = all_stats[wname].get(metric, 'N/A')
            if isinstance(val, float):
                row.append(f'{val:.4f}')
            else:
                row.append(str(val))
        table_data.append(row)

    fig, ax = plt.subplots(figsize=(4 + 3 * len(weather_names), 
                                     1 + 0.5 * len(key_metrics)))
    ax.axis('tight')
    ax.axis('off')

    col_labels = ['Metric'] + weather_names
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # 表头样式
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', weight='bold')

    plt.title('Domain Shift Statistics: Clean → Weather', fontsize=14, pad=20)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved Stats Table: {save_path}")


# ============ 7. 综合对比报告 ============

def generate_comparison_report(clean_dir, weather_dirs, output_dir,
                               sample_indices=None, num_samples=5):
    """
    生成完整的对比报告

    Args:
        clean_dir: 原始velodyne目录
        weather_dirs: dict {名称: velodyne目录路径}
        output_dir: 输出目录
        sample_indices: 指定帧编号列表，如 ['000001', '000050']
        num_samples: 随机采样帧数 (sample_indices=None时使用)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件列表
    bin_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.bin')])

    if sample_indices is not None:
        selected = [f'{idx}.bin' if not idx.endswith('.bin') else idx
                    for idx in sample_indices]
    else:
        selected = list(np.random.choice(bin_files, min(num_samples, len(bin_files)),
                                         replace=False))

    print(f"Generating comparison report for {len(selected)} frames...")
    print(f"Weather types: {list(weather_dirs.keys())}")

    # ===== 逐帧分析 =====
    all_stats_per_weather = {name: [] for name in weather_dirs}

    for fname in selected:
        frame_id = fname.replace('.bin', '')
        print(f"\n--- Frame {frame_id} ---")

        clean_pts = load_bin(os.path.join(clean_dir, fname))
        weather_pts_dict = OrderedDict()
        for name, wdir in weather_dirs.items():
            wpath = os.path.join(wdir, fname)
            if os.path.exists(wpath):
                weather_pts_dict[name] = load_bin(wpath)

        if not weather_pts_dict:
            continue

        # 可视化
        plot_bev_comparison(
            clean_pts, weather_pts_dict,
            os.path.join(output_dir, f'{frame_id}_bev.png'),
            title_prefix=f"Frame {frame_id} - "
        )

        plot_side_view_comparison(
            clean_pts, weather_pts_dict,
            os.path.join(output_dir, f'{frame_id}_side.png')
        )

        plot_intensity_distribution(
            clean_pts, weather_pts_dict,
            os.path.join(output_dir, f'{frame_id}_intensity_dist.png')
        )

        plot_distance_intensity(
            clean_pts, weather_pts_dict,
            os.path.join(output_dir, f'{frame_id}_dist_int.png')
        )

        plot_point_density(
            clean_pts, weather_pts_dict,
            os.path.join(output_dir, f'{frame_id}_density.png')
        )

        plot_bev_diff_heatmap(
            clean_pts, weather_pts_dict,
            os.path.join(output_dir, f'{frame_id}_diff_heatmap.png')
        )

        # 统计指标
        for name, pts in weather_pts_dict.items():
            s = compute_statistics(clean_pts, pts)
            all_stats_per_weather[name].append(s)

    # ===== 汇总统计 =====
    print("\n===== Aggregated Statistics =====")
    aggregated = OrderedDict()
    for name, stats_list in all_stats_per_weather.items():
        if not stats_list:
            continue
        agg = OrderedDict()
        for key in stats_list[0].keys():
            values = [s[key] for s in stats_list if isinstance(s[key], (int, float))]
            if values:
                agg[key] = float(np.mean(values))
        aggregated[name] = agg

        print(f"\n[{name}]")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}")

    # 统计表格
    if aggregated:
        plot_statistics_table(
            aggregated,
            os.path.join(output_dir, 'statistics_table.png')
        )

    # 保存JSON
    json_path = os.path.join(output_dir, 'statistics.json')
    with open(json_path, 'w') as f:
        json.dump({k: {kk: round(vv, 6) for kk, vv in v.items()}
                    for k, v in aggregated.items()}, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    return aggregated


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Visualize and compare clean vs weather point clouds",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python visualize_weather.py \\
    --clean_dir /data/kitti/training/velodyne \\
    --weather_dirs rain:/data/kitti_weather/rain_moderate/velodyne \\
                   snow:/data/kitti_weather/snow_moderate/velodyne \\
                   fog:/data/kitti_weather/fog_moderate/velodyne \\
    --output_dir /data/kitti_weather/analysis \\
    --num_samples 5

  # 指定帧
  python visualize_weather.py \\
    --clean_dir /data/kitti/training/velodyne \\
    --weather_dirs rain:/data/kitti_weather/rain_moderate/velodyne \\
    --output_dir /data/kitti_weather/analysis \\
    --frames 000001 000050 000100
        """)
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--weather_dirs", nargs='+', required=True,
                        help="格式: name:path, 如 rain:/path/to/rain/velodyne")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--frames", nargs='+', default=None,
                        help="指定帧编号, 如 000001 000050")
    args = parser.parse_args()

    weather_dirs = OrderedDict()
    for item in args.weather_dirs:
        name, path = item.split(':')
        weather_dirs[name] = path

    generate_comparison_report(
        args.clean_dir, weather_dirs, args.output_dir,
        sample_indices=args.frames,
        num_samples=args.num_samples
    )