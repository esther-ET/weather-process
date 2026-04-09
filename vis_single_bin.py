"""
vis_single_bin.py - 单个/成对 KITTI bin 可视化工具

用途:
1. 只可视化一个 bin 文件
2. 对比两个 bin 文件（例如 clean vs rain）并输出统计

cd /home/ubuntu/SWW/code/weather-process
# 2) 对比两张 bin（例如 clean vs rain）
python vis_single_bin.py \
  --bin_path /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne/000022.bin \
  --output_dir /home/ubuntu/SWW/analysis/tmp \
  --compare_bin /mnt/nvme0n1p2/data/datasets/kitti_weather_random/rain_random/velodyne/000022.bin \
  --compare_name rain

"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from utils import load_kitti_points
from vis_and_diff import (
    compute_statistics,
    plot_bev_comparison,
    plot_distance_intensity,
    plot_intensity_distribution,
    plot_point_density,
    plot_side_view_comparison,
)


def _plot_single_bev(points, save_path, title, xlim=(-40, 40), ylim=(0, 70)):
    mask = (
        (points[:, 0] > xlim[0]) & (points[:, 0] < xlim[1]) &
        (points[:, 1] > ylim[0]) & (points[:, 1] < ylim[1])
    )
    p = points[mask]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    sc = ax.scatter(p[:, 1], p[:, 0], c=p[:, 3], cmap='viridis', s=0.12, vmin=0, vmax=1, alpha=0.8)
    ax.set_xlim(ylim)
    ax.set_ylim(xlim)
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    ax.set_aspect('equal')
    ax.set_title(f"{title} BEV ({len(p):,} pts)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Intensity')
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _plot_single_side(points, save_path, title, xlim=(0, 70), zlim=(-3, 3)):
    mask = (
        (points[:, 0] > xlim[0]) & (points[:, 0] < xlim[1]) &
        (points[:, 2] > zlim[0]) & (points[:, 2] < zlim[1])
    )
    p = points[mask]

    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    sc = ax.scatter(p[:, 0], p[:, 2], c=p[:, 3], cmap='viridis', s=0.12, vmin=0, vmax=1, alpha=0.7)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title(f"{title} Side View ({len(p):,} pts)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Intensity')
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _plot_single_hist(points, save_path, title):
    dist = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(points[:, 3], bins=100, color='#2E86AB', alpha=0.85)
    axes[0].set_title('Intensity Histogram')
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(dist, bins=120, color='#F18F01', alpha=0.85)
    axes[1].set_title('Distance Histogram')
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def _save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize one KITTI .bin directly, optional pairwise comparison.',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python vis_single_bin.py --bin_path /data/000123.bin --output_dir /tmp/vis_123\n\n"
            "  python vis_single_bin.py --bin_path /data/clean/000123.bin "
            "--compare_bin /data/rain/000123.bin --compare_name rain "
            "--output_dir /tmp/vis_123_cmp"
        ),
    )
    parser.add_argument('--bin_path', type=str, required=True, help='主 bin 文件路径')
    parser.add_argument('--compare_bin', type=str, default=None, help='对比 bin 文件路径（可选）')
    parser.add_argument('--compare_name', type=str, default='compare', help='对比名称（默认 compare）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--name', type=str, default=None, help='图标题名称（默认取 bin 文件名）')
    parser.add_argument('--xlim', nargs=2, type=float, default=[-40, 40], metavar=('XMIN', 'XMAX'))
    parser.add_argument('--ylim', nargs=2, type=float, default=[0, 70], metavar=('YMIN', 'YMAX'))
    parser.add_argument('--zlim', nargs=2, type=float, default=[-3, 3], metavar=('ZMIN', 'ZMAX'))
    args = parser.parse_args()

    bin_path = Path(args.bin_path).expanduser()
    compare_path = Path(args.compare_bin).expanduser() if args.compare_bin else None
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bin_path.exists():
        raise FileNotFoundError(f'bin file not found: {bin_path}')
    if compare_path is not None and not compare_path.exists():
        raise FileNotFoundError(f'compare bin not found: {compare_path}')

    title_name = args.name if args.name else bin_path.stem

    src = load_kitti_points(str(bin_path))

    meta = {
        'source_bin': str(bin_path),
        'num_points': int(src.shape[0]),
        'intensity_mean': float(src[:, 3].mean()),
        'intensity_std': float(src[:, 3].std()),
    }

    if compare_path is None:
        _plot_single_bev(
            src,
            str(output_dir / 'single_bev.png'),
            title_name,
            xlim=tuple(args.xlim),
            ylim=tuple(args.ylim),
        )
        _plot_single_side(
            src,
            str(output_dir / 'single_side.png'),
            title_name,
            xlim=(0, args.ylim[1]),
            zlim=tuple(args.zlim),
        )
        _plot_single_hist(
            src,
            str(output_dir / 'single_hist.png'),
            title_name,
        )
        _save_json(str(output_dir / 'single_stats.json'), meta)
        print(f'[OK] Saved single-bin visualizations to {output_dir}')
        return

    cmp_pts = load_kitti_points(str(compare_path))
    weather = {args.compare_name: cmp_pts}

    plot_bev_comparison(
        src,
        weather,
        str(output_dir / 'pair_bev.png'),
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
        title_prefix='',
    )
    plot_side_view_comparison(
        src,
        weather,
        str(output_dir / 'pair_side.png'),
        xlim=(0, args.ylim[1]),
        zlim=tuple(args.zlim),
    )
    plot_intensity_distribution(src, weather, str(output_dir / 'pair_intensity.png'))
    plot_distance_intensity(src, weather, str(output_dir / 'pair_dist_intensity.png'))
    plot_point_density(src, weather, str(output_dir / 'pair_density.png'))

    stats = compute_statistics(src, cmp_pts)
    result = {
        'source_bin': str(bin_path),
        'compare_bin': str(compare_path),
        'compare_name': args.compare_name,
        'statistics': {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in stats.items()},
    }
    _save_json(str(output_dir / 'pair_stats.json'), result)
    print(f'[OK] Saved pairwise visualizations to {output_dir}')


if __name__ == '__main__':
    main()
