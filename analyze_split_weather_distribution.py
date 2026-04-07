"""
analyze_split_weather_distribution.py

根据参数日志（如 fog_random_params.txt）和 ImageSets/train.txt,val.txt，
统计并可视化 train/val 两个 split 的模拟参数分布。

示例：
python analyze_split_weather_distribution.py \
  --weather_root /mnt/nvme0n1p2/data/datasets/kitti_weather_random \
  --split_dir /mnt/nvme0n1p2/data/datasets/KITTI2_weather_eval/fog/ImageSets \
  --weather fog \
  --output_dir /mnt/nvme0n1p2/data/datasets/kitti_weather_random/analysis
"""

import argparse
import csv
import json
import math
import os
from collections import Counter


def _normalize_frame_id(name: str) -> str:
    name = name.strip()
    if name.endswith('.bin'):
        name = name[:-4]
    return name.zfill(6)


def read_split_file(path: str):
    frames = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(_normalize_frame_id(line))
    return frames


def read_param_log(path: str):
    rows = {}
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if 'filename' not in (reader.fieldnames or []):
            raise ValueError(f"Invalid param log (missing filename column): {path}")

        for row in reader:
            fname = row.get('filename', '').strip()
            frame_id = _normalize_frame_id(fname)
            rows[frame_id] = row
    return rows


def to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def compute_split_stats(split_name, split_ids, param_map):
    matched = [param_map[fid] for fid in split_ids if fid in param_map]
    missing = [fid for fid in split_ids if fid not in param_map]

    vis = [to_float(r.get('visibility')) for r in matched]
    vis = [v for v in vis if v is not None]

    fog_types = [r.get('fog_type', 'unknown') for r in matched]
    type_counter = Counter(fog_types)

    vis_sorted = sorted(vis)
    n = len(vis_sorted)

    def _percentile(p):
        if n == 0:
            return None
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(vis_sorted[int(k)])
        d0 = vis_sorted[f] * (c - k)
        d1 = vis_sorted[c] * (k - f)
        return float(d0 + d1)

    if n:
        mean_val = float(sum(vis_sorted) / n)
        var = sum((x - mean_val) ** 2 for x in vis_sorted) / n
        std_val = float(math.sqrt(var))
        min_val = float(vis_sorted[0])
        max_val = float(vis_sorted[-1])
    else:
        mean_val = std_val = min_val = max_val = None

    stats = {
        'split': split_name,
        'num_split_frames': len(split_ids),
        'num_matched': len(matched),
        'num_missing': len(missing),
        'missing_examples': missing[:20],
        'visibility': {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'p25': _percentile(25),
            'p50': _percentile(50),
            'p75': _percentile(75),
            'max': max_val,
        },
        'fog_type_counts': dict(type_counter),
    }

    return stats, vis, type_counter


def save_csv(path, stats):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'num_split_frames', 'num_matched', 'num_missing',
                         'vis_mean', 'vis_std', 'vis_min', 'vis_p25', 'vis_p50', 'vis_p75', 'vis_max'])
        for s in stats:
            v = s['visibility']
            writer.writerow([
                s['split'], s['num_split_frames'], s['num_matched'], s['num_missing'],
                v['mean'], v['std'], v['min'], v['p25'], v['p50'], v['p75'], v['max']
            ])


def try_plot(output_dir, weather, split_to_vis, split_to_type_count, allow_skip=False):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        if allow_skip:
            print(f"[WARN] matplotlib not available, skip plotting: {e}")
            return []
        raise RuntimeError(
            "matplotlib is required to generate visualization plots. "
            "Install it or rerun with --no_plot."
        ) from e

    saved = []

    # 1) visibility histogram train vs val
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = [30, 50, 100, 200, 500, 1000, 2000, 3000]

    has_data = False
    for split_name, vis in split_to_vis.items():
        if len(vis) == 0:
            continue
        has_data = True
        ax.hist(vis, bins=bins, alpha=0.55, label=f'{split_name} (n={len(vis)})')

    if has_data:
        ax.set_xlabel('Visibility (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'{weather} random visibility distribution by split')
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        p1 = os.path.join(output_dir, f'{weather}_split_visibility_hist.png')
        fig.savefig(p1, dpi=180)
        saved.append(p1)
    plt.close(fig)

    # 2) fog_type bar chart
    all_types = sorted({k for c in split_to_type_count.values() for k in c.keys()})
    if all_types:
        x = list(range(len(all_types)))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (split_name, counter) in enumerate(split_to_type_count.items()):
            vals = [counter.get(t, 0) for t in all_types]
            offset = (i - 0.5) * width if len(split_to_type_count) == 2 else i * width
            x_shifted = [xi + offset for xi in x]
            ax.bar(x_shifted, vals, width=width, label=split_name)

        ax.set_xticks(x)
        ax.set_xticklabels(all_types)
        ax.set_ylabel('Count')
        ax.set_title(f'{weather} fog_type distribution by split')
        ax.legend()
        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()
        p2 = os.path.join(output_dir, f'{weather}_split_fog_type_bar.png')
        fig.savefig(p2, dpi=180)
        saved.append(p2)
        plt.close(fig)

    # 3) split-level visibility boxplot
    box_data = []
    box_labels = []
    for split_name, vis in split_to_vis.items():
        if len(vis) > 0:
            box_data.append(vis)
            box_labels.append(split_name)
    if box_data:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.boxplot(box_data, labels=box_labels, showmeans=True)
        ax.set_ylabel('Visibility (m)')
        ax.set_title(f'{weather} visibility boxplot by split')
        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout()
        p3 = os.path.join(output_dir, f'{weather}_split_visibility_boxplot.png')
        fig.savefig(p3, dpi=180)
        saved.append(p3)
        plt.close(fig)

    return saved


def main():
    parser = argparse.ArgumentParser(description='Analyze train/val weather simulation distributions from param logs.')
    parser.add_argument('--weather_root', type=str, required=True,
                        help='kitti_weather_random 根目录（其下含 fog_random/ 等目录）')
    parser.add_argument('--split_dir', type=str, required=True,
                        help='ImageSets 目录（包含 train.txt / val.txt）')
    parser.add_argument('--weather', type=str, default='fog', choices=['fog', 'rain', 'snow'],
                        help='分析哪种天气的 random 参数日志')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='统计输出目录，默认: <weather_root>/analysis')
    parser.add_argument('--no_plot', action='store_true',
                        help='不生成可视化图（默认会生成）')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.weather_root, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    param_path = os.path.join(args.weather_root, f'{args.weather}_random', f'{args.weather}_random_params.txt')
    if not os.path.exists(param_path):
        raise FileNotFoundError(f'Cannot find param log: {param_path}')

    train_path = os.path.join(args.split_dir, 'train.txt')
    val_path = os.path.join(args.split_dir, 'val.txt')
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f'train/val split files not found in: {args.split_dir}')

    param_map = read_param_log(param_path)
    train_ids = read_split_file(train_path)
    val_ids = read_split_file(val_path)

    all_stats = []
    split_to_vis = {}
    split_to_type_count = {}

    for split_name, split_ids in [('train', train_ids), ('val', val_ids)]:
        stats, vis, counter = compute_split_stats(split_name, split_ids, param_map)
        all_stats.append(stats)
        split_to_vis[split_name] = vis
        split_to_type_count[split_name] = counter

    json_path = os.path.join(output_dir, f'{args.weather}_split_distribution.json')
    with open(json_path, 'w') as f:
        json.dump({
            'weather': args.weather,
            'param_log': param_path,
            'split_dir': args.split_dir,
            'stats': all_stats,
        }, f, indent=2)

    csv_path = os.path.join(output_dir, f'{args.weather}_split_distribution_summary.csv')
    save_csv(csv_path, all_stats)

    plot_paths = []
    if args.no_plot:
        print('[INFO] --no_plot enabled, skip plotting.')
    else:
        plot_paths = try_plot(output_dir, args.weather, split_to_vis, split_to_type_count, allow_skip=False)

    print('\n=== Split Distribution Summary ===')
    for s in all_stats:
        v = s['visibility']
        print(f"[{s['split']}] matched={s['num_matched']}/{s['num_split_frames']} "
              f"missing={s['num_missing']} vis_mean={v['mean']} vis_p50={v['p50']} "
              f"fog_type={s['fog_type_counts']}")

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    if plot_paths:
        print('Saved plots:')
        for p in plot_paths:
            print(f'  - {p}')


if __name__ == '__main__':
    main()
