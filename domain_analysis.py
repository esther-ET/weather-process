"""
domain_analysis.py - 证明不同天气构成不同Domain的分析
用于论文中证明domain shift的存在

包含:
1. A-distance (proxy A-distance / PAD)
2. 特征空间t-SNE可视化
3. 分类器交叉验证 (domain classifier)
4. Fréchet Point Cloud Distance (FPD)
5. 雷达图 (多维度domain差异总览)
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import OrderedDict
import json


def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def get_distance(pts):
    return np.sqrt(np.sum(pts[:, :3] ** 2, axis=1))


# ============ 特征提取 ============

def extract_frame_features(pts, num_dist_bins=20, num_int_bins=20):
    """
    从单帧点云提取统计特征向量 (用于domain分类)

    特征包括:
    - 点数量
    - 强度统计 (mean, std, median, skewness, kurtosis)
    - 距离统计 (mean, std, median)
    - 距离直方图 (归一化)
    - 强度直方图 (归一化)
    - 近距离点占比 (<5m, <10m, <20m)
    - 远距离点占比 (>50m, >70m)
    - z坐标统计
    """
    from scipy.stats import skew, kurtosis

    dist = get_distance(pts)
    features = []

    # 基本
    features.append(len(pts))  # 点数

    # 强度统计
    intensity = pts[:, 3]
    features.extend([
        intensity.mean(), intensity.std(), np.median(intensity),
        float(skew(intensity)), float(kurtosis(intensity)),
        np.percentile(intensity, 10), np.percentile(intensity, 90),
    ])

    # 距离统计
    features.extend([
        dist.mean(), dist.std(), np.median(dist),
    ])

    # 距离直方图
    d_hist = np.histogram(dist, bins=np.linspace(0, 80, num_dist_bins + 1),
                          density=True)[0]
    features.extend(d_hist.tolist())

    # 强度直方图
    i_hist = np.histogram(intensity, bins=np.linspace(0, 1, num_int_bins + 1),
                          density=True)[0]
    features.extend(i_hist.tolist())

    # 距离区间占比
    for thresh in [5, 10, 20]:
        features.append(np.mean(dist < thresh))
    for thresh in [50, 70]:
        features.append(np.mean(dist > thresh))

    # z坐标
    features.extend([pts[:, 2].mean(), pts[:, 2].std()])

    return np.array(features, dtype=np.float64)


def extract_features_from_dir(velodyne_dir, max_frames=200):
    """从目录中提取所有帧的特征"""
    bins = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
    bins = bins[:max_frames]

    features = []
    for fname in bins:
        pts = load_bin(os.path.join(velodyne_dir, fname))
        feat = extract_frame_features(pts)
        features.append(feat)

    return np.array(features)


# ============ 1. Proxy A-distance ============

def compute_proxy_a_distance(features_source, features_target):
    """
    Proxy A-distance (PAD):
    训练线性SVM区分两个domain，PAD = 2(1 - 2*error)
    
    PAD ∈ [0, 2]:
      0 = 完全相同的分布 (不可区分)
      2 = 完全不同的分布 (完美区分)
    
    论文引用: Ben-David et al., "A theory of learning from different domains", 2010
    """
    X = np.vstack([features_source, features_target])
    y = np.concatenate([
        np.zeros(len(features_source)),
        np.ones(len(features_target))
    ])

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 5折交叉验证
    svm = SVC(kernel='linear', C=1.0)
    scores = cross_val_score(svm, X, y, cv=5, scoring='accuracy')
    error = 1 - scores.mean()

    pad = 2 * (1 - 2 * error)
    return max(pad, 0), scores.mean()


def plot_a_distance_matrix(domain_features, save_path):
    """
    绘制所有domain对之间的PAD矩阵热力图
    """
    names = list(domain_features.keys())
    n = len(names)
    pad_matrix = np.zeros((n, n))
    acc_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                pad_matrix[i, j] = 0
                acc_matrix[i, j] = 0.5
            elif i < j:
                pad, acc = compute_proxy_a_distance(
                    domain_features[names[i]],
                    domain_features[names[j]]
                )
                pad_matrix[i, j] = pad
                pad_matrix[j, i] = pad
                acc_matrix[i, j] = acc
                acc_matrix[j, i] = acc
                print(f"  PAD({names[i]} ↔ {names[j]})"
                      f" = {pad:.4f} (classifier acc={acc:.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PAD matrix
    im1 = axes[0].imshow(pad_matrix, cmap='YlOrRd', vmin=0, vmax=2)
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_yticklabels(names)
    axes[0].set_title('Proxy A-distance (PAD)\n0=same, 2=completely different', fontsize=12)
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f'{pad_matrix[i, j]:.3f}',
                        ha='center', va='center', fontsize=10,
                        color='white' if pad_matrix[i, j] > 1 else 'black')
    plt.colorbar(im1, ax=axes[0])

    # Classifier accuracy
    im2 = axes[1].imshow(acc_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_yticklabels(names)
    axes[1].set_title('Domain Classifier Accuracy\n0.5=indistinguishable, 1.0=perfectly separable', fontsize=12)
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f'{acc_matrix[i, j]:.3f}',
                        ha='center', va='center', fontsize=10,
                        color='white' if acc_matrix[i, j] > 0.8 else 'black')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved A-distance matrix: {save_path}")

    return pad_matrix, acc_matrix


# ============ 2. t-SNE 可视化 ============

def plot_tsne_domains(domain_features, save_path, perplexity=30):
    """
    t-SNE将各domain的帧级特征投影到2D空间
    不同domain应形成不同的聚类
    """
    all_features = []
    all_labels = []
    all_names = []

    for name, feats in domain_features.items():
        all_features.append(feats)
        all_labels.extend([name] * len(feats))
        all_names.append(name)

    X = np.vstack(all_features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA降维到50维（加速t-SNE）
    if X_scaled.shape[1] > 50:
        pca = PCA(n_components=50)
        X_scaled = pca.fit_transform(X_scaled)

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(X) // 4),
                random_state=42, n_iter=1000)
    X_2d = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_names)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

    offset = 0
    for i, (name, feats) in enumerate(domain_features.items()):
        n = len(feats)
        ax.scatter(X_2d[offset:offset + n, 0], X_2d[offset:offset + n, 1],
                   c=[colors[i]], marker=markers[i % len(markers)],
                   s=40, alpha=0.7, label=name, edgecolors='white', linewidths=0.5)
        offset += n

    ax.set_title('t-SNE: Frame-level Feature Space\n'
                 '(Each point = one frame, separated clusters = domain shift)',
                 fontsize=13)
    ax.legend(fontsize=11, markerscale=2)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved t-SNE: {save_path}")


# ============ 3. 雷达图 ============

def plot_radar_chart(all_stats, save_path):
    """
    多维度domain差异雷达图
    展示不同天气在各指标上的domain shift程度
    """
    # 选择关键差异指标（都归一化到 [0, 1]）
    metrics_config = {
        'Intensity Shift': ('intensity_wasserstein', 0, 0.3),
        'KL Divergence': ('intensity_KL_divergence', 0, 2.0),
        'Point Loss': ('point_ratio', 0.5, 1.5),  # 越偏离1越大
        'Near Noise': ('near_point_ratio_weather', 0, 0.1),
        'Far Loss': ('far_point_retention', 0, 1.0),  # 越小loss越大
        'MMD (xyz)': ('MMD_xyz', 0, 0.1),
        'MMD (int)': ('MMD_intensity', 0, 0.1),
        'KS Statistic': ('intensity_KS_statistic', 0, 1.0),
    }

    metric_names = list(metrics_config.keys())
    n_metrics = len(metric_names)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = plt.cm.Set1(np.linspace(0, 1, len(all_stats)))

    for (wname, stats_dict), color in zip(all_stats.items(), colors):
        values = []
        for mname, (key, vmin, vmax) in metrics_config.items():
            raw = stats_dict.get(key, 0)
            if key == 'point_ratio':
                raw = abs(1 - raw)  # 转为"偏离度"
            elif key == 'far_point_retention':
                raw = 1 - raw  # 转为"丢失率"
            normalized = np.clip((raw - vmin) / (vmax - vmin + 1e-10), 0, 1)
            values.append(normalized)
        values += values[:1]

        ax.plot(angles, values, '-o', color=color, linewidth=2,
                label=wname, markersize=6)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Domain Shift Radar Chart\n(larger area = greater domain gap)',
                 fontsize=14, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved Radar Chart: {save_path}")


# ============ 4. 综合Domain分析报告 ============

def run_domain_analysis(clean_dir, weather_dirs, output_dir, max_frames=200):
    """
    完整domain分析流程

    Args:
        clean_dir: 原始velodyne目录
        weather_dirs: dict {名称: velodyne路径}
        output_dir: 输出目录
        max_frames: 最大采样帧数
    """
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Domain Shift Analysis")
    print("="*60)

    # 1. 提取所有domain的特征
    print("\n[1/4] Extracting frame-level features...")
    domain_features = OrderedDict()
    domain_features['Clean'] = extract_features_from_dir(clean_dir, max_frames)
    print(f"  Clean: {len(domain_features['Clean'])} frames")

    for name, wdir in weather_dirs.items():
        domain_features[name] = extract_features_from_dir(wdir, max_frames)
        print(f"  {name}: {len(domain_features[name])} frames")

    # 2. Proxy A-distance
    print("\n[2/4] Computing Proxy A-distance...")
    pad_matrix, acc_matrix = plot_a_distance_matrix(
        domain_features,
        os.path.join(output_dir, 'a_distance_matrix.png')
    )

    # 3. t-SNE可视化
    print("\n[3/4] Running t-SNE visualization...")
    plot_tsne_domains(
        domain_features,
        os.path.join(output_dir, 'tsne_domains.png')
    )

    # 4. 逐帧统计 + 雷达图
    print("\n[4/4] Computing per-domain statistics...")
    from vis_and_diff import compute_statistics

    aggregated_stats = OrderedDict()
    for wname, wdir in weather_dirs.items():
        bins = sorted([f for f in os.listdir(wdir) if f.endswith('.bin')])[:max_frames]
        stats_list = []
        for fname in bins:
            clean_path = os.path.join(clean_dir, fname)
            weather_path = os.path.join(wdir, fname)
            if os.path.exists(clean_path) and os.path.exists(weather_path):
                c = load_bin(clean_path)
                w = load_bin(weather_path)
                stats_list.append(compute_statistics(c, w))

        if stats_list:
            agg = {}
            for key in stats_list[0]:
                vals = [s[key] for s in stats_list if isinstance(s[key], (int, float))]
                if vals:
                    agg[key] = float(np.mean(vals))
            aggregated_stats[wname] = agg

    if aggregated_stats:
        plot_radar_chart(aggregated_stats,
                         os.path.join(output_dir, 'domain_radar.png'))

    # 保存完整结果
    results = {
        'pad_matrix': {
            'names': list(domain_features.keys()),
            'values': pad_matrix.tolist()
        },
        'classifier_accuracy': acc_matrix.tolist(),
        'per_domain_stats': {
            k: {kk: round(vv, 6) for kk, vv in v.items()}
            for k, v in aggregated_stats.items()
        }
    }
    with open(os.path.join(output_dir, 'domain_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Domain shift analysis between clean and weather point clouds",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example:
  python domain_analysis.py \\
    --clean_dir /data/kitti/training/velodyne \\
    --weather_dirs rain:/data/kitti_weather/rain_moderate/velodyne \\
                   snow:/data/kitti_weather/snow_moderate/velodyne \\
                   fog:/data/kitti_weather/fog_moderate/velodyne \\
    --output_dir /data/analysis/domain_shift \\
    --max_frames 200
        """)
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--weather_dirs", nargs='+', required=True,
                        help="name:path pairs")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=200)
    args = parser.parse_args()

    weather_dirs = OrderedDict()
    for item in args.weather_dirs:
        name, path = item.split(':')
        weather_dirs[name] = path

    run_domain_analysis(args.clean_dir, weather_dirs, args.output_dir, args.max_frames)