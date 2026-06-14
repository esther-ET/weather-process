"""
Analyze point-cloud distributions in weather-nus against clean nuScenes.

weather-nus stores simulated point clouds as 4-float .pcd.bin files:
    x, y, z, intensity
clean nuScenes stores LIDAR_TOP .pcd.bin files as 5-float records:
    x, y, z, intensity, ring

The script matches files by basename and compares only x/y/z/intensity.
"""

import argparse
import csv
import json
import os
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm


WEATHER_SPECS = OrderedDict([
    ("rain", ("Rain-Nuscenes", "rain_velodyne")),
    ("snow", ("Snow-Nuscenes", "snow_velodyne")),
    ("fog", ("Fog-Nuscenes", "fog_velodyne")),
])


def load_points(path, dims):
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % dims != 0:
        raise ValueError(f"{path} has {arr.size} floats, not divisible by {dims}")
    return arr.reshape(-1, dims)[:, :4]


def distance(points):
    return np.linalg.norm(points[:, :3], axis=1)


def normalize_intensity(values):
    values = values.astype(np.float32)
    if values.size == 0:
        return values
    vmax = float(np.nanmax(values))
    vmin = float(np.nanmin(values))
    if vmin >= -0.1 and vmax <= 1.5:
        return np.clip(values, 0.0, 1.0)
    if vmin >= 0.0 and vmax <= 255.0:
        return np.clip(values / 255.0, 0.0, 1.0)
    lo, hi = np.nanpercentile(values, [1, 99])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def hist_prob(hist):
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    if total <= 0:
        return np.ones_like(hist) / max(len(hist), 1)
    return hist / total


def kl_js(clean_hist, weather_hist):
    p = hist_prob(clean_hist) + 1e-12
    q = hist_prob(weather_hist) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl = float(np.sum(p * np.log(p / q)))
    js = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    return kl, js


def ks_hist(clean_hist, weather_hist):
    return float(np.max(np.abs(np.cumsum(hist_prob(clean_hist)) - np.cumsum(hist_prob(weather_hist)))))


def wdist_hist(clean_hist, weather_hist, bins):
    centers = 0.5 * (bins[:-1] + bins[1:])
    return float(wasserstein_distance(centers, centers, u_weights=hist_prob(clean_hist), v_weights=hist_prob(weather_hist)))


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def find_clean_file(clean_dir, name):
    path = clean_dir / name
    return path if path.exists() else None


def analyze_dir(clean_dir, weather_dir, max_frames=None):
    files = sorted(weather_dir.glob("*.pcd.bin"))
    if max_frames is not None:
        files = files[:max_frames]

    distance_bins = np.arange(0.0, 102.0, 2.0)
    intensity_bins = np.linspace(0.0, 1.0, 101)
    z_bins = np.linspace(-6.0, 8.0, 71)

    agg = {
        "clean_distance_hist": np.zeros(len(distance_bins) - 1),
        "weather_distance_hist": np.zeros(len(distance_bins) - 1),
        "clean_intensity_hist": np.zeros(len(intensity_bins) - 1),
        "weather_intensity_hist": np.zeros(len(intensity_bins) - 1),
        "clean_z_hist": np.zeros(len(z_bins) - 1),
        "weather_z_hist": np.zeros(len(z_bins) - 1),
    }
    frame_metrics = OrderedDict((k, []) for k in [
        "point_ratio",
        "num_points_clean",
        "num_points_weather",
        "mean_distance_clean",
        "mean_distance_weather",
        "mean_distance_diff",
        "near_ratio_clean",
        "near_ratio_weather",
        "far_retention",
        "intensity_mean_clean",
        "intensity_mean_weather",
        "intensity_mean_diff",
    ])

    matched = 0
    missing = 0
    for weather_path in tqdm(files, desc=weather_dir.parent.name, leave=False):
        clean_path = find_clean_file(clean_dir, weather_path.name)
        if clean_path is None:
            missing += 1
            continue

        clean = load_points(clean_path, 5)
        weather = load_points(weather_path, 4)
        clean_d = distance(clean)
        weather_d = distance(weather)
        clean_i = normalize_intensity(clean[:, 3])
        weather_i = normalize_intensity(weather[:, 3])

        agg["clean_distance_hist"] += np.histogram(clean_d, distance_bins)[0]
        agg["weather_distance_hist"] += np.histogram(weather_d, distance_bins)[0]
        agg["clean_intensity_hist"] += np.histogram(clean_i, intensity_bins)[0]
        agg["weather_intensity_hist"] += np.histogram(weather_i, intensity_bins)[0]
        agg["clean_z_hist"] += np.histogram(clean[:, 2], z_bins)[0]
        agg["weather_z_hist"] += np.histogram(weather[:, 2], z_bins)[0]

        far_clean = int(np.sum(clean_d > 50.0))
        far_weather = int(np.sum(weather_d > 50.0))
        frame_metrics["point_ratio"].append(float(len(weather) / len(clean)) if len(clean) else 0.0)
        frame_metrics["num_points_clean"].append(float(len(clean)))
        frame_metrics["num_points_weather"].append(float(len(weather)))
        frame_metrics["mean_distance_clean"].append(float(np.mean(clean_d)))
        frame_metrics["mean_distance_weather"].append(float(np.mean(weather_d)))
        frame_metrics["mean_distance_diff"].append(float(np.mean(weather_d) - np.mean(clean_d)))
        frame_metrics["near_ratio_clean"].append(float(np.mean(clean_d < 5.0)))
        frame_metrics["near_ratio_weather"].append(float(np.mean(weather_d < 5.0)))
        frame_metrics["far_retention"].append(float(far_weather / far_clean) if far_clean else 0.0)
        frame_metrics["intensity_mean_clean"].append(float(np.mean(clean_i)))
        frame_metrics["intensity_mean_weather"].append(float(np.mean(weather_i)))
        frame_metrics["intensity_mean_diff"].append(float(np.mean(weather_i) - np.mean(clean_i)))
        matched += 1

    summary = OrderedDict()
    summary["num_weather_files"] = len(files)
    summary["num_matched_clean_files"] = matched
    summary["num_missing_clean_files"] = missing
    for key, values in frame_metrics.items():
        for stat_name, stat_value in summarize(values).items():
            summary[f"{key}_{stat_name}"] = stat_value

    for prefix, bins in [("distance", distance_bins), ("intensity", intensity_bins), ("z", z_bins)]:
        clean_hist = agg[f"clean_{prefix}_hist"]
        weather_hist = agg[f"weather_{prefix}_hist"]
        kld, jsd = kl_js(clean_hist, weather_hist)
        summary[f"{prefix}_KL"] = kld
        summary[f"{prefix}_JS"] = jsd
        summary[f"{prefix}_KS_hist"] = ks_hist(clean_hist, weather_hist)
        summary[f"{prefix}_wasserstein_hist"] = wdist_hist(clean_hist, weather_hist, bins)

    histograms = {
        "distance_bins": distance_bins.tolist(),
        "intensity_bins": intensity_bins.tolist(),
        "z_bins": z_bins.tolist(),
        **{k: v.tolist() for k, v in agg.items()},
    }
    return summary, histograms


def plot_level_curves(results, output_dir):
    metrics = [
        ("point_ratio_mean", "Point Ratio"),
        ("mean_distance_diff_mean", "Mean Distance Diff"),
        ("far_retention_mean", "Far >50m Retention"),
        ("intensity_mean_diff_mean", "Intensity Mean Diff"),
        ("distance_JS", "Distance JS"),
        ("intensity_JS", "Intensity JS"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        for weather, level_map in results.items():
            levels = sorted(level_map)
            vals = [level_map[level]["summary"].get(metric, 0.0) for level in levels]
            ax.plot(levels, vals, marker="o", linewidth=2, label=weather)
        ax.set_title(title)
        ax.set_xlabel("weather-nus level")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "weather_nus_level_metrics.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(results, output_path):
    rows = []
    for weather, level_map in results.items():
        for level, payload in level_map.items():
            row = OrderedDict()
            row["weather"] = weather
            row["level"] = level
            row.update(payload["summary"])
            rows.append(row)
    fields = list(rows[0].keys()) if rows else []
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather_nus_root", default="/mnt/nvme0n1p2/data/datasets/weather-nus")
    parser.add_argument("--clean_lidar_dir", default="/mnt/nvme0n1p2/data/datasets/nuScenes/samples/LIDAR_TOP")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    weather_root = Path(args.weather_nus_root)
    clean_dir = Path(args.clean_lidar_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    results = OrderedDict()
    for weather, (top_dir, velo_dir) in WEATHER_SPECS.items():
        results[weather] = OrderedDict()
        for level_dir in sorted((weather_root / top_dir).iterdir()):
            if not level_dir.is_dir():
                continue
            weather_dir = level_dir / velo_dir
            if not weather_dir.is_dir():
                continue
            print(f"[{weather}/{level_dir.name}] {weather_dir}")
            summary, histograms = analyze_dir(clean_dir, weather_dir, args.max_frames)
            results[weather][level_dir.name] = {
                "weather_dir": str(weather_dir),
                "summary": summary,
                "histograms": histograms,
            }

    with open(os.path.join(args.output_dir, "weather_nus_distribution.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_summary_csv(results, os.path.join(args.output_dir, "weather_nus_summary.csv"))
    plot_level_curves(results, args.output_dir)
    print(f"Saved analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
