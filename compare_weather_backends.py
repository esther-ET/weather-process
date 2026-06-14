"""
Compare external weather simulations against fast heuristic weather simulations.

This script is intentionally aggregate-first: it scans all matched KITTI .bin
frames and computes distribution metrics without the expensive per-frame MMD
used by vis_and_diff.py/domain_analysis.py.
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


WEATHERS = ("rain", "snow", "fog")
METHODS = ("external", "heuristic")


def load_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def distance(pts):
    return np.sqrt(np.sum(pts[:, :3] ** 2, axis=1))


def normalize_intensity(intensity):
    intensity = intensity.astype(np.float32)
    if intensity.size == 0:
        return intensity

    vmin = float(np.nanmin(intensity))
    vmax = float(np.nanmax(intensity))
    if vmax <= 1.5 and vmin >= -0.1:
        return np.clip(intensity, 0.0, 1.0)
    if vmin >= 0.0 and vmax <= 255.0:
        return np.clip(intensity / 255.0, 0.0, 1.0)

    lo, hi = np.quantile(intensity, [0.01, 0.99])
    if hi <= lo:
        return np.zeros_like(intensity)
    return np.clip((intensity - lo) / (hi - lo), 0.0, 1.0)


def safe_div(num, den):
    return float(num / den) if den else 0.0


def hist_prob(hist):
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    if total <= 0:
        return np.ones_like(hist) / max(len(hist), 1)
    return hist / total


def kl_js_from_hist(clean_hist, weather_hist):
    p = hist_prob(clean_hist) + 1e-12
    q = hist_prob(weather_hist) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl = float(np.sum(p * np.log(p / q)))
    js = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))
    return kl, js


def cdf_ks_from_hist(clean_hist, weather_hist):
    p = np.cumsum(hist_prob(clean_hist))
    q = np.cumsum(hist_prob(weather_hist))
    return float(np.max(np.abs(p - q)))


def wasserstein_from_hist(clean_hist, weather_hist, bin_edges):
    p = hist_prob(clean_hist)
    q = hist_prob(weather_hist)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return float(wasserstein_distance(centers, centers, u_weights=p, v_weights=q))


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


def list_bins(path):
    return sorted(f for f in os.listdir(path) if f.endswith(".bin"))


def analyze_pair(clean_dir, weather_dir, max_frames=None):
    clean_files = set(list_bins(clean_dir))
    weather_files = set(list_bins(weather_dir))
    files = sorted(clean_files & weather_files)
    if max_frames is not None:
        files = files[:max_frames]

    intensity_bins = np.linspace(0.0, 1.0, 101)
    distance_bins = np.arange(0.0, 82.0, 2.0)
    z_bins = np.linspace(-4.0, 4.0, 81)

    agg = {
        "clean_intensity_hist": np.zeros(len(intensity_bins) - 1, dtype=np.float64),
        "weather_intensity_hist": np.zeros(len(intensity_bins) - 1, dtype=np.float64),
        "clean_distance_hist": np.zeros(len(distance_bins) - 1, dtype=np.float64),
        "weather_distance_hist": np.zeros(len(distance_bins) - 1, dtype=np.float64),
        "clean_z_hist": np.zeros(len(z_bins) - 1, dtype=np.float64),
        "weather_z_hist": np.zeros(len(z_bins) - 1, dtype=np.float64),
    }

    metrics = {
        "point_ratio": [],
        "num_points_delta": [],
        "near_ratio_clean": [],
        "near_ratio_weather": [],
        "far_retention": [],
        "mean_distance_clean": [],
        "mean_distance_weather": [],
        "mean_distance_diff": [],
        "intensity_mean_clean": [],
        "intensity_mean_weather": [],
        "intensity_mean_diff": [],
        "intensity_std_clean": [],
        "intensity_std_weather": [],
    }

    for fname in tqdm(files, desc=Path(weather_dir).parent.name, leave=False):
        clean = load_bin(os.path.join(clean_dir, fname))
        weather = load_bin(os.path.join(weather_dir, fname))

        clean_d = distance(clean)
        weather_d = distance(weather)
        clean_i = normalize_intensity(clean[:, 3])
        weather_i = normalize_intensity(weather[:, 3])

        agg["clean_intensity_hist"] += np.histogram(clean_i, bins=intensity_bins)[0]
        agg["weather_intensity_hist"] += np.histogram(weather_i, bins=intensity_bins)[0]
        agg["clean_distance_hist"] += np.histogram(clean_d, bins=distance_bins)[0]
        agg["weather_distance_hist"] += np.histogram(weather_d, bins=distance_bins)[0]
        agg["clean_z_hist"] += np.histogram(clean[:, 2], bins=z_bins)[0]
        agg["weather_z_hist"] += np.histogram(weather[:, 2], bins=z_bins)[0]

        far_clean = int(np.sum(clean_d > 50.0))
        far_weather = int(np.sum(weather_d > 50.0))

        metrics["point_ratio"].append(safe_div(len(weather), len(clean)))
        metrics["num_points_delta"].append(float(len(weather) - len(clean)))
        metrics["near_ratio_clean"].append(float(np.mean(clean_d < 5.0)))
        metrics["near_ratio_weather"].append(float(np.mean(weather_d < 5.0)))
        metrics["far_retention"].append(safe_div(far_weather, far_clean))
        metrics["mean_distance_clean"].append(float(np.mean(clean_d)))
        metrics["mean_distance_weather"].append(float(np.mean(weather_d)))
        metrics["mean_distance_diff"].append(float(np.mean(weather_d) - np.mean(clean_d)))
        metrics["intensity_mean_clean"].append(float(np.mean(clean_i)))
        metrics["intensity_mean_weather"].append(float(np.mean(weather_i)))
        metrics["intensity_mean_diff"].append(float(np.mean(weather_i) - np.mean(clean_i)))
        metrics["intensity_std_clean"].append(float(np.std(clean_i)))
        metrics["intensity_std_weather"].append(float(np.std(weather_i)))

    intensity_kl, intensity_js = kl_js_from_hist(
        agg["clean_intensity_hist"], agg["weather_intensity_hist"]
    )
    distance_kl, distance_js = kl_js_from_hist(
        agg["clean_distance_hist"], agg["weather_distance_hist"]
    )
    z_kl, z_js = kl_js_from_hist(agg["clean_z_hist"], agg["weather_z_hist"])

    summary = OrderedDict()
    summary["num_frames"] = len(files)
    for key, values in metrics.items():
        for stat_name, stat_value in summarize(values).items():
            summary[f"{key}_{stat_name}"] = stat_value

    summary["intensity_KL"] = intensity_kl
    summary["intensity_JS"] = intensity_js
    summary["intensity_KS_hist"] = cdf_ks_from_hist(
        agg["clean_intensity_hist"], agg["weather_intensity_hist"]
    )
    summary["intensity_wasserstein_hist"] = wasserstein_from_hist(
        agg["clean_intensity_hist"], agg["weather_intensity_hist"], intensity_bins
    )
    summary["distance_KL"] = distance_kl
    summary["distance_JS"] = distance_js
    summary["distance_KS_hist"] = cdf_ks_from_hist(
        agg["clean_distance_hist"], agg["weather_distance_hist"]
    )
    summary["distance_wasserstein_hist"] = wasserstein_from_hist(
        agg["clean_distance_hist"], agg["weather_distance_hist"], distance_bins
    )
    summary["z_KL"] = z_kl
    summary["z_JS"] = z_js
    summary["z_KS_hist"] = cdf_ks_from_hist(agg["clean_z_hist"], agg["weather_z_hist"])

    retention = agg["weather_distance_hist"] / (agg["clean_distance_hist"] + 1e-6)
    histograms = {
        "intensity_bins": intensity_bins.tolist(),
        "distance_bins": distance_bins.tolist(),
        "z_bins": z_bins.tolist(),
        "clean_intensity_hist": agg["clean_intensity_hist"].tolist(),
        "weather_intensity_hist": agg["weather_intensity_hist"].tolist(),
        "clean_distance_hist": agg["clean_distance_hist"].tolist(),
        "weather_distance_hist": agg["weather_distance_hist"].tolist(),
        "clean_z_hist": agg["clean_z_hist"].tolist(),
        "weather_z_hist": agg["weather_z_hist"].tolist(),
        "distance_retention": retention.tolist(),
    }
    return summary, histograms


def write_summary_csv(results, output_path):
    rows = []
    for method, weather_map in results.items():
        for weather, payload in weather_map.items():
            row = OrderedDict()
            row["method"] = method
            row["weather"] = weather
            row.update(payload["summary"])
            rows.append(row)

    fields = list(rows[0].keys()) if rows else []
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_csv(results, output_path):
    fields = [
        "weather",
        "metric",
        "external",
        "heuristic",
        "heuristic_minus_external",
        "heuristic_div_external",
    ]
    metrics = [
        "point_ratio_mean",
        "near_ratio_weather_mean",
        "far_retention_mean",
        "mean_distance_diff_mean",
        "intensity_mean_diff_mean",
        "intensity_JS",
        "intensity_KS_hist",
        "intensity_wasserstein_hist",
        "distance_JS",
        "distance_KS_hist",
        "distance_wasserstein_hist",
        "z_JS",
    ]
    rows = []
    for weather in WEATHERS:
        ext = results["external"][weather]["summary"]
        heu = results["heuristic"][weather]["summary"]
        for metric in metrics:
            e = float(ext.get(metric, 0.0))
            h = float(heu.get(metric, 0.0))
            rows.append({
                "weather": weather,
                "metric": metric,
                "external": e,
                "heuristic": h,
                "heuristic_minus_external": h - e,
                "heuristic_div_external": h / e if abs(e) > 1e-12 else "",
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_bars(results, output_path):
    metrics = [
        ("point_ratio_mean", "Point Ratio"),
        ("near_ratio_weather_mean", "Near <5m Ratio"),
        ("far_retention_mean", "Far >50m Retention"),
        ("intensity_JS", "Intensity JS"),
        ("distance_JS", "Distance JS"),
        ("intensity_wasserstein_hist", "Intensity W-dist"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    x = np.arange(len(WEATHERS))
    width = 0.36
    for ax, (metric, title) in zip(axes, metrics):
        ext = [results["external"][w]["summary"].get(metric, 0.0) for w in WEATHERS]
        heu = [results["heuristic"][w]["summary"].get(metric, 0.0) for w in WEATHERS]
        ax.bar(x - width / 2, ext, width, label="external")
        ax.bar(x + width / 2, heu, width, label="heuristic")
        ax.set_xticks(x)
        ax.set_xticklabels(WEATHERS)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_retention_curves(results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, weather in zip(axes, WEATHERS):
        for method in METHODS:
            h = results[method][weather]["histograms"]
            bins = np.asarray(h["distance_bins"])
            centers = 0.5 * (bins[:-1] + bins[1:])
            ax.plot(centers, h["distance_retention"], label=method, linewidth=2)
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.4)
        ax.set_title(weather)
        ax.set_xlabel("Distance (m)")
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 2)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Weather / Clean Count")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_overlays(results, output_dir):
    for kind, bins_key, clean_key, weather_key, xlabel in [
        ("intensity", "intensity_bins", "clean_intensity_hist", "weather_intensity_hist", "Intensity"),
        ("distance", "distance_bins", "clean_distance_hist", "weather_distance_hist", "Distance (m)"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, weather in zip(axes, WEATHERS):
            for method in METHODS:
                h = results[method][weather]["histograms"]
                bins = np.asarray(h[bins_key])
                centers = 0.5 * (bins[:-1] + bins[1:])
                clean_p = hist_prob(h[clean_key])
                weather_p = hist_prob(h[weather_key])
                if method == "external":
                    ax.plot(centers, clean_p, color="black", alpha=0.5, linewidth=1.5, label="clean")
                ax.plot(centers, weather_p, linewidth=2, label=method)
            ax.set_title(weather)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Probability")
            ax.grid(True, alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{kind}_distribution_overlay.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--external_root", required=True)
    parser.add_argument("--heuristic_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    roots = {
        "external": args.external_root,
        "heuristic": args.heuristic_root,
    }
    results = OrderedDict((m, OrderedDict()) for m in METHODS)

    for method in METHODS:
        for weather in WEATHERS:
            weather_dir = os.path.join(roots[method], f"{weather}_random", "velodyne")
            if not os.path.isdir(weather_dir):
                raise FileNotFoundError(weather_dir)
            print(f"[{method}/{weather}] {weather_dir}")
            summary, histograms = analyze_pair(args.clean_dir, weather_dir, args.max_frames)
            results[method][weather] = {
                "weather_dir": weather_dir,
                "summary": summary,
                "histograms": histograms,
            }

    json_path = os.path.join(args.output_dir, "weather_backend_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    write_summary_csv(results, os.path.join(args.output_dir, "weather_backend_summary.csv"))
    write_comparison_csv(results, os.path.join(args.output_dir, "weather_backend_external_vs_heuristic.csv"))
    plot_metric_bars(results, os.path.join(args.output_dir, "metric_bars_external_vs_heuristic.png"))
    plot_retention_curves(results, os.path.join(args.output_dir, "distance_retention_curves.png"))
    plot_distribution_overlays(results, args.output_dir)

    print(f"Saved analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
