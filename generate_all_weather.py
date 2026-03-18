"""
generate_all_weather.py - 一键生成（支持固定/随机参数）
"""

import os
import argparse
from rain_simulation import process_kitti_rain
from snow_simulation import process_kitti_snow
from fog_simulation import process_kitti_fog


FIXED_PRESETS = {
    'rain': {
        'light': {'rain_rate': 2.5}, 'moderate': {'rain_rate': 10.0},
        'heavy': {'rain_rate': 30.0}, 'extreme': {'rain_rate': 60.0},
    },
    'snow': {
        'light': {'snowfall_rate': 0.5}, 'moderate': {'snowfall_rate': 2.5},
        'heavy': {'snowfall_rate': 5.0}, 'extreme': {'snowfall_rate': 10.0},
    },
    'fog': {
        'light': {'visibility': 1000.0, 'fog_type': 'uniform'},
        'moderate': {'visibility': 500.0, 'fog_type': 'uniform'},
        'heavy': {'visibility': 200.0, 'fog_type': 'inhomogeneous'},
        'extreme': {'visibility': 50.0, 'fog_type': 'inhomogeneous'},
    }
}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # 固定参数 moderate 等级
  python generate_all_weather.py \\
    --input_dir /data/kitti/training/velodyne \\
    --output_dir /data/kitti_weather \\
    --severities moderate

  # 每帧随机参数 (log-uniform分布)
  python generate_all_weather.py \\
    --input_dir /data/kitti/training/velodyne \\
    --output_dir /data/kitti_weather_random \\
    --random_params --sample_mode log --seed 42

  # 每帧随机参数 (按等级类别采样)
  python generate_all_weather.py \\
    --input_dir /data/kitti/training/velodyne \\
    --output_dir /data/kitti_weather_random \\
    --random_params --sample_mode category
        """)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weather", nargs='+', default=['rain', 'snow', 'fog'],
                        choices=['rain', 'snow', 'fog'])

    # 固定参数模式
    parser.add_argument("--severities", nargs='+', default=['moderate'],
                        choices=['light', 'moderate', 'heavy', 'extreme'],
                        help="固定参数模式的严重程度等级")

    # 随机参数模式
    parser.add_argument("--random_params", action='store_true',
                        help="启用随机参数模式（每帧独立采样）")
    parser.add_argument("--sample_mode", type=str, default='log',
                        choices=['uniform', 'log', 'category'],
                        help="随机采样分布:\n"
                             "  uniform  - 均匀分布\n"
                             "  log      - 对数均匀 (推荐，中低强度概率更大)\n"
                             "  category - 先选等级再在等级内均匀采样")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，确保可复现")
    args = parser.parse_args()

    if args.random_params:
        # ===== 随机参数模式 =====
        for w in args.weather:
            out = os.path.join(args.output_dir, f"{w}_random", "velodyne")
            print(f"\n{'='*50}")
            print(f"{w} - RANDOM params (mode={args.sample_mode})")
            print(f"  rain_rate  ∈ [1, 80] mm/h")
            print(f"  snowfall   ∈ [0.5, 10] mm/h")
            print(f"  visibility ∈ [30, 2000] m")
            print(f"{'='*50}")

            if w == 'rain':
                process_kitti_rain(args.input_dir, out,
                                   random_params=True,
                                   sample_mode=args.sample_mode,
                                   seed=args.seed)
            elif w == 'snow':
                process_kitti_snow(args.input_dir, out,
                                   random_params=True,
                                   sample_mode=args.sample_mode,
                                   seed=args.seed)
            elif w == 'fog':
                process_kitti_fog(args.input_dir, out,
                                  random_params=True,
                                  sample_mode=args.sample_mode,
                                  seed=args.seed)
    else:
        # ===== 固定参数模式 =====
        for w in args.weather:
            for s in args.severities:
                if s not in FIXED_PRESETS[w]:
                    continue
                out = os.path.join(args.output_dir, f"{w}_{s}", "velodyne")
                params = FIXED_PRESETS[w][s]
                print(f"\n{'='*50}")
                print(f"{w} - {s}: {params}")
                print(f"{'='*50}")

                if w == 'rain':
                    process_kitti_rain(args.input_dir, out, **params)
                elif w == 'snow':
                    process_kitti_snow(args.input_dir, out, **params)
                elif w == 'fog':
                    process_kitti_fog(args.input_dir, out, **params)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()