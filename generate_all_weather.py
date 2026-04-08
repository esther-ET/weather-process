"""
generate_all_weather.py - 增强版天气数据生成
支持:
  - 固定参数 / 随机参数
  - 指定帧号 / 帧范围 / 帧列表文件
  - 单天气 / 混合天气模式
  - 并行处理
  - 生成后自动预览
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

from rain_simulation import RainSimulation, process_kitti_rain
from snow_simulation import SnowSimulation, process_kitti_snow
from fog_simulation import FogSimulation, process_kitti_fog
from utils import (load_kitti_points, save_kitti_points,
                   sample_rain_rate, sample_snowfall_rate, sample_visibility)


# ============ 预设参数 ============

FIXED_PRESETS = {
    'rain': {
        'light':    {'rain_rate': 2.5},
        'moderate': {'rain_rate': 10.0},
        'heavy':    {'rain_rate': 30.0},
        'extreme':  {'rain_rate': 60.0},
    },
    'snow': {
        'light':    {'snowfall_rate': 0.5},
        'moderate': {'snowfall_rate': 2.5},
        'heavy':    {'snowfall_rate': 5.0},
        'extreme':  {'snowfall_rate': 10.0},
    },
    'fog': {
        'light':    {'visibility': 1000.0, 'fog_type': 'uniform'},
        'moderate': {'visibility': 500.0,  'fog_type': 'uniform'},
        'heavy':    {'visibility': 200.0,  'fog_type': 'inhomogeneous'},
        'extreme':  {'visibility': 50.0,   'fog_type': 'inhomogeneous'},
    }
}


# ============ 帧选择 ============

def resolve_frame_list(input_dir, frames=None, frame_range=None,
                       frame_file=None, all_frames=False):
    """
    解析要处理的帧列表

    Args:
        input_dir: velodyne目录
        frames: 指定帧号列表, 如 ['000001', '000050']
        frame_range: 帧号范围, 如 [0, 100] 表示000000~000099
        frame_file: 帧列表文件路径 (如KITTI的train.txt)
        all_frames: 处理所有帧

    Returns:
        list of filenames (如 ['000001.bin', '000050.bin'])
    """
    all_bins = sorted([f for f in os.listdir(input_dir) if f.endswith('.bin')])

    if frames is not None:
        # 指定帧号
        selected = []
        for f in frames:
            f = f.strip()
            if not f.endswith('.bin'):
                f = f.zfill(6) + '.bin'
            if f in all_bins:
                selected.append(f)
            else:
                print(f"  WARNING: {f} not found in {input_dir}, skipping")
        return selected

    elif frame_range is not None:
        # 帧号范围 [start, end)
        start, end = frame_range
        selected = []
        for f in all_bins:
            idx = int(f.replace('.bin', ''))
            if start <= idx < end:
                selected.append(f)
        return selected

    elif frame_file is not None:
        # 从文件读取帧号列表 (兼容KITTI的ImageSets格式)
        with open(frame_file, 'r') as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
        selected = []
        for l in lines:
            fname = l.zfill(6) + '.bin' if not l.endswith('.bin') else l
            if fname in all_bins:
                selected.append(fname)
            else:
                print(f"  WARNING: {fname} not found, skipping")
        return selected

    else:
        # 默认: 所有帧
        return all_bins


# ============ 单帧处理函数 ============

def process_single_frame(input_path, output_path, weather_type,
                         params, random_params=False, sample_mode='log',
                         rain_backend='auto', lisa_path=None,
                         snow_backend='auto', lidar_snow_sim_path=None,
                         particle_file_prefix=None, beam_divergence=0.35,
                         only_camera_fov=False, noise_floor=0.7, root_path=None):
    """
    处理单帧点云

    Args:
        input_path: 输入bin文件路径
        output_path: 输出bin文件路径
        weather_type: 'rain' | 'snow' | 'fog'
        params: 固定参数字典 (random_params=False时使用)
        random_params: 是否随机采样参数
        sample_mode: 随机采样模式

    Returns:
        dict: 该帧使用的实际参数
    """
    points = load_kitti_points(input_path)

    if weather_type == 'rain':
        rr = sample_rain_rate(sample_mode) if random_params else params.get('rain_rate', 10.0)
        sim = RainSimulation(rain_rate=rr, backend=rain_backend, lisa_path=lisa_path)
        result = sim.simulate(points)
        actual_params = {'rain_rate': round(rr, 2)}

    elif weather_type == 'snow':
        sr = sample_snowfall_rate(sample_mode) if random_params else params.get('snowfall_rate', 2.5)
        sim = SnowSimulation(snowfall_rate=sr,
                             backend=snow_backend,
                             lidar_snow_sim_path=lidar_snow_sim_path,
                             particle_file_prefix=particle_file_prefix,
                             beam_divergence=beam_divergence,
                             only_camera_fov=only_camera_fov,
                             noise_floor=noise_floor,
                             root_path=root_path)
        result = sim.simulate(points)
        actual_params = {'snowfall_rate': round(sr, 2)}

    elif weather_type == 'fog':
        if random_params:
            vis = sample_visibility(sample_mode)
            ft = 'inhomogeneous' if vis < 200 else 'uniform'
        else:
            vis = params.get('visibility', 500.0)
            ft = params.get('fog_type', 'uniform')
        sim = FogSimulation(visibility=vis, fog_type=ft)
        result = sim.simulate(points)
        actual_params = {'visibility': round(vis, 1), 'fog_type': ft}

    else:
        raise ValueError(f"Unknown weather type: {weather_type}")

    save_kitti_points(result, output_path)
    actual_params['num_points_in'] = len(points)
    actual_params['num_points_out'] = len(result)
    return actual_params


# ============ 批量处理（支持选帧） ============

def process_selected_frames(input_dir, output_dir, weather_type,
                            frame_list, params=None,
                            random_params=False, sample_mode='log',
                            num_workers=1,
                            rain_backend='auto', lisa_path=None,
                            snow_backend='auto', lidar_snow_sim_path=None,
                            particle_file_prefix=None, beam_divergence=0.35,
                            only_camera_fov=False, noise_floor=0.7, root_path=None):
    """
    处理选定的帧列表

    Args:
        input_dir: 输入velodyne目录
        output_dir: 输出velodyne目录
        weather_type: 天气类型
        frame_list: 帧文件名列表
        params: 固定参数
        random_params: 是否随机
        sample_mode: 采样模式
        num_workers: 并行worker数 (1=单进程)

    Returns:
        dict: 每帧的参数记录
    """
    os.makedirs(output_dir, exist_ok=True)
    params = params or {}
    param_log = OrderedDict()

    if random_params:
        desc = f"{weather_type} (random, {sample_mode})"
    else:
        desc = f"{weather_type} ({params})"

    print(f"  Processing {len(frame_list)} frames: {desc}")

    if num_workers <= 1:
        # 单进程
        for fname in tqdm(frame_list, desc=weather_type):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)
            actual = process_single_frame(
                input_path, output_path, weather_type,
                params, random_params, sample_mode,
                rain_backend=rain_backend, lisa_path=lisa_path,
                snow_backend=snow_backend, lidar_snow_sim_path=lidar_snow_sim_path,
                particle_file_prefix=particle_file_prefix, beam_divergence=beam_divergence,
                only_camera_fov=only_camera_fov, noise_floor=noise_floor, root_path=root_path
            )
            param_log[fname] = actual
    else:
        # 多进程
        from functools import partial
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for fname in frame_list:
                input_path = os.path.join(input_dir, fname)
                output_path = os.path.join(output_dir, fname)
                future = executor.submit(
                    process_single_frame,
                    input_path, output_path, weather_type,
                    params, random_params, sample_mode,
                    rain_backend, lisa_path,
                    snow_backend, lidar_snow_sim_path,
                    particle_file_prefix, beam_divergence,
                    only_camera_fov, noise_floor, root_path
                )
                futures[future] = fname

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=weather_type):
                fname = futures[future]
                try:
                    actual = future.result()
                    param_log[fname] = actual
                except Exception as e:
                    print(f"  ERROR processing {fname}: {e}")

    return param_log


# ============ 混合天气模式 ============

def process_mixed_weather(input_dir, output_dir, frame_list,
                          weather_types=None, weather_weights=None,
                          random_params=True, sample_mode='log',
                          num_workers=1, seed=None,
                          rain_backend='auto', lisa_path=None,
                          snow_backend='auto', lidar_snow_sim_path=None,
                          particle_file_prefix=None, beam_divergence=0.35,
                          only_camera_fov=False, noise_floor=0.7, root_path=None):
    """
    混合天气模式: 每帧随机分配一种天气类型

    Args:
        weather_types: 天气类型列表, 如 ['rain', 'snow', 'fog']
        weather_weights: 各天气类型的概率权重, 如 [0.4, 0.3, 0.3]
    """
    if seed is not None:
        np.random.seed(seed)

    weather_types = weather_types or ['rain', 'snow', 'fog']
    if weather_weights is None:
        weather_weights = [1.0 / len(weather_types)] * len(weather_types)
    weather_weights = np.array(weather_weights)
    weather_weights /= weather_weights.sum()

    os.makedirs(output_dir, exist_ok=True)

    # 为每帧分配天气类型
    assignments = np.random.choice(weather_types, size=len(frame_list),
                                   p=weather_weights)

    param_log = OrderedDict()

    print(f"\n  Mixed weather assignment:")
    for wt in weather_types:
        count = np.sum(assignments == wt)
        print(f"    {wt}: {count} frames ({count/len(frame_list)*100:.1f}%)")

    for fname, wtype in tqdm(zip(frame_list, assignments),
                              total=len(frame_list), desc="Mixed"):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        actual = process_single_frame(
            input_path, output_path, wtype,
            {}, random_params, sample_mode,
            rain_backend=rain_backend, lisa_path=lisa_path,
            snow_backend=snow_backend, lidar_snow_sim_path=lidar_snow_sim_path,
            particle_file_prefix=particle_file_prefix, beam_divergence=beam_divergence,
            only_camera_fov=only_camera_fov, noise_floor=noise_floor, root_path=root_path
        )
        actual['weather_type'] = wtype
        param_log[fname] = actual

    return param_log


# ============ 参数日志保存 ============

def save_param_log(output_dir, param_log, tag='weather'):
    """保存参数日志到txt和npy"""
    log_dir = os.path.dirname(output_dir.rstrip('/'))
    os.makedirs(log_dir, exist_ok=True)

    txt_path = os.path.join(log_dir, f'{tag}_params.txt')
    with open(txt_path, 'w') as f:
        # 写表头
        if param_log:
            first = list(param_log.values())[0]
            header = "filename," + ",".join(first.keys())
            f.write(header + "\n")
            for fname, params in param_log.items():
                vals = ",".join(str(v) for v in params.values())
                f.write(f"{fname},{vals}\n")

    npy_path = os.path.join(log_dir, f'{tag}_params.npy')
    np.save(npy_path, param_log, allow_pickle=True)
    print(f"  Param log: {txt_path}")


# ============ 生成后预览 ============

def auto_preview(clean_dir, output_dir, weather_name, frame_list, num_preview=3):
    """生成后自动可视化前几帧"""
    try:
        from visualize_weather import plot_bev_comparison, plot_intensity_distribution
    except ImportError:
        print("  (visualize_weather.py not found, skipping preview)")
        return

    preview_dir = os.path.join(os.path.dirname(output_dir.rstrip('/')), 'preview')
    os.makedirs(preview_dir, exist_ok=True)

    preview_frames = frame_list[:num_preview]
    for fname in preview_frames:
        fid = fname.replace('.bin', '')
        clean_path = os.path.join(clean_dir, fname)
        weather_path = os.path.join(output_dir, fname)

        if not (os.path.exists(clean_path) and os.path.exists(weather_path)):
            continue

        clean_pts = load_kitti_points(clean_path)
        weather_pts = load_kitti_points(weather_path)

        plot_bev_comparison(
            clean_pts, {weather_name: weather_pts},
            os.path.join(preview_dir, f'{fid}_{weather_name}_bev.png')
        )
        plot_intensity_distribution(
            clean_pts, {weather_name: weather_pts},
            os.path.join(preview_dir, f'{fid}_{weather_name}_intensity.png')
        )

    print(f"  Preview saved to: {preview_dir}")


# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced weather data generation for KITTI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
========== 用法示例 ==========

# 1. 对指定帧生成三种天气 (moderate)
python generate_all_weather.py \\
    --input_dir /data/kitti/training/velodyne \\
    --output_dir /data/kitti_weather \\
    --frames 000001 000050 000100 000200 \\
    --weather rain snow fog \\
    --severities moderate

# 2. 对指定帧范围生成 (帧号 0~499)
python generate_all_weather.py \\
    --input_dir /data/kitti/training/velodyne \\
    --output_dir /data/kitti_weather \\
    --frame_range 0 500 \\
    --weather rain \\
    --severities heavy

# 3. 使用KITTI train.txt指定帧
python generate_all_weather.py \\
    --input_dir /data/kitti/training/velodyne \\
    --output_dir /data/kitti_weather \\
    --frame_file /data/kitti/ImageSets/train.txt \\
    --weather rain snow fog \\
    --severities moderate

# 4. 对指定帧随机参数生成
python generate_all_weather.py \\
    --input_dir /data/kitti/training/velodyne \\
    --output_dir /data/kitti_weather \\
    --frames 000001 000050 000100 \\
    --weather rain snow fog \\
    --random_params --sample_mode log --seed 42

# 5. 混合天气模式 (每帧随机一种天气)
python generate_all_weather.py \\
    --input_dir /data/kitti/training/velodyne \\
    --output_dir /data/kitti_weather \\
    --frames 000001 000050 000100 \\
    --mode mixed \\
    --weather_weights 0.4 0.3 0.3

# 6. 全部帧 + 并行 + 自动预览
python generate_all_weather.py \\
    --input_dir /data/kitti/training/velodyne \\
    --output_dir /data/kitti_weather \\
    --weather rain snow fog \\
    --severities light moderate heavy \\
    --num_workers 4 \\
    --preview

# 7. 只对单帧生成所有天气所有强度 (快速测试)
python generate_all_weather.py \\
    --input_dir /data/kitti/training/velodyne \\
    --output_dir /data/kitti_weather_test \\
    --frames 000001 \\
    --weather rain snow fog \\
    --severities light moderate heavy extreme \\
    --preview
        """)

    # ===== 输入输出 =====
    parser.add_argument("--input_dir", type=str, required=True,
                        help="KITTI velodyne 目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出根目录")

    # ===== 帧选择 (互斥) =====
    frame_group = parser.add_argument_group("帧选择 (默认: 全部帧)")
    frame_group.add_argument("--frames", nargs='+', default=None,
                             help="指定帧号, 如: 000001 000050 000100")
    frame_group.add_argument("--frame_range", nargs=2, type=int, default=None,
                             metavar=('START', 'END'),
                             help="帧号范围 [start, end), 如: 0 500")
    frame_group.add_argument("--frame_file", type=str, default=None,
                             help="帧列表文件路径 (如KITTI的train.txt)")

    # ===== 天气类型 =====
    parser.add_argument("--weather", nargs='+', default=['rain', 'snow', 'fog'],
                        choices=['rain', 'snow', 'fog'],
                        help="天气类型")

    # ===== 模式选择 =====
    parser.add_argument("--mode", type=str, default='separate',
                        choices=['separate', 'mixed'],
                        help="separate: 每种天气单独生成\n"
                             "mixed: 每帧随机分配一种天气")

    # ===== 固定参数模式 =====
    parser.add_argument("--severities", nargs='+', default=['moderate'],
                        choices=['light', 'moderate', 'heavy', 'extreme'])

    # ===== 随机参数模式 =====
    parser.add_argument("--random_params", action='store_true',
                        help="每帧随机采样参数")
    parser.add_argument("--sample_mode", type=str, default='log',
                        choices=['uniform', 'log', 'category'])
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--rain_backend", type=str, default='auto',
                        choices=['auto', 'heuristic', 'lisa'],
                        help="雨模拟后端：auto(优先LISA)/heuristic/lisa")
    parser.add_argument("--lisa_path", type=str, default=None,
                        help="LISA仓库路径(应包含 atmos_models.py)")
    parser.add_argument("--snow_backend", type=str, default='auto',
                        choices=['auto', 'heuristic', 'lidar_snow_sim'],
                        help="雪模拟后端：auto(优先LiDAR_snow_sim)/heuristic/lidar_snow_sim")
    parser.add_argument("--lidar_snow_sim_path", type=str, default=None,
                        help="LiDAR_snow_sim 中 simulation.py 所在目录")
    parser.add_argument("--particle_file_prefix", type=str, default=None,
                        help="LiDAR_snow_sim augment 必需参数，如 gunn_4.816236598076465_1.1574074074074074")
    parser.add_argument("--beam_divergence", type=float, default=0.35,
                        help="LiDAR_snow_sim beam_divergence (degree)")
    parser.add_argument("--only_camera_fov", action='store_true',
                        help="LiDAR_snow_sim only_camera_fov")
    parser.add_argument("--noise_floor", type=float, default=0.7,
                        help="LiDAR_snow_sim noise_floor")
    parser.add_argument("--root_path", type=str, default=None,
                        help="LiDAR_snow_sim root_path (如STF root)")

    # ===== 混合天气选项 =====
    parser.add_argument("--weather_weights", nargs='+', type=float, default=None,
                        help="混合模式下各天气概率, 如: 0.4 0.3 0.3")

    # ===== 性能 =====
    parser.add_argument("--num_workers", type=int, default=1,
                        help="并行worker数 (1=单进程)")

    # ===== 附加功能 =====
    parser.add_argument("--preview", action='store_true',
                        help="生成后自动可视化预览前3帧")
    parser.add_argument("--dry_run", action='store_true',
                        help="仅显示将处理的帧列表，不实际生成")

    args = parser.parse_args()

    # ===== 设置随机种子 =====
    if args.seed is not None:
        np.random.seed(args.seed)

    # ===== 解析帧列表 =====
    frame_list = resolve_frame_list(
        args.input_dir,
        frames=args.frames,
        frame_range=args.frame_range,
        frame_file=args.frame_file
    )

    if not frame_list:
        print("ERROR: No frames found!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Weather Data Generation")
    print(f"{'='*60}")
    print(f"  Input:       {args.input_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Frames:      {len(frame_list)}")
    if len(frame_list) <= 10:
        print(f"  Frame list:  {[f.replace('.bin','') for f in frame_list]}")
    else:
        print(f"  Frame list:  {[f.replace('.bin','') for f in frame_list[:5]]} "
              f"... {[f.replace('.bin','') for f in frame_list[-3:]]}")
    print(f"  Weather:     {args.weather}")
    print(f"  Mode:        {args.mode}")
    if args.random_params:
        print(f"  Params:      RANDOM (mode={args.sample_mode})")
    else:
        print(f"  Severities:  {args.severities}")
    print(f"  Workers:     {args.num_workers}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without processing.")
        return

    # ===== 执行生成 =====

    if args.mode == 'mixed':
        # ---- 混合天气模式 ----
        out_dir = os.path.join(args.output_dir, "mixed_weather", "velodyne")
        print(f"\n>>> Mixed weather mode")

        param_log = process_mixed_weather(
            args.input_dir, out_dir, frame_list,
            weather_types=args.weather,
            weather_weights=args.weather_weights,
            random_params=args.random_params or True,
            sample_mode=args.sample_mode,
            num_workers=args.num_workers,
            seed=args.seed,
            rain_backend=args.rain_backend,
            lisa_path=args.lisa_path,
            snow_backend=args.snow_backend,
            lidar_snow_sim_path=args.lidar_snow_sim_path,
            particle_file_prefix=args.particle_file_prefix,
            beam_divergence=args.beam_divergence,
            only_camera_fov=args.only_camera_fov,
            noise_floor=args.noise_floor,
            root_path=args.root_path
        )
        save_param_log(out_dir, param_log, 'mixed')

        if args.preview:
            # 混合模式下按天气类型分别预览
            for wt in args.weather:
                wt_frames = [f for f, p in param_log.items()
                             if p.get('weather_type') == wt]
                if wt_frames:
                    auto_preview(args.input_dir, out_dir, f"mixed_{wt}",
                                 wt_frames[:2])

    else:
        # ---- 分离天气模式 ----
        if args.random_params:
            # 随机参数
            for w in args.weather:
                out_dir = os.path.join(args.output_dir, f"{w}_random", "velodyne")
                print(f"\n>>> {w} (random params, mode={args.sample_mode})")

                param_log = process_selected_frames(
                    args.input_dir, out_dir, w, frame_list,
                    random_params=True, sample_mode=args.sample_mode,
                    num_workers=args.num_workers,
                    rain_backend=args.rain_backend,
                    lisa_path=args.lisa_path,
                    snow_backend=args.snow_backend,
                    lidar_snow_sim_path=args.lidar_snow_sim_path,
                    particle_file_prefix=args.particle_file_prefix,
                    beam_divergence=args.beam_divergence,
                    only_camera_fov=args.only_camera_fov,
                    noise_floor=args.noise_floor,
                    root_path=args.root_path
                )
                save_param_log(out_dir, param_log, f'{w}_random')

                if args.preview:
                    auto_preview(args.input_dir, out_dir, f"{w}_random",
                                 frame_list)
        else:
            # 固定参数
            for w in args.weather:
                for s in args.severities:
                    if s not in FIXED_PRESETS[w]:
                        continue
                    out_dir = os.path.join(args.output_dir, f"{w}_{s}", "velodyne")
                    params = FIXED_PRESETS[w][s]
                    print(f"\n>>> {w} - {s}: {params}")

                    param_log = process_selected_frames(
                        args.input_dir, out_dir, w, frame_list,
                        params=params, random_params=False,
                        num_workers=args.num_workers,
                        rain_backend=args.rain_backend,
                        lisa_path=args.lisa_path,
                        snow_backend=args.snow_backend,
                        lidar_snow_sim_path=args.lidar_snow_sim_path,
                        particle_file_prefix=args.particle_file_prefix,
                        beam_divergence=args.beam_divergence,
                        only_camera_fov=args.only_camera_fov,
                        noise_floor=args.noise_floor,
                        root_path=args.root_path
                    )
                    save_param_log(out_dir, param_log, f'{w}_{s}')

                    if args.preview:
                        auto_preview(args.input_dir, out_dir, f"{w}_{s}",
                                     frame_list)

    print(f"\n{'='*60}")
    print(f"✓ All done! Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
