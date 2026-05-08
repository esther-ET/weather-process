雨mm/h
毛毛雨   0.1 - 1.0
小雨     1.0 - 5.0
中雨     5.0 - 20.0
大雨     20.0 - 50.0
暴雨     50.0 - 100.0
大暴雨   100.0 - 150.0   ← 超过80后LiDAR几乎无法工作

雪mm/h
毛毛雨   0.1 - 1.0
小雨     1.0 - 5.0
中雨     5.0 - 20.0
大雨     20.0 - 50.0
暴雨     50.0 - 100.0
大暴雨   100.0 - 150.0   ← 超过80后LiDAR几乎无法工作

雾能见度m
轻雾     1000 - 2000
中雾     500 - 1000
大雾     200 - 500
浓雾     50 - 200
强浓雾   30 - 50         ← <30m后点云几乎全丢

# 天气分类
light
moderate
heavy
extreme

# 三种采样模式对比：
sample_mode='uniform'   均匀分布 → 每个值等概率
sample_mode='log'       对数均匀 → 偏向中低强度，更接近真实天气分布 (推荐)
sample_mode='category'  分类采样 → 先等概率选等级(light/moderate/heavy/extreme)，再在等级内均匀

# 推荐：每帧随机，log分布，可复现
python generate_all_weather.py \
    --input_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
    --output_dir /mnt/nvme0n1p2/data/datasets/kitti_weather_random \
    --weather fog \
    --random_params \
    --sample_mode log \
    --seed 42
#  对指定帧生成三种指定强度的天气 
python generate_all_weather.py \
    --input_dir  /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
    --output_dir  /mnt/nvme0n1p2/data/datasets/kitti_weather_random/extreme \
    --frames 000001 000050 000100 \
    --weather rain snow fog \
    --severities extreme

# 跨域分析命令汇总
```python
# 1. 生成天气数据（随机参数）
# particle_file_prefix 是可选的
python generate_all_weather.py \
 --weather snow \
 --input_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
 --output_dir /mnt/nvme0n1p2/data/datasets/kitti_weather_random \
 --rain_backend auto \
 --lisa_path ~/SWW/code/LiDAR_snow_sim/lib/LISA \
 --snow_backend auto \
 --lidar_snow_sim_path ~/SWW/code/LiDAR_snow_sim/tools/snowfall \
 --random_params --sample_mode log --seed 42 \
 --particle_file_prefix gunn_4.816236598076465_1.1574074074074074e-06\
--lidar_parallel_backend process

# 雪模拟太慢时（如 7891 帧耗时过长），建议优先这样跑：
# 1) 只开雪 weather，避免和雨雾串行
# 2) 帧级并行 --num_workers N（N≈物理核数的一半到1倍）
# 3) LiDAR_snow_sim 内部并行使用 process（CPU 充足时）
# 4) 添加 --skip_existing 支持断点续处理（已处理的帧直接跳过）
# particle_file_prefix是可选的
python snow_simulation.py \
  --input_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
  --output_dir /mnt/nvme0n1p2/data/datasets/kitti_weather_random/snow_fast/velodyne \
  --backend lidar_snow_sim \
  --lidar_snow_sim_path ~/SWW/code/LiDAR_snow_sim \
  --particle_file_prefix gunn_4.816236598076465_1.1574074074074074e-06 \
  --num_workers 8 \
  --lidar_parallel_backend process \
  --channel_mode infer \
  --skip_existing

# 2. 可视化对比 在仅仅生成Saved JSON: /home/ubuntu/SWW/analysis/visualization_moderate/statistics.json 时候说明路径有问题了
python vis_and_diff.py \
  --clean_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
  --weather_dirs \
    rain:/mnt/nvme0n1p2/data/datasets/kitti_weather_random/rain_random/velodyne \
    snow:/mnt/nvme0n1p2/data/datasets/kitti_weather_random/snow_random/velodyne \
    fog:/mnt/nvme0n1p2/data/datasets/kitti_weather_random/fog_random/velodyne \
  --output_dir /home/ubuntu/SWW/analysis/visualization_1 \
  --frames 000001 000050 000100
# # 特别地之后用这个
python vis_and_diff.py \
  --clean_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
  --weather_dirs \
    rain:/mnt/nvme0n1p2/data/datasets/kitti_weather_random/extreme/rain_extreme/velodyne \
    snow:/mnt/nvme0n1p2/data/datasets/kitti_weather_random/extreme/snow_extreme/velodyne\
    fog:/mnt/nvme0n1p2/data/datasets/kitti_weather_random/extreme/fog_extreme/velodyne \
  --output_dir /home/ubuntu/SWW/analysis/visualization_extreme\
  --frames 000001 000050 000100


# 3. Domain shift分析
python domain_analysis.py \
    --clean_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
    --weather_dirs \
        rain:/mnt/nvme0n1p2/data/datasets/kitti_weather_random/rain_random/velodyne \
        snow:/mnt/nvme0n1p2/data/datasets/kitti_weather_random/snow_random/velodyne \
        fog:/mnt/nvme0n1p2/data/datasets/kitti_weather_random/fog_random/velodyne \
    --output_dir /data/analysis/domain_shift \
    --max_frames 500

# 4. /home/ubuntu/SWW/code/weather-process/analyze_split_weather_distribution.py
训练集和测试集的分布分析。
```

compute_statistics(clean_pts, weather_pts) 的作用：计算 clean 与 weather 点云之间的统计差异，返回一个字典，主要包括：
点数与点数比例（clean_num_points/weather_num_points/point_ratio）
强度分布差异（均值差、KL、JS、Wasserstein、KS）
空间分布差异（MMD_xyz、MMD_intensity）
近距噪声比例与远距保留率（near_point_ratio_*、far_point_retention


# 参数说明
## 雾积分表（本地化，无外部耦合）
- 雾 soft target 需要积分查表文件，默认目录：`/home/ubuntu/SWW/code/weather-process/integral_lookup_tables/original`
- 文件名格式：`integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_*.pickle`
- 若目录为空，代码会给出 warning，并退化为仅 hard attenuation（仍可运行）

如果你本机已有 LiDAR_fog_sim 的积分表，可一次性复制到 weather-process：
```bash
mkdir -p /home/ubuntu/SWW/code/weather-process/integral_lookup_tables/original
cp /home/ubuntu/SWW/code/LiDAR_snow_sim/lib/LiDAR_fog_sim/integral_lookup_tables/original/*.pickle \
  /home/ubuntu/SWW/code/weather-process/integral_lookup_tables/original/
```

# 深度复用外部物理仿真后端（LISA / LiDAR_snow_sim）
- 雨支持后端：
  - `--rain_backend heuristic`：当前仓库启发式
  - `--rain_backend lisa`：强制使用 LISA（需可导入 `atmos_models.py`）
  - `--rain_backend auto`：优先 LISA，不可用时回退 heuristic
- 雪支持后端：
  - `--snow_backend heuristic`：当前仓库启发式
  - `--snow_backend lidar_snow_sim`：强制使用 LiDAR_snow_sim（需可导入 `tools/snowfall/simulation.py`）
  - `--snow_backend auto`：优先 LiDAR_snow_sim，不可用时回退 heuristic
- 注意：`LiDAR_snow_sim` 的 `augment` 接口原生要求 **N×5**（`x,y,z,intensity,channel`）。
  本仓库已内置 Nx4 适配：
  - `--channel_mode infer`（默认）：按点的仰角 + FOV 估计 pseudo ring/channel
  - `--channel_mode zero`：第5维全0
  - `--channel_mode require`：强制必须输入Nx5

  - 可选 `--beam_divergence --only_camera_fov --noise_floor --root_path`
  - 适配参数：`--num_lasers --fov_down_deg --fov_up_deg`

示例：
雪生成：先每个档位平均，再在档位里面取值！！！！！！！！！！！！ 之后用这个
```bash
python generate_all_weather.py \
 --weather snow\
 --input_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
 --output_dir /mnt/nvme0n1p2/data/datasets/kitti_weather_random \
 --rain_backend auto \
 --lisa_path ~/SWW/code/LiDAR_snow_sim/lib/LISA \
 --snow_backend auto \
 --lidar_snow_sim_path ~/SWW/code/LiDAR_snow_sim/tools/snowfall \
 --random_params --sample_mode log --seed 42 \
  --particle_model gunn\
  --rainfall_rate_levels 2 8 17 34 70 \
  --rainfall_level_sampling balanced
```

若你的目录结构是：
`~/SWW/code/weather-process` 与 `~/SWW/code/LiDAR_snow_sim` 同级，
推荐直接这样指定：
```bash
--lisa_path ~/SWW/code/LiDAR_snow_sim/lib/LISA \
--lidar_snow_sim_path ~/SWW/code/LiDAR_snow_sim
```
（`--lidar_snow_sim_path` 既支持仓库根目录，也支持直接给 `tools/snowfall` 目录）
现在导入失败时 warning / error 会打印已搜索路径与导入异常，便于定位依赖或路径问题。

## 雪分布相关：
中间两个参数：等效降雨率，mm/h 二维采样域内雪花盘面积占比
gunn_2.621627143512277_7.716049382716048e-07_1.npy
gunn_2.2383844962893775_6.944444444444445e-07_1.npy
gunn_3.1282374334279206_8.680555555555554e-07_1.npy
gunn_3.8219731266508847_9.92063492063492e-07_1.npy
gunn_4.816236598076465_1.1574074074074074e-06_1.npy
gunn_6.331107424916213_1.388888888888889e-06_1.npy
gunn_7.4150813236809965_1.5432098765432096e-06_1.npy
gunn_8.847991609353935_1.7361111111111108e-06_1.npy
gunn_10.810172461470367_1.984126984126984e-06_1.npy
gunn_11.63098702334301_2.0833333333333334e-06_1.npy
gunn_13.622374233194783_2.3148148148148144e-06_1.npy
gunn_13.622374233194787_2.3148148148148148e-06_1.npy
gunn_16.254798518508064_2.604166666666666e-06_1.npy
gunn_17.90707597031502_2.777777777777778e-06_1.npy
gunn_19.859554921566634_2.976190476190476e-06_1.npy
gunn_20.973017148098215_3.086419753086419e-06_1.npy
gunn_25.025899467423365_3.4722222222222215e-06_1.npy
gunn_29.31068252276024_3.858024691358024e-06_1.npy
gunn_30.575785013207078_3.968253968253968e-06_1.npy
gunn_32.89739918439432_4.166666666666667e-06_1.npy
gunn_34.97475775452152_4.340277777777777e-06_1.npy
gunn_38.52989278461172_4.6296296296296296e-06_1.npy
gunn_42.730958596843955_4.9603174603174595e-06_1.npy
gunn_45.97551303703239_5.208333333333332e-06_1.npy
gunn_50.6488593993297_5.555555555555556e-06_1.npy
gunn_53.84716214510654_5.787037037037037e-06_1.npy
gunn_70.78393287483146_6.944444444444444e-06_1.npy
gunn_98.92355351431581_8.680555555555554e-06_1.npy
gunn_108.97899386555827_9.259259259259257e-06_1.npy
gunn_130.0383881480645_1.0416666666666665e-05_1.npy
gunn_152.30277400182553_1.1574074074074072e-05_1.npy
gunn_200.20719573938692_1.3888888888888886e-05_1.npy
gunn_279.79806203617215_1.7361111111111108e-05_1.npy
gunn_367.80410429625914_2.083333333333333e-05_1.npy
gunn_566.2714629986518_2.7777777777777772e-05_1.npy
gunn_791.3884281145265_3.4722222222222215e-05_1.npy

原仓库：
0.5 2.0 2.2383844962893775 =2
1.0 1.6 8.847991609353935  =8
2.0 2.0 17.90707597031502  =17
2.5 1.6 34.97475775452152  =34
1.5 0.6 70.78393287483148  =70