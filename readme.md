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


# 1) 只开雪 weather，避免和雨雾串行
# 2) 帧级并行 --num_workers N（N≈物理核数的一半到1倍）
# 3) LiDAR_snow_sim 内部并行使用 process（CPU 充足时）
# 4) 添加 --skip_existing 支持断点续处理（已处理的帧直接跳过）
# particle_file_prefix是可选的
python snow_simulation.py \
  --input_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
  --output_dir /mnt/nvme0n1p2/data/datasets/kitti_weather_random/snow/velodyne \
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

## 雾模型参数与能见度对标

本仓库的雾模拟以 MartinHahner/LiDAR_fog_sim 的 `fog_simulation.py` 和
`generate_integral_lookup_table.py` 为对齐目标；新增的 `ParameterSet` 字段不是重新设计的
经验参数，而是上游 `ParameterSet` 中用于 hard target 衰减、soft target 查表与积分表生成的物理/数值参数。

### 核心物理量

| 参数 | 含义 | 本仓库/上游中的作用 |
| --- | --- | --- |
| `alpha` | 大气消光/衰减系数，单位约为 `1/m`，数值越大雾越浓 | hard target 用 `exp(-2 * alpha * r)` 做双程强度衰减；soft target 用它选择最近的积分表 |
| `mor` | Meteorological Optical Range，气象光学距离/能见度 | 由 Koschmieder 关系 `mor = log(20) / alpha` 得到；本仓库输入的 `visibility` 会映射为 `alpha = log(20) / visibility` |
| `beta` | 后向散射系数，单位约为 `1/sr` | soft target 的雾回波强度缩放项；为贴近 LiDAR_fog_sim，上游覆盖 `alpha` 时不会自动重算 `beta` |
| `gamma` | hard target 反射率相关参数 | 决定 `beta_0 = gamma / pi`，参与真实物体回波与雾回波的相对强度标定 |
| `beta_0` | hard target differential reflectivity | soft target 强度缩放中的分母，用于比较雾回波是否强过衰减后的物体回波 |

### 新增/保留参数来自哪里

- `n`、`r_range` 及其 min/max：积分表生成时的距离采样精度和最大积分范围。
- `p_0`、`tau_h`、`e_p`：发射脉冲峰值功率、半高脉宽、脉冲能量；`tau_h` 也出现在积分表文件名中。
- `a_r`、`l_r`、`c_a`：接收孔径、接收光学损耗和组合常数，soft target 积分归一化/反归一化会用到。
- `D`、`ROH_T`、`ROH_R`、`GAMMA_T*`、`GAMMA_R*`、`r_1`、`r_2`、`linear_xsi`：上游单束理论模型中的发射器/接收器几何与视场重叠参数；当前批量点云增强主要依赖预计算表，但保留它们可以保持参数集语义一致。
- `r_0`：单束理论/积分表生成时的 hard target 距离变量；批量点云增强时每个点会用自己的欧氏距离替代。
- `*_min`、`*_max`、`*_scale`：上游 GUI/参数调节范围，批处理仿真不直接使用；保留它们是为了和上游 `ParameterSet` 接近。
- `gain`：上游 `simulate_fog(..., gain=...)` 的可选强度归一化开关，默认关闭。

### 能见度如何对标实际、与积分表的关系

- 实际对标使用 Koschmieder 定律：`visibility = mor = log(20) / alpha`，即 5% 对比度阈值下的气象光学距离。
  例如 `visibility=500m` 时，代码使用 `alpha≈0.00599`；`visibility=50m` 时，`alpha≈0.0599`，雾更浓。
- hard target 衰减只依赖 `alpha` 和点距，不依赖积分表；即使没有积分表，也会生成真实物体回波变暗的结果。
- soft target 依赖积分表：积分表按离散 `alpha` 和 `tau_h` 预计算雾滴回波峰值距离/响应；运行时会选择与当前 `alpha` 最近的表。
- 因此积分表不会重新定义“能见度”，但会影响雾点替换的位置和强度；若目标 visibility 对应的 `alpha` 与表中离散值差距较大，soft target 会近似到最近表，和真实目标 visibility 之间会产生量化误差。
- 若积分表缺失，本仓库会退化为仅 hard attenuation；这种输出仍按目标能见度变暗，但不会产生 LiDAR_fog_sim 的近距雾回波点。


## 生成符合指定能见度分布的雾天

如果目标是让雾天点云的强度分布更接近某个数据集，通常需要先指定或拟合一组 `visibility` 分布；
`visibility` 会通过 `alpha = log(20) / visibility` 控制 hard target 衰减，并通过最近的积分表控制 soft target 雾回波。

本仓库支持两类随机雾参数：

- 内置分布：`--random_params --sample_mode uniform|log|category`，最终调用 `utils.sample_visibility()`。
- 自定义离散分布：`--fog_visibility_values` 给定能见度取值，`--fog_visibility_weights` 给定对应权重。

`fog_type` 和 `fog_visibility` 是两个独立维度：

- `fog_visibility` 决定雾的浓度，也就是 `alpha = log(20) / visibility`；visibility 越小，雾越浓。
- `fog_type` 决定 `alpha` 在空间上如何使用：`uniform` 使用同一个 `alpha`，更贴近 LiDAR_fog_sim；`inhomogeneous` 是本仓库扩展，会在每个点上加入随机的局部 `alpha` 扰动。
- 设置 `--fog_visibility_values` 或 `--fog_visibility_weights` 只会影响每帧采样到的 `visibility`，不会覆盖 `--fog_type`。例如 `--fog_type uniform --fog_visibility_values 50 100` 仍然生成 uniform fog，只是每帧能见度从 50m/100m 中采样。

示例：按 50m/100m/200m/500m 四个能见度档位生成雾，其中 200m 权重最高：

```bash
python generate_all_weather.py \
  --weather fog \
  --input_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
  --output_dir /mnt/nvme0n1p2/data/datasets/kitti_weather_random \
  --random_params --sample_mode log --seed 42 \
  --fog_type uniform \
  --fog_visibility_values 50 100 200 500 \
  --fog_visibility_weights 0.1 0.2 0.5 0.2
```

输出会写到 `OUTPUT/fog_random/velodyne`，每帧实际采样到的 `visibility` 会记录在
`OUTPUT/fog_random_params.txt` 和 `OUTPUT/fog_random_params.npy`。如果不提供
`--fog_visibility_values`，则继续使用 `--sample_mode` 的内置采样逻辑。

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