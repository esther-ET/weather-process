# 参数说明
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
    --weather rain \
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
python generate_all_weather.py \
    --input_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
    --output_dir /mnt/nvme0n1p2/data/datasets/kitti_weather_random \
    --weather rain snow fog \
    --random_params --sample_mode log --seed 42

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
```

compute_statistics(clean_pts, weather_pts) 的作用：计算 clean 与 weather 点云之间的统计差异，返回一个字典，主要包括：
点数与点数比例（clean_num_points/weather_num_points/point_ratio）
强度分布差异（均值差、KL、JS、Wasserstein、KS）
空间分布差异（MMD_xyz、MMD_intensity）
近距噪声比例与远距保留率（near_point_ratio_*、far_point_retention