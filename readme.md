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

三种采样模式对比：
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

# 跨域分析命令汇总
```python
# 1. 生成天气数据（随机参数）
python generate_all_weather.py \
    --input_dir /mnt/nvme0n1p2/data/datasets/KITTI2/object/training/velodyne \
    --output_dir /mnt/nvme0n1p2/data/datasets/kitti_weather_random \
    --weather rain snow fog \
    --random_params --sample_mode log --seed 42

# 2. 可视化对比
python vis_and_diff.py \
    --clean_dir /data/kitti/training/velodyne \
    --weather_dirs \
        rain:/data/kitti_weather/rain_random/velodyne \
        snow:/data/kitti_weather/snow_random/velodyne \
        fog:/data/kitti_weather/fog_random/velodyne \
    --output_dir /data/analysis/visualization \
    --num_samples 10

# 3. Domain shift分析
python domain_analysis.py \
    --clean_dir /data/kitti/training/velodyne \
    --weather_dirs \
        rain:/data/kitti_weather/rain_random/velodyne \
        snow:/data/kitti_weather/snow_random/velodyne \
        fog:/data/kitti_weather/fog_random/velodyne \
    --output_dir /data/analysis/domain_shift \
    --max_frames 200
```