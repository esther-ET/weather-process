"""
utils.py - 通用工具函数
"""

import numpy as np
import os


def load_kitti_points(bin_path):
    """加载KITTI bin文件 -> (N, 4)"""
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)


def save_kitti_points(points, save_path):
    """保存为KITTI bin文件，4维"""
    points = points[:, :4].astype(np.float32)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    points.tofile(save_path)


def get_lidar_distance(points):
    """每个点到原点的距离"""
    return np.sqrt(np.sum(points[:, :3] ** 2, axis=1))


# ============ 随机参数采样 ============

def sample_rain_rate(mode='uniform'):
    """
    随机采样降雨率 (mm/h)
    范围: [1, 80]
    
    Args:
        mode: 
            'uniform'  - 均匀分布 [1, 80]
            'log'      - 对数均匀分布 (偏向中小雨，更真实)
            'category' - 先随机选等级，再在等级内均匀采样
    """
    if mode == 'uniform':
        return np.random.uniform(1.0, 80.0)

    elif mode == 'log':
        # log-uniform: 在 [log(1), log(80)] 上均匀采样后取exp
        # 这样小雨概率更大，更符合真实气象分布
        return np.exp(np.random.uniform(np.log(1.0), np.log(80.0)))

    elif mode == 'category':
        # 先等概率选等级，再在范围内均匀采样
        categories = {
            'light':    (1.0,  5.0),
            'moderate': (5.0,  20.0),
            'heavy':    (20.0, 50.0),
            'extreme':  (50.0, 80.0),
        }
        cat = np.random.choice(list(categories.keys()))
        lo, hi = categories[cat]
        return np.random.uniform(lo, hi)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def sample_snowfall_rate(mode='uniform'):
    """
    随机采样降雪率 (mm/h 水当量)
    范围: [0.5, 10]
    """
    if mode == 'uniform':
        return np.random.uniform(0.5, 10.0)

    elif mode == 'log':
        return np.exp(np.random.uniform(np.log(0.5), np.log(10.0)))

    elif mode == 'category':
        categories = {
            'light':    (0.5, 1.0),
            'moderate': (1.0, 4.0),
            'heavy':    (4.0, 7.0),
            'extreme':  (7.0, 10.0),
        }
        cat = np.random.choice(list(categories.keys()))
        lo, hi = categories[cat]
        return np.random.uniform(lo, hi)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def sample_visibility(mode='uniform'):
    """
    随机采样能见度 (m)
    范围: [30, 2000]
    注意: 值越小雾越浓
    """
    if mode == 'uniform':
        return np.random.uniform(30.0, 2000.0)

    elif mode == 'log':
        # log-uniform: 使中/浓雾概率更大一些
        return np.exp(np.random.uniform(np.log(30.0), np.log(2000.0)))

    elif mode == 'category':
        categories = {
            'dense':    (30.0,   200.0),
            'thick':    (200.0,  500.0),
            'moderate': (500.0,  1000.0),
            'light':    (1000.0, 2000.0),
        }
        cat = np.random.choice(list(categories.keys()))
        lo, hi = categories[cat]
        return np.random.uniform(lo, hi)

    else:
        raise ValueError(f"Unknown mode: {mode}")