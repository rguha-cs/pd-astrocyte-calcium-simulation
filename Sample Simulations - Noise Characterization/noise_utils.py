# noise_utils.py

import cupy as cp
import numpy as np

def gaussian_noise_map(data, level=0.01):
    return level * data * cp.random.randn(*data.shape)

def poisson_noise_map(rate=0.1, shape=None):
    return cp.random.poisson(rate, size=shape)

def ornstein_uhlenbeck_noise(shape, theta=0.2, sigma=0.02, dt=0.0001, steps=10):
    noise = cp.zeros(shape)
    for _ in range(steps):
        noise += theta * (-noise) * dt + sigma * cp.sqrt(dt) * cp.random.randn(*shape)
    return noise

def localized_patch_noise(shape, num_patches=5, strength=0.05, patch_size_range=(5, 15)):
    noise = cp.zeros(shape)
    for _ in range(num_patches):
        i, j = np.random.randint(0, shape[0]), np.random.randint(0, shape[1])
        size = np.random.randint(*patch_size_range)
        patch = strength * cp.random.randn(size, size)
        noise[i:i+size, j:j+size] += patch[:shape[0]-i, :shape[1]-j]
    return noise

def apply_noise(data, noise_type='gaussian', mode='mild', **kwargs):
    level_map = {
        'mild': 0.005,
        'moderate': 0.01,
        'heavy': 0.02
    }
    # Prefer sigma if passed, else use mode lookup
    level = kwargs.get("sigma", level_map.get(mode, 0.01))

    if noise_type == 'gaussian':
        return data + gaussian_noise_map(data, level=level)
    elif noise_type == 'ou':
        return data + ornstein_uhlenbeck_noise(data.shape, sigma=level)
    elif noise_type == 'poisson':
        rate = kwargs.get("rate", 0.1)
        return data + poisson_noise_map(rate=rate, shape=data.shape)
    elif noise_type == 'localized':
        return data + localized_patch_noise(data.shape, strength=level)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
