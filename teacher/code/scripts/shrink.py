import os
import numpy as np


def f(path):
    return [os.path.join(path, i) for i in os.listdir(path) ]

path = '/media/disk4/zp/dataset/dance/rgb_random_view'
for clazz in f(path):
    for video in f(clazz):
        l = sorted(video)
        np.linespace()
