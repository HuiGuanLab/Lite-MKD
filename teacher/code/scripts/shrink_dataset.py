import os
import shutil
import numpy as np


def f(path):
    l = []
    for i in os.listdir(path):
        l.append(os.path.join(path, i))
    return l
l = ["rgb_Camera_0",  "rgb_Camera_1",  "rgb_Camera_2",  "rgb_Camera_3",  "rgb_Camera_4",  "rgb_Camera_5"]

def shrink(prefix, outpath):
    for _class in f(prefix):
        c = _class.split('/')[-1]
        for video in f(_class):
            v = video.split('/')[-1]
            l = sorted(f(video))
            index_list = np.linspace(0, len(l)-1, 8)
            for index, i in enumerate(index_list):
                i = int(i)
                t = os.path.join(outpath, c, v)
                if not os.path.exists(t):
                    os.makedirs(t)
                out = os.path.join(t, str(index+1).zfill(8)+'.jpg' )
                try:
                    shutil.copy(l[i], out)
                except:
                    print(video, len(l), i, index_list)
                    exit()
        
prefix = '/media/disk4/zp/dataset/dance/'
out_path = prefix + 'all_view_rgb_l8'
for i in l:
    input = os.path.join(prefix, i)
    out = os.path.join(out_path, i)
    shrink(input, out)