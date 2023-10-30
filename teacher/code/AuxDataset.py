import argparse
import json
from pathlib import Path, PurePath
import pdb
import yaml
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import io
import numpy as np
import random
import re
import pickle
from glob import glob
import pdb
from utils import parsing_label

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip, CenterCrop, TenCrop
from videotransforms.volume_transforms import ClipToTensor

"""Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
class Split():
    def __init__(self):
        self.gt_a_list = []
        self.videos = []
    
    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def get_max_video_len(self):
        max_len = 0
        for v in self.videos:
            l = len(v)
            if l > max_len:
                max_len = l
        return max_len

    def __len__(self):
        return len(self.gt_a_list)


class AuxDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.get_item_counter = 0

        self.data_dir = args.path
        self.seq_len = args.seq_len
        self.mode = args.mode
        self.train = self.mode == "train"
        self.tensor_transform = transforms.ToTensor()
        self.img_size = args.img_size
        self.annotation_path = args.traintestlist
        self.d = parsing_label(os.path.join(self.args.traintestlist, "trainlist03.txt"))
        
        self.setup_transforms()
        self._select_fold()

    """Setup crop sizes/flips for augmentation during training and centre crop for testing"""
    def setup_transforms(self):
        video_transform_list = []
        video_test_list = []

        if self.img_size == 84:
            video_transform_list.append(Resize(96))
            video_test_list.append(Resize(96))
        elif self.img_size == 224:
            video_transform_list.append(Resize(256))
            video_test_list.append(Resize(256))
        else:
            print("img size transforms not setup")
            exit(1)
        video_transform_list.extend((RandomHorizontalFlip(), RandomCrop(self.img_size)))

        video_test_list.append(CenterCrop(self.img_size))

        self.transform = {"train": Compose(video_transform_list)}
        self.transform["test"] = Compose(video_test_list)
    
        
    """ 读取训练集和测试集的所有视频 """ 
    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            fname = "{}list{:02d}.txt".format(name, self.args.split)
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.replace(' ', '_') for x in data]
                data = [x.strip().split(" ")[0] for x in data]
#                 if "kinetics" in self.args.path:
#                     data = [x[0:11] for x in data]
                
                selected_files.extend(data)
                
            lists[name] = selected_files
        self.train_test_lists = lists

    """ 返回训练集或测试集的所有视频数 """
    def __len__(self):
        return len(self.train_test_lists[self.mode])
   
    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        return sorted(set(c.gt_a_list))
    
    """Loads a single image from a specified path """
    def read_single_image(self, path):
        with Image.open(path) as i:
            i = i.convert("RGB")
            i.load()
            return i
    
    def f(self, path):
        return [os.path.join(path, i) for i in os.listdir(path)]

    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, video_path, modality='rgb'):
        path = os.path.join(self.data_dir, video_path)
        path_list = path.split("/")
        path_list[-3] = f'{modality}_l8'
        path = '/'.join(path_list)
        paths = self.f(path)
        paths = sorted(paths)
        if len(paths) > self.seq_len:
            l = np.linspace(0, len(paths) - 1, self.seq_len).tolist()
            t = [paths[int(i)] for i in l]
            paths = t
        n_frames = len(paths)
        if len(paths) == 0:
            l = 1
            raise Exception(f"No frames found for video {video_path}")
        if n_frames == self.args.seq_len:
            idxs = [int(f) for f in range(n_frames)]
        else:
            if self.train:
                excess_frames = n_frames - self.seq_len
                excess_pad = int(min(5, excess_frames / 2))
                if excess_pad < 1:
                    start = 0
                    end = n_frames - 1
                else:
                    start = random.randint(0, excess_pad)
                    end = random.randint(n_frames-1 -excess_pad, n_frames-1)
            else:
                start = 1
                end = n_frames - 2

            if end - start < self.seq_len:
                end = n_frames - 1
                start = 0
            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]

            if self.seq_len == 1:
                idxs = [random.randint(start, end-1)]

        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            transform = self.transform["train"] if self.train else self.transform["test"]
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)
        return (video_path, imgs)

    def get_feature_seq(self, path, deit=""):
        print("-------现在特征提取为多模态特征提取-----")
        print(path)
        path = path.split(os.sep)
        print('*', path)
        class_folders, video_folders = path[-2], path[-1]
        rgb_feature_path = os.path.join(self.args.feature_save_path, f'rgb{deit}', class_folders, video_folders)
        depth_feature_path = os.path.join(self.args.feature_save_path, f'depth{deit}', class_folders, video_folders)
        flow_feature_path = os.path.join(self.args.feature_save_path, f'flow{deit}', class_folders, video_folders)
        rgb_feature = np.load(os.path.join(rgb_feature_path,"feature.npy"))
        depth_feature = np.load(os.path.join(depth_feature_path,"feature.npy"))
        flow_feature = np.load(os.path.join(flow_feature_path,"feature.npy"))
        return torch.from_numpy(rgb_feature), torch.from_numpy(depth_feature), torch.from_numpy(flow_feature)

    def get_multi_fea(self, index):
        video_path = self.train_test_lists[self.mode][index]
        rgb_feature, depth_feature, flow_feature = self.get_feature_seq(video_path)
        feature = dict(rgb=rgb_feature, depth=depth_feature, flow=flow_feature)
        return (video_path, feature)

    def get_video(self, index):
        video_path = self.train_test_lists[self.mode][index]
        return self.get_seq(video_path)
    
    def get_video_with_label(self, index):
        video_path = self.train_test_lists[self.mode][index]
        c = video_path.split('/')[0]
        label = self.d[c]
        path, video = self.get_seq(video_path, modality=self.args.modality)
        return dict(path=path, label=label, video=video)
    
    def __getitem__(self, index):
        func = getattr(self, self.args.getitem_name)
        return func(index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="imp_datasets/video_datasets/data/rgb_hmdb51_256x256q5_l8")
    parser.add_argument("--seq_len", default=8, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--traintestlist", default="imp_datasets/video_datasets/splits/hmdb51TrainTestlist")
    parser.add_argument("--split", default=3, type=int)
    parser.add_argument("--mode", default="test")
    args = parser.parse_args()
    dataset = AuxDataset(args)
    for i in dataset:
        print(i.shape)
        break
    
