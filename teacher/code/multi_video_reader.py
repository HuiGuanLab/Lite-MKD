import json
import pdb
import yaml
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import zipfile
import io
import numpy as np
import random
import re
import pickle
from glob import glob

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
    

"""Dataset for few-shot videos, which returns few-shot tasks. """
class MultiVideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.get_item_counter = 0

        self.data_dir = args.path
        self.seq_len = args.seq_len
        self.train = True
        self.tensor_transform = transforms.ToTensor()
        self.img_size = args.img_size

        self.annotation_path = args.traintestlist

        self.way=args.way
        self.shot=args.shot
        self.query_per_class=args.query_per_class

        self.train_split = Split()
        self.test_split = Split()
        self.feature_save_path = args.feature_save_path

        fixed_episodes = json.load(open(args.fixed_test_eposide, "r"))
        fixed_episodes = json.loads(fixed_episodes)

        self.setup_transforms()
        self._select_fold()
        self.read_dir()

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
        video_transform_list.append(RandomHorizontalFlip())
        video_transform_list.append(RandomCrop(self.img_size))

        video_test_list.append(CenterCrop(self.img_size))

        self.transform = {}
        self.transform["train"] = Compose(video_transform_list)
        self.transform["test"] = Compose(video_test_list)
    
    """Loads all videos into RAM from an uncompressed zip. Necessary as the filesystem has a large block size, which is unsuitable for lots of images. """
    """Contains some legacy code for loading images directly, but this has not been used/tested for a while so might not work with the current codebase. """
    def read_dir(self):
        # load zipfile into memory
        if self.data_dir.endswith('.zip'):
            self.zip = True
            zip_fn = os.path.join(self.data_dir)
            self.mem = open(zip_fn, 'rb').read()
            self.zfile = zipfile.ZipFile(io.BytesIO(self.mem))
        else:
            self.zip = False
        # go through zip and populate splits with frame locations and action groundtruths
        if self.zip:
            # When using 'png' based datasets like kinetics, replace 'jpg' to 'png'
            dir_list = list(set([x for x in self.zfile.namelist() if '.jpg' not in x]))

            class_folders = list(set([x.split(os.sep)[-3] for x in dir_list if len(x.split(os.sep)) > 2]))
            class_folders.sort()
            self.class_folders = class_folders
            video_folders = list(set([x.split(os.sep)[-2] for x in dir_list if len(x.split(os.sep)) > 3]))
            video_folders.sort()
            self.video_folders = video_folders

            class_folders_indexes = {v: k for k, v in enumerate(self.class_folders)}
            video_folders_indexes = {v: k for k, v in enumerate(self.video_folders)}
            
            img_list = [x for x in self.zfile.namelist() if '.jpg' in x]
            img_list.sort()

            c = self.get_train_or_test_db(video_folders[0])

            last_video_folder = None
            last_video_class = -1
            insert_frames = []
            for img_path in img_list:
            
                class_folder, video_folder, jpg = img_path.split(os.sep)[-3:]

                if video_folder != last_video_folder:
                    if len(insert_frames) >= self.seq_len:
                        c = self.get_train_or_test_db(last_video_folder.lower())
                        if c != None:
                            c.add_vid(insert_frames, last_video_class)
                        else:
                            pass
                    insert_frames = []
                    class_id = class_folders_indexes[class_folder]
                    vid_id = video_folders_indexes[video_folder]
               
                insert_frames.append(img_path)
                last_video_folder = video_folder
                last_video_class = class_id

            c = self.get_train_or_test_db(last_video_folder)
            if c != None and len(insert_frames) >= self.seq_len:
                c.add_vid(insert_frames, last_video_class)

        else:
            class_folders = os.listdir(self.data_dir)
            class_folders.sort()
            self.class_folders = class_folders
            for class_folder in class_folders:
                video_folders = os.listdir(os.path.join(self.data_dir, class_folder))
                video_folders.sort()
                if self.args.debug_loader:
                    video_folders = video_folders[0:1]
                for video_folder in video_folders:
                    c = self.get_train_or_test_db(video_folder.lower())
                    if c == None:
                        continue
                    imgs = os.listdir(os.path.join(self.data_dir, class_folder, video_folder))
                    if len(imgs) < self.seq_len:
                        continue            
                    imgs.sort()
                    paths = [os.path.join(self.data_dir, class_folder, video_folder, img) for img in imgs]
                    paths.sort()
                    class_id =  class_folders.index(class_folder)
                    c.add_vid(paths, class_id)
        print("loaded {}".format(self.data_dir))
        print("train: {}, test: {}".format(len(self.train_split), len(self.test_split)))

    """ return the current split being used """
    def get_train_or_test_db(self, split=None):
        if split is None:
            get_train_split = self.train
        else:
            if split in self.train_test_lists["train"]:
                get_train_split = True
            elif split in self.train_test_lists["test"]:
                get_train_split = False
            else:
                return None
        if get_train_split:
            return self.train_split
        else:
            return self.test_split
    
    """ load the paths of all videos in the train and test splits. """ 
    def _select_fold(self):
        lists = {}
        for name in ["train", "test"]:
            fname = "{}list{:02d}.txt".format(name, self.args.split)
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.replace(' ', '_').lower() for x in data]
                data = [x.strip().split(" ")[0] for x in data]
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                
                # if "kinetics" in self.args.path:
                #     data = [x[0:11] for x in data]
                
                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists

    """ Set len to large number as we use lots of random tasks. Stopping point controlled in run.py. """
    def __len__(self):
        c = self.get_train_or_test_db()
        return 1000000
        return len(c)
   
    """ Get the classes used for the current split """
    def get_split_class_list(self):
        c = self.get_train_or_test_db()
        classes = list(set(c.gt_a_list))
        classes.sort()
        return classes
    
    """Loads a single image from a specified path """
    def read_single_image(self, path):
        if self.zip:
            with self.zfile.open(path, 'r') as f:
                with Image.open(f) as i:
                    i.load()
                    return i
        else:
            with Image.open(path) as i:
                i.load()
                return i
    
    def get_feature_seq(self, bc, bl):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(bc, bl)
        path = paths[0].split(os.sep)
        class_folders, video_folders = path[-3], path[-2]
        m1_feature_path = os.path.join(self.feature_save_path, f'{self.args.m1}', class_folders, video_folders)
        m2_feature_path = os.path.join(self.feature_save_path, f'{self.args.m2}', class_folders, video_folders)
        m3_feature_path = os.path.join(self.feature_save_path, f'{self.args.m3}', class_folders, video_folders)
        m4_feature_path = os.path.join(self.feature_save_path, f'{self.args.m4}', class_folders, video_folders)
        m5_feature_path = os.path.join(self.feature_save_path, f'{self.args.m5}', class_folders, video_folders)

        m1_feature = np.load(os.path.join(m1_feature_path,"feature.npy"))
        try:
            m2_feature = np.load(os.path.join(m2_feature_path,"feature.npy"))
        except Exception:
            m2_feature = np.zeros_like(m1_feature)
        try:
            m3_feature = np.load(os.path.join(m3_feature_path,"feature.npy"))
        except Exception:
            m3_feature = np.zeros_like(m1_feature)
        try:
            m4_feature = np.load(os.path.join(m4_feature_path,"feature.npy"))
        except Exception:
            m4_feature = np.zeros_like(m1_feature)
        try:
            m5_feature = np.load(os.path.join(m5_feature_path,"feature.npy"))
        except Exception:
            m5_feature = np.zeros_like(m1_feature)
        
        return torch.from_numpy(m1_feature), torch.from_numpy(m2_feature), torch.from_numpy(m3_feature), torch.from_numpy(m4_feature), torch.from_numpy(m5_feature), [class_folders, video_folders]

    def __getitem__(self, index):

        #select classes to use for this task
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        m1_support_fea = []
        m2_support_fea = []
        m3_support_fea = []
        m4_support_fea = []
        m5_support_fea = []
        support_labels = []
        m1_target_fea = []
        m2_target_fea = []
        m3_target_fea = []
        m4_target_fea = []
        m5_target_fea = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []
        
        # c_v means class and video infomation
        support_c_v = []
        target_c_v = []

        for bl, bc in enumerate(batch_classes):
            
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)

            for idx in idxs[0:self.args.shot]:
                vid_fea = self.get_feature_seq(bc, idx)
                m1_support_fea.append(vid_fea[0])
                m2_support_fea.append(vid_fea[1])
                m3_support_fea.append(vid_fea[2])
                m4_support_fea.append(vid_fea[3])
                m5_support_fea.append(vid_fea[4])
                support_labels.append(bl)
                real_support_labels.append(bc)
                support_c_v.append(vid_fea[4])

            for idx in idxs[self.args.shot:]:
                vid_fea = self.get_feature_seq(bc, idx)
                m1_target_fea.append(vid_fea[0])
                m2_target_fea.append(vid_fea[1])
                m3_target_fea.append(vid_fea[2])
                m4_target_fea.append(vid_fea[3])
                m5_target_fea.append(vid_fea[4])
                target_labels.append(bl)
                real_target_labels.append(bc)
                target_c_v.append(vid_fea[4])
        
        s = list(zip(m1_support_fea, m2_support_fea, m3_support_fea, m4_support_fea, m5_support_fea, support_labels, real_support_labels, support_c_v))
        random.shuffle(s)
        m1_support_fea, m2_support_fea, m3_support_fea, m4_support_fea, m5_support_fea, support_labels, real_support_labels, support_c_v = zip(*s)
        
        t = list(zip(m1_target_fea, m2_target_fea, m3_target_fea, m4_target_fea, m5_target_fea, target_labels, real_target_labels, target_c_v))
        random.shuffle(t)
        m1_target_fea, m2_target_fea, m3_target_fea, m4_target_fea, m5_target_fea, target_labels, real_target_labels, target_c_v = zip(*t)


        m1_support_fea = torch.cat(m1_support_fea)
        m2_support_fea = torch.cat(m2_support_fea)
        m3_support_fea = torch.cat(m3_support_fea)
        m4_support_fea = torch.cat(m4_support_fea)
        m5_support_fea = torch.cat(m5_support_fea)

        m1_target_fea = torch.cat(m1_target_fea)
        m2_target_fea = torch.cat(m2_target_fea)
        m3_target_fea = torch.cat(m3_target_fea)
        m4_target_fea = torch.cat(m4_target_fea)
        m5_target_fea = torch.cat(m5_target_fea)

        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes) 
        
        return {"support_fea": {self.args.m1:m1_support_fea, self.args.m2:m2_support_fea, self.args.m3:m3_support_fea, self.args.m4:m4_support_fea, self.args.m5:m5_support_fea},
                "support_labels":support_labels, 
                "target_fea": {self.args.m1:m1_target_fea, self.args.m2:m2_target_fea, self.args.m3:m3_target_fea, self.args.m4:m4_target_fea, self.args.m5:m5_target_fea}, 
                "target_labels":target_labels, 
                "real_target_labels":real_target_labels, "batch_class_list": batch_classes,
                "support_c_v":support_c_v,
                "target_c_v":target_c_v}


