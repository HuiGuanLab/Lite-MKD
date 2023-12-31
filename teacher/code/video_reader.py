import argparse
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
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.get_item_counter = 0

        self.data_dir = args.path
        self.seq_len = args.seq_len
        self.train = args.test_model_only == False
        self.tensor_transform = transforms.ToTensor()
        self.img_size = args.img_size

        self.annotation_path = args.traintestlist

        self.way=args.way
        self.shot=args.shot
        self.query_per_class = args.query_per_class
        self.feature_save_path = args.feature_save_path

        self.train_split = Split()
        self.test_split = Split()

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
        self.zip = False
        class_folders = os.listdir(self.data_dir)
        class_folders.sort()
        self.class_folders = class_folders
        for class_folder in class_folders:
            video_folders = os.listdir(os.path.join(self.data_dir, class_folder))
            video_folders.sort()
            if self.args.debug_loader:
                video_folders = video_folders[0:1]
            for video_folder in video_folders:
                c = self.get_train_or_test_db(video_folder)
                if c == None:
                    continue
                imgs = os.listdir(os.path.join(self.data_dir, class_folder, video_folder))
                imgs.sort()
                paths = [os.path.join(self.data_dir, class_folder, video_folder, img) for img in imgs]
                paths.sort()
                class_id =  class_folders.index(class_folder)
                c.add_vid(paths, class_id)
        c = self.get_train_or_test_db()
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
                data = [x.replace(' ', '_') for x in data]
                data = [x.strip().split(" ")[0] for x in data]
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                
#                 if "kinetics" in self.args.path:
#                     data = [x[0:11] for x in data]
                
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
                    i = i.convert("RGB")
                    i.load()
                    return i
        else:
            with Image.open(path) as i:
                i = i.convert("RGB")
                i.load()
                return i
    
    """Gets a single video sequence. Handles sampling if there are more frames than specified. """
    def get_seq(self, label, idx=-1, modality='rgb'):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        if modality == 'S3D':
            s3d = np.load(os.path.join(paths[0]))
            s3d = torch.from_numpy(s3d).unsqueeze(dim=0)
            return s3d, vid_id

        for i, path in enumerate(paths):
            l = path.split('/')
            l[-4] = modality + '_l8'
            paths[i] = '/'.join(l)
        n_frames = len(paths)
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
            else:
                pass
    
            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]
            
            if self.seq_len == 1:
                idxs = [random.randint(start, end-1)]

        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)
        return imgs, vid_id

    def get_cross_view_rgb_seq(self, label, idx=-1, support=True, query_view=3, modality='rgb'):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        if modality == 'S3D':
            s3d = np.load(os.path.join(paths[0]))
            s3d = torch.from_numpy(s3d).unsqueeze(dim=0)
            return s3d, vid_id

        l = paths[0].split('/')
        clazz, video = l[-3], l[-2]
        prefix = os.path.join('/'.join(self.args.path.split('/')[:-1]), "all_view_rgb_l8")
        view_list = sorted(os.listdir(prefix))
        if support:
            view_list = view_list[:query_view] + view_list[query_view:]
            view = random.choice(view_list)
        else:
            view = view_list[query_view]

        path = os.path.join(prefix, view, clazz, video)
        paths = [os.path.join(path, i) for i in os.listdir(path)]

        n_frames = len(paths)
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
            else:
                pass
    
            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]
            
            if self.seq_len == 1:
                idxs = [random.randint(start, end-1)]

        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)
        return imgs, vid_id

    def get_fixed_view_modality_seq(self, label, idx=-1, view='Camera_0', modality='rgb'):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(label, idx)
        if modality == 'S3D':
            s3d = np.load(os.path.join(paths[0]))
            s3d = torch.from_numpy(s3d).unsqueeze(dim=0)
            return s3d, vid_id

        # l = paths[0].split('/')
        # clazz, video = l[-3], l[-2]
        # prefix = os.path.join('/'.join(self.args.path.split('/')[:-1]), f"all_view_{modality}_l8")

        # path = os.path.join(prefix, view, clazz, video)
        # paths = [os.path.join(path, i) for i in os.listdir(path)]

        n_frames = len(paths)
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
            else:
                pass
    
            idx_f = np.linspace(start, end, num=self.seq_len)
            idxs = [int(f) for f in idx_f]
            
            if self.seq_len == 1:
                idxs = [random.randint(start, end-1)]

        imgs = [self.read_single_image(paths[i]) for i in idxs]
        if (self.transform is not None):
            if self.train:
                transform = self.transform["train"]
            else:
                transform = self.transform["test"]
            
            imgs = [self.tensor_transform(v) for v in transform(imgs)]
            imgs = torch.stack(imgs)
        return imgs, vid_id

    def get_multi_seq(self, label, idx=-1):
        rgb_imgs, vid_id = self.get_seq(label, idx, 'rgb')
        flow_imgs, vid_id = self.get_seq(label, idx, 'flow')
        depth_imgs, vid_id = self.get_seq(label, idx, 'depth')
        return rgb_imgs, flow_imgs, depth_imgs, vid_id

    def get_feature_seq(self, bc, bl):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(bc, bl)
        path = paths[0].split(os.sep)
        class_folders, video_folders = path[-3], path[-2]
        feature_path = os.path.join(self.feature_save_path, 'rgb',class_folders, video_folders)
        feature = np.load(os.path.join(feature_path,"feature.npy"))
        return torch.from_numpy(feature)

    def get_mul_feature_seq(self, bc, bl):
        c = self.get_train_or_test_db()
        paths, vid_id = c.get_rand_vid(bc, bl)
        path = paths[0].split(os.sep)
        class_folders, video_folders = path[-3], path[-2]
        prefix = '/'.join(self.args.path.split('/')[:-2])
        feature_path = os.path.join(prefix, f'{self.args.dataset}_feature','multi_feature',class_folders, video_folders)
        feature = np.load(os.path.join(feature_path,"feature.npy"))
        return torch.from_numpy(feature)

    def get_multi_feature_task(self, index):

        #select classes to use for this task
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_fea = []
        support_labels = []
        target_fea = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        for bl, bc in enumerate(batch_classes):
            
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)

            for idx in idxs[0:self.args.shot]:
                vid_fea = self.get_mul_feature_seq(bc, idx)
                support_fea.append(vid_fea)
                support_labels.append(bl)

            for idx in idxs[self.args.shot:]:
                vid_fea = self.get_mul_feature_seq(bc, idx)
                target_fea.append(vid_fea)
                target_labels.append(bl)
                real_target_labels.append(bc)
        

        support_fea = torch.cat(support_fea)
        target_fea = torch.cat(target_fea)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes) 
        
        return {"support_set":support_fea, "support_labels":support_labels, 
                "target_set":target_fea, "target_labels":target_labels, 
                "real_target_labels":real_target_labels, "batch_class_list": batch_classes}

    def get_single_modality_task(self, index):
        # 默认getitem
        # 读取单一模态的数据，构造小样本任务
        
        # select classes to use for this task
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_set = []
        support_labels = []
        support_vid = []
        target_set = []
        target_labels = []
        target_vid = []
        real_support_labels = []
        real_target_labels = []

        for bl, bc in enumerate(batch_classes):
            
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)

            for idx in idxs[0:self.args.shot]:
                if self.args.fixed_view is not None:
                    vid, vid_id = self.get_fixed_view_modality_seq(bc, idx, self.args.fixed_view)
                elif self.args.cross_view:
                    vid, vid_id = self.get_cross_view_rgb_seq(bc, idx, True, self.args.view)
                else:
                    vid, vid_id = self.get_seq(bc, idx, self.args.modality)
                support_set.append(vid)
                support_labels.append(bl)
                support_vid.append(vid_id)
                real_support_labels.append(bc)

            for idx in idxs[self.args.shot:]:
                if self.args.fixed_view is not None:
                    vid, vid_id = self.get_fixed_view_modality_seq(bc, idx, self.args.fixed_view)
                elif self.args.cross_view:
                    vid, vid_id = self.get_cross_view_rgb_seq(bc, idx, False, self.args.view)
                else:
                    vid, vid_id = self.get_seq(bc, idx, self.args.modality)
                target_set.append(vid)
                target_labels.append(bl)
                target_vid.append(vid_id)
                real_target_labels.append(bc)
        
        s = list(zip(support_set, support_labels, real_support_labels, support_vid))
        random.shuffle(s)
        support_set, support_labels, real_support_labels, support_vid = zip(*s)
        
        t = list(zip(target_set, target_labels, real_target_labels, target_vid))
        random.shuffle(t)
        target_set, target_labels, real_target_labels, target_vid = zip(*t)
        
        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_support_labels = torch.FloatTensor(real_support_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        support_vid = torch.LongTensor(support_vid)
        target_vid = torch.LongTensor(target_vid)
        batch_classes = torch.FloatTensor(batch_classes) 

        return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, "target_labels":target_labels,
                "real_support_labels":real_support_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes,
                "support_vid":support_vid, "target_vid":target_vid}

    def get_random_modality_task(self, index):

        #select classes to use for this task
        c = self.get_train_or_test_db()
        classes = c.get_unique_classes()
        batch_classes = random.sample(classes, self.way)

        if self.train:
            n_queries = self.args.query_per_class
        else:
            n_queries = self.args.query_per_class_test

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        modality_candidates = ['rgb', 'depth', 'flow']

        for bl, bc in enumerate(batch_classes):
            
            #select shots from the chosen classes
            n_total = c.get_num_videos_for_class(bc)
            idxs = random.sample([i for i in range(n_total)], self.args.shot + n_queries)
            modality = random.choice(modality_candidates)
            if self.train == False:
                modality = 'rgb'
            for idx in idxs[0:self.args.shot]:
                vid, vid_id = self.get_seq(bc, idx, modality)
                support_set.append(vid)
                support_labels.append(bl)

            for idx in idxs[self.args.shot:]:
                vid, vid_id = self.get_seq(bc, idx, modality)
                target_set.append(vid)
                target_labels.append(bl)
                real_target_labels.append(bc)
        
        s = list(zip(support_set, support_labels))
        random.shuffle(s)
        support_set, support_labels = zip(*s)
        
        t = list(zip(target_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)
        
        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes) 
        
        return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, "target_labels":target_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes}

    def __getitem__(self, index):
        func = getattr(self, self.args.getitem_name)
        return func(index)

    def __getitem_from_fixed__(self, index):
        # 获取固定的episode

        c = self.get_train_or_test_db()
        data = self.fixed_test_eposide[index]

        support_set = []
        support_labels = []
        target_set = []
        target_labels = []
        real_support_labels = []
        real_target_labels = []

        supports_info = data['support']
        query_info = data['query']
        for support_info in supports_info:
            bl, bc, idx = support_info['id'], support_info['class_bc'], support_info['video_idx']
            vid, vid_id = self.get_seq(bc, idx, self.args.modality)
            support_set.append(vid)
            support_labels.append(bl)
        
        for query_info in query_info:
            bl, bc, idx = query_info['id'], query_info['class_bc'], query_info['video_idx']
            vid, vid_id = self.get_seq(bc, idx, self.args.modality)
            target_set.append(vid)
            target_labels.append(bl)
            real_target_labels.append(bc)
        
        
        s = list(zip(support_set, support_labels))
        random.shuffle(s)
        support_set, support_labels = zip(*s)
        
        t = list(zip(target_set, target_labels, real_target_labels))
        random.shuffle(t)
        target_set, target_labels, real_target_labels = zip(*t)
        
        support_set = torch.cat(support_set)
        target_set = torch.cat(target_set)
        support_labels = torch.FloatTensor(support_labels)
        target_labels = torch.FloatTensor(target_labels)
        real_target_labels = torch.FloatTensor(real_target_labels)
        batch_classes = torch.FloatTensor(batch_classes) 
        return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, "target_labels":target_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["ssv2", "kinetics", "hmdb", "ucf"],
        default="hmdb",
        help="Dataset to use.",
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.001, help="Learning rate."
    )
    parser.add_argument(
        "--tasks_per_batch",
        type=int,
        default=16,
        help="Number of tasks between parameter optimizations.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        "-c",
        default=None,
        help="Directory to save checkpoint to.",
    )
    parser.add_argument(
        "--test_model_path",
        "-m",
        default=None,
        help="Path to model to load and test.",
    )
    parser.add_argument(
        "--training_iterations",
        "-i",
        type=int,
        default=100020,
        help="Number of meta-training iterations.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        "-r",
        dest="resume_from_checkpoint",
        default=False,
        action="store_true",
        help="Restart from latest checkpoint.",
    )
    parser.add_argument("--way", type=int, default=5,
                        help="Way of each task.")
    parser.add_argument("--shot", type=int, default=5,
                        help="Shots per class.")
    parser.add_argument(
        "--query_per_class",
        type=int,
        default=5,
        help="Target samples (i.e. queries) per class used for training.",
    )
    parser.add_argument(
        "--query_per_class_test",
        type=int,
        default=1,
        help="Target samples (i.e. queries) per class used for testing.",
    )
    parser.add_argument(
        "--test_iters",
        nargs="+",
        type=int,
        help="iterations to test at. Default is for ssv2 otam split.",
        default=[75000],
    )
    parser.add_argument(
        "--num_test_tasks",
        type=int,
        default=10000,
        help="number of random tasks to test on.",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=1000,
        help="print and log every n iterations.",
    )
    parser.add_argument("--seq_len", type=int,
                        default=8, help="Frames per video.")
    parser.add_argument(
        "--num_workers", type=int, default=10, help="Num dataloader workers."
    )
    parser.add_argument(
        "--method",
        choices=["resnet18", "resnet34", "resnet50"],
        default="resnet50",
        help="method",
    )
    parser.add_argument(
        "--trans_linear_out_dim",
        type=int,
        default=1152,
        help="Transformer linear_out_dim",
    )
    parser.add_argument(
        "--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer"
    )
    parser.add_argument(
        "--trans_dropout", type=int, default=0.1, help="Transformer dropout"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=5000,
        help="Number of iterations between checkpoint saves.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size to the CNN after cropping.",
    )
    parser.add_argument(
        "--temp_set",
        nargs="+",
        type=int,
        help="cardinalities e.g. 2,3 is pairs and triples",
        default=[2, 3],
    )
    parser.add_argument(
        "--scratch",
        choices=["bc", "bp", "new"],
        default="new",
        help="directory containing dataset, splits, and checkpoint saves.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to split the ResNet over",
    )
    parser.add_argument(
        "--debug_loader",
        default=False,
        action="store_true",
        help="Load 1 vid per class for debugging",
    )
    parser.add_argument("--split", type=int, default=3,
                        help="Dataset split.")
    parser.add_argument(
        "--sch",
        nargs="+",
        type=int,
        help="iters to drop learning rate",
        default=[1000000],
    )
    parser.add_argument(
        "--test_model_only",
        type=bool,
        default=True,
        help="Only testing the model from the given checkpoint",
    )
    parser.add_argument(
        "--fixed_test_eposide",
        type=str,
        default="splits/hmdb_ARN/fixed_test.json",
        help="fixed test episodes",
    )
    parser.add_argument(
        "--feature_save_path", type=str, default="imp_datasets/video_datasets/data/hmdb_feature"
    )

    args = parser.parse_args()

    if args.scratch == "bc":
        args.scratch = "/mnt/storage/home2/tp8961/scratch"
    elif args.scratch == "bp":
        args.num_gpus = 4
        # this is low becuase of RAM constraints for the data loader
        args.num_workers = 3
        args.scratch = "/work/tp8961"
    elif args.scratch == "new":
        args.scratch = "./imp_datasets/"


    if (args.method == "resnet50") or (args.method == "resnet34"):
        args.img_size = 224
    if args.method == "resnet50":
        args.trans_linear_in_dim = 2048
    else:
        args.trans_linear_in_dim = 512

    if args.dataset == "ssv2":
        args.traintestlist = os.path.join(
            args.scratch, "video_datasets/splits/somethingsomethingv2TrainTestlist"
        )
        args.path = os.path.join(
            args.scratch,
            "video_datasets/data/somethingsomethingv2_256x256q5_7l8.zip",
        )
    elif args.dataset == "kinetics":
        args.traintestlist = os.path.join(
            args.scratch, "video_datasets/splits/kineticsTrainTestlist"
        )
        args.path = os.path.join(
            args.scratch, "video_datasets/data/kinetics_256q5_1.zip"
        )
    elif args.dataset == "ucf":
        args.traintestlist = os.path.join(
            args.scratch, "video_datasets/splits/ucfTrainTestlist"
        )
        args.path = os.path.join(
            args.scratch, "video_datasets/data/UCF-101_320.zip"
        )
    elif args.dataset == "hmdb":
        args.traintestlist = os.path.join(
            args.scratch, "video_datasets/splits/hmdb51TrainTestlist"
        )
        args.path = os.path.join(
            args.scratch, "video_datasets/data/rgb_hmdb51_256q5.zip"
        )
        # args.path = os.path.join(args.scratch, "video_datasets/data/hmdb51_jpegs_256.zip")

    dataset = VideoDataset(args)
    video_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=8)
    for i in video_loader:
        print(i['support_fea'].shape)
        exit()