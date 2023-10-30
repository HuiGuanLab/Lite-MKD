import random
import os
import zipfile
import io
import numpy as np
from ruamel import yaml

def generate_yaml_doc_ruamel(yaml_file, py_object):
    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(py_object, file, Dumper=yaml.RoundTripDumper)
    file.close()

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


class FixedSplitGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_split = Split()
        self.test_split = Split()
        self.seq_len = 8
        self.annotation_path = "splits/hmdb_ARN"

        lists = {}
        for name in ["train", "test"]:
            fname = f"{name}list03.txt"
            f = os.path.join(self.annotation_path, fname)
            selected_files = []
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.replace(' ', '_').lower() for x in data]
                data = [x.strip().split(" ")[0] for x in data]
                data = [os.path.splitext(os.path.split(x)[1])[0] for x in data]
                
                
                selected_files.extend(data)
            lists[name] = selected_files
        self.train_test_lists = lists

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
                print(class_folders)
                video_folders = os.listdir(os.path.join(self.data_dir, class_folder))
                video_folders.sort()
                for video_folder in video_folders:
                    c = self.get_train_or_test_db(video_folder)
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

    def gen_test(self):
        classes = self.test_split.get_unique_classes()
        classes = random.sample(classes, 5)
        di = {}
        for eposidex_idx in range(10000):
            support = []
            query = []
            support_idx = 0
            query_idx = 0
            for bl, bc in enumerate(classes):
                n_total = self.test_split.get_num_videos_for_class(bc)
                idxs = random.sample([i for i in range(n_total)], 9)
                for j in idxs[:5]:
                    d = {}
                    d['id'] = support_idx
                    d['class_bc'] = bc
                    d['video_idx'] = self.test_split.get_rand_vid(bc, j)[1]
                    support_idx += 1
                    support.append(d)
                for j in idxs[5:]:
                    d = {}
                    d['id'] = query_idx
                    d['class_bc'] = bc
                    d['video_idx'] = self.test_split.get_rand_vid(bc, j)[1]
                    query_idx += 1
                    query.append(d)
            di[eposidex_idx] = {'support': support, 'query': query}
        generate_yaml_doc_ruamel("splits/hmdb_ARN/fixed_test.yaml", di)
        

generator = FixedSplitGenerator("/media/disk4/zp/dataset/hmdb/hmdb_256q5.zip")
generator.read_dir()
generator.gen_test()