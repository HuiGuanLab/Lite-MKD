from AuxDataset import AuxDataset
import torch
import numpy as np
import argparse
import os
from loguru import logger


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        gpu_device = "cuda"
        self.parse_command_line()
        self.device = torch.device(
            gpu_device if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()
        self.dataset = self.args.dataset
        self.vd = AuxDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(
            self.vd, batch_size=1, num_workers=self.args.num_workers
        )
        self.load_checkpoint()
        logger.add("work/logger/extract.log")

    def init_model(self):
        file_name = "model"
        class_name = self.args.model
        modules = __import__(file_name)
        model = getattr(modules, class_name)(self.args)
        model = model.to(self.device) 
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def parse_command_line(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seq_len", default=8, type=int)
        parser.add_argument("--img_size", default=224, type=int)
        parser.add_argument("--traintestlist")
        parser.add_argument("--split", default=3, type=int)
        parser.add_argument("--mode", default="test")
        parser.add_argument("--method", default="resnet50")
        parser.add_argument("--path")
        parser.add_argument("--modality", default="rgb", type=str)
        parser.add_argument("--checkpoint_dir")
        parser.add_argument(
            "--temp_set", nargs="+", type=int, 
            help="cardinalities e.g. 2,3 is pairs and triples", default=[2],
        )
        parser.add_argument(
            "--trans_linear_out_dim",
            type=int,
            default=1152,
            help="Transformer linear_out_dim",
        )
        parser.add_argument("--num_gpus", type=int, default=4,)
        parser.add_argument("--num_workers", type=int, default=1, help="Num dataloader workers.")
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
        parser.add_argument("--dataset", type=str, default="hmdb")
        parser.add_argument("--model", type=str)
        parser.add_argument("--getitem_name", type=str)
        parser.add_argument("--num_classes", type=int, default=64)


        args = parser.parse_args()
        if args.method == "resnet50":
            args.trans_linear_in_dim = 2048
        # imp_datasets/video_datasets/data/hmdb_feature
        args.feature_save_path = f"imp_datasets/video_datasets/data/{args.dataset}_{args.method}_{args.model}_feature/{args.modality}"
        os.makedirs(args.feature_save_path, exist_ok=True)
        self.args = args

    def load_checkpoint(self):
        logger.info(f"Loading checkpoint... {self.args.checkpoint_dir}")
        checkpoint = torch.load(self.args.checkpoint_dir)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    def extract(self):
        self.model.eval()
        with torch.no_grad():
            self.video_loader.dataset.train = self.args.mode
            for video_path, video in self.video_loader:
                video = video.squeeze().cuda()
                feature = self.model.extract_feature(video)
                save_path = os.path.join(self.args.feature_save_path, video_path[0])
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, "feature.npy")
                np.save(save_path, feature.cpu().numpy())
                logger.info(f"feature saved in {save_path}")


def main():
    learner = Learner()
    learner.extract()


main()
