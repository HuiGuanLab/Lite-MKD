import os
import pathlib
import sys
sys.path[-1] = "/home/zp/code/strm"
from loguru import logger
from AuxDataset import AuxDataset
import torch
import argparse
import torch.optim as optim
import numpy as np
import torch.nn as nn

class Learner:

    def __init__(self):
        self.args = self.parse_command_line()
        gpu_device = "cuda"
        self.parse_command_line()
        
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        logger.add(pathlib.PurePath(self.args.checkpoint_dir) / "pretrain.log", rotation="500 MB")
        
        self.device = torch.device(
            gpu_device if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()
        self.dataset = self.args.dataset
        self.vd = AuxDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(
            self.vd, batch_size=8, num_workers=self.args.num_workers, shuffle=True)
        
        self.optimizer_1 = optim.SGD(self.model.convnet.parameters(), lr=self.args.lr_1, momentum=0.9)
        self.optimizer_2 = optim.SGD(self.model.fc.parameters(), lr=self.args.lr_2, momentum=0.9)
        self.scheduler_1 = optim.lr_scheduler.StepLR(
            self.optimizer_1, step_size=10, gamma=0.1
        )
        self.scheduler_2 = optim.lr_scheduler.StepLR(
            self.optimizer_2, step_size=10, gamma=0.1
        )
        
        if self.args.resume:
            self.load_checkpoint()
        
        self.epoch_nums = 50
        self.best_acc = 0
        

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
        parser.add_argument("--traintestlist", default="splits/pretrain/kinetics", type=str)
        parser.add_argument("--split", default=3, type=int)
        parser.add_argument("--mode", default="train")
        parser.add_argument("--method", default="resnet50")
        parser.add_argument("--model", type=str)
        parser.add_argument("--modality", default="rgb", type=str)
        parser.add_argument("--getitem_name", type=str)
        parser.add_argument("--checkpoint_dir", "-c", type=str, help="path to save checkpoint")
        parser.add_argument(
            "--temp_set", nargs="+", type=int,
            help="cardinalities e.g. 2,3 is pairs and triples", default=[2],
        )
        parser.add_argument(
            "--trans_linear_out_dim", type=int, default=1152,
            help="Transformer linear_out_dim",
        )
        parser.add_argument("--path", type=str)
        parser.add_argument("--num_gpus", type=int, default=4,)
        parser.add_argument("--num_workers", type=int,default=1, help="Num dataloader workers.")
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
        parser.add_argument("--dataset", type=str, default="hmdb")
        parser.add_argument("--learning_rate", '-lr', type=float, default=0.1)
        parser.add_argument("--num_classes", type=int, default=64)
        parser.add_argument("--lr_1", type=float, default=0.000001)
        parser.add_argument("--lr_2", type=float, default=0.01)
        parser.add_argument("--resume", "-r", type=bool, default=False)
        args = parser.parse_args()
        if args.method == "resnet50":
            args.trans_linear_in_dim = 2048
        
        self.args = args


    def train(self):
        logger.debug("模型的具体参数如下")
        logger.debug(self.args)
        logger.info("loss will write:")


        criterion = nn.CrossEntropyLoss()

        # train
        for epoch in range(self.epoch_nums):
            self.model.train()
            running_loss = 0
            acc_list = []
            self.video_loader.train = True
            self.scheduler_1.step()
            self.scheduler_2.step()
            for i_batch, sample_batched in enumerate(self.video_loader):
                # get the inputs
                video, label = sample_batched["video"], sample_batched["label"]
                label = label.long()
                video, label = video.cuda(), label.cuda()

                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()

                # forward
                output = self.model(video)
                loss = criterion(output, label)
                loss.backward()
                self.optimizer_1.step()
                self.optimizer_2.step()

                # caculate accuracy
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                predicted_y = np.argmax(output, axis=1)
                accuracy = np.mean(label == predicted_y)
                acc_list.append(accuracy)

                # print statistics
                running_loss = running_loss + loss.item()
                step = 50
                if (i_batch + 1) % step == 0:
                    logger.info(f"epoch{epoch+1}/{self.epoch_nums}  {i_batch+1}/{len(self.video_loader)} loss: {round(running_loss/step, 3)} accuracy: {sum(acc_list)/step}")
                    running_loss = 0.0
                    acc_list = []


            self.model.eval()
            with torch.no_grad():
                sum_right, sum_num = 0, 0
                self.video_loader.train = False
                for sample_batched in self.video_loader:
                    # get the inputs
                    video, label = sample_batched["video"], sample_batched["label"]
                    label = label.long()
                    video, label = video.cuda(), label.cuda()

                    # forward
                    output = self.model(video)

                    # caculate accuracy
                    label = label.data.cpu().numpy()
                    output = output.data.cpu().numpy()
                    predicted_y = np.argmax(output, axis=1)

                    right = np.sum(label == predicted_y)
                    sum_right += right
                    sum_num += 8
                    acc = sum_right/sum_num
                    
            logger.warning(f"epoch{epoch+1} accuracy:{acc}")
            if acc > self.best_acc:
                path = os.path.join(self.args.checkpoint_dir, "best.pt")
                self.best_acc = acc
                self.save_checkpoint(epoch+1)
                logger.warning(f"Epoch{epoch+1} saved")


    def save_checkpoint(self, iteration):
        d = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer1_state_dict": self.optimizer_1.state_dict(),
            "optimizer2_state_dict": self.optimizer_2.state_dict(),
            "scheduler1": self.scheduler_1.state_dict(),
            "scheduler2": self.scheduler_2.state_dict(),
        }

        torch.save(d, os.path.join(self.args.checkpoint_dir, "best_checkpoint.pt"))

    def load_checkpoint(self):
        checkpoint = torch.load(
                os.path.join(self.args.checkpoint_dir, "best_checkpoint.pt"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer_1.load_state_dict(checkpoint["optimizer1_state_dict"])
        self.optimizer_2.load_state_dict(checkpoint["optimizer2_state_dict"])
        self.scheduler_1.load_state_dict(checkpoint["scheduler1"])
        self.scheduler_2.load_state_dict(checkpoint["scheduler2"])
        self.start_iteration = checkpoint['iteration']


def main():
    learner = Learner()
    learner.train()


main()
