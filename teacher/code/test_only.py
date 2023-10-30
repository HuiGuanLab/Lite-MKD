from collections import OrderedDict
import logging
import random
import video_reader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import tensorflow as tf
import torch
import numpy as np
import argparse
import os
import pickle
from utils import (
    print_and_log,
    get_log_files,
    TestAccuracies,
    loss,
    aggregate_accuracy,
    verify_checkpoint_dir,
    task_confusion,
)
from model import TRX, ThreeStrm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Quiet TensorFlow warnings


formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# logger for training accuracies
train_logger = setup_logger(
    "Training_accuracy", "./runs_strm/train_output.log")

# logger for evaluation accuracies
eval_logger = setup_logger("Evaluation_accuracy",
                           "./runs_strm/eval_output.log")

#############################################
# setting up seeds
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
########################################################


def main():
    learner = Learner()
    learner.test_only()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        (
            self.checkpoint_dir,
            self.logfile,
            self.checkpoint_path_validation,
            self.checkpoint_path_final,
        ) = get_log_files(
            self.args.checkpoint_dir, self.args.resume_from_checkpoint, False
        )

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" %
                      self.checkpoint_dir)

        gpu_device = "cuda"
        self.device = torch.device(
            gpu_device if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()

        self.vd = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(
            self.vd, batch_size=1, num_workers=self.args.num_workers
        )

        self.loss = loss
        self.accuracy_fn = aggregate_accuracy

        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate
            )
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.args.learning_rate
            )
        self.test_accuracies = TestAccuracies(self.test_set)

        self.scheduler = MultiStepLR(
            self.optimizer, milestones=self.args.sch, gamma=0.1
        )

        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        model = ThreeStrm(self.args)
        model = model.to(self.device)
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        return train_set, validation_set, test_set

    """
    Command line parser
    """

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--dataset",
            choices=["ssv2", "kinetics", "hmdb", "ucf"],
            default="ssv2",
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
            default="bp",
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
        parser.add_argument("--split", type=int, default=7,
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
            "--feature_save_path", type=str, default=f"imp_datasets/video_datasets/data/kinetics_feature"
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

        if args.checkpoint_dir == None:
            print("need to specify a checkpoint dir")
            exit(1)

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
                args.scratch, "video_datasets/data/kinetics/rgb_l8"
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

        with open("args.pkl", "wb") as f:
            pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

        return args

    def test_only(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:

            if self.args.test_model_only:
                print("Model being tested at path: " +
                      self.args.test_model_path)
                self.load_checkpoint()
                accuracy_dict = self.test(session, 1)
                print(accuracy_dict)
                self.logfile.close()
                return

        self.logfile.close()

    def test(self, session, num_episode):
        self.model.eval()
        with torch.no_grad():
            self.video_loader.dataset.train = False
            accuracy_dict = {}
            accuracies = []
            losses = []
            iteration = 0
            item = self.args.dataset
            for task_dict in self.video_loader:
                if iteration >= self.args.num_test_tasks:
                    break
                iteration += 1

                (
                    context_fea,
                    target_fea,
                    context_labels,
                    target_labels,
                    real_target_labels,
                    batch_class_list,
                ) = self.prepare_task(task_dict)
                model_dict = self.model.test_only(
                    context_fea, context_labels, target_fea)
                target_logits = model_dict["logits"].to(self.device)

                # Target logits after applying query-distance-based similarity metric on patch-level enriched features
                # target_logits_post_pat = model_dict["logits_post_pat"].to(
                #     self.device)

                target_labels = target_labels.to(self.device)

                # Add the logits before computing the accuracy
                # target_logits = target_logits + 0.1 * target_logits_post_pat

                accuracy = self.accuracy_fn(target_logits, target_labels)

                loss = (
                    self.loss(target_logits, target_labels, self.device)
                    / self.args.num_test_tasks
                )

                # Loss using the new distance metric after  patch-level enrichment
                # loss_post_pat = (
                #     self.loss(target_logits_post_pat,
                #               target_labels, self.device)
                #     / self.args.num_test_tasks
                # )

                # Joint loss
                # loss = loss + 0.1 * loss_post_pat

                eval_logger.info(
                    "For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(
                        iteration + 1, loss.item(), accuracy.item()
                    )
                )
                losses.append(loss.item())
                accuracies.append(accuracy.item())
                del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / \
                np.sqrt(len(accuracies))
            loss = np.array(losses).mean()
            accuracy_dict[item] = {
                "accuracy": accuracy,
                "confidence": confidence,
                "loss": loss,
            }
            eval_logger.info(
                "For Task: {0}, the testing loss is {1} and Testing Accuracy is {2}".format(
                    num_episode, loss, accuracy
                )
            )

            self.video_loader.dataset.train = True
        self.model.train()

        return accuracy_dict

    def prepare_task(self, task_dict, images_to_device=True):
        context_fea, context_labels = (
            task_dict["support_fea"][0],
            task_dict["support_labels"][0],
        )
        target_fea, target_labels = (
            task_dict["target_fea"][0],
            task_dict["target_labels"][0],
        )
        real_target_labels = task_dict["real_target_labels"][0]
        batch_class_list = task_dict["batch_class_list"][0]
        if images_to_device:
            context_fea = context_fea.to(self.device)
            target_fea = target_fea.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return (
            context_fea,
            target_fea,
            context_labels,
            target_labels,
            real_target_labels,
            batch_class_list,
        )

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def save_checkpoint(self, iteration):
        d = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        torch.save(
            d, os.path.join(self.checkpoint_dir,
                            "checkpoint{}.pt".format(iteration))
        )
        torch.save(d, os.path.join(self.checkpoint_dir, "checkpoint.pt"))

    def load_checkpoint(self):
        if self.args.resume_from_checkpoint is True:
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, "checkpoint.pt"))
        else:
            checkpoint = torch.load(self.args.test_model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])



if __name__ == "__main__":
    main()
