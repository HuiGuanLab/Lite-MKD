import argparse
import logging
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import pywebio
import tensorflow as tf
import multi_video_reader
import torch
import torchvision
import yaml
from pywebio.input import FLOAT, input
from pywebio.output import *
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from yaml import parse
import json
from model import TRX, FourStrm, ThreeStrm
from utils import (TestAccuracies, aggregate_accuracy, get_log_files, loss,
                   print_and_log, task_confusion, verify_checkpoint_dir)
# import wandb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Quiet TensorFlow warnings

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',
                              datefmt=" %Y-%m-%d %H:%M:%S")


def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


# logger for training accuracies
train_logger = setup_logger("Training_accuracy",
                            "./runs_strm/train_output.log")

# logger for evaluation accuracies
eval_logger = setup_logger("Evaluation_accuracy",
                           "./runs_strm/eval_output.log")

#############################################
# setting up seeds
manualSeed = 0
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
########################################################


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        (
            self.checkpoint_dir,
            self.logfile,
            self.checkpoint_path_validation,
            self.checkpoint_path_final,
        ) = get_log_files(self.args.checkpoint_dir,
                          self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile,
                      "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = "cuda"
        self.device = torch.device(
            gpu_device if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()

        self.vd = multi_video_reader.MultiVideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(
            self.vd, batch_size=1, num_workers=self.args.num_workers)

        self.loss = loss
        self.accuracy_fn = aggregate_accuracy

        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.args.learning_rate)
        self.test_accuracies = TestAccuracies(self.test_set)

        self.scheduler = MultiStepLR(self.optimizer,
                                     milestones=self.args.sch,
                                     gamma=0.1)

        self.start_iteration = 0

        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()
        # wandb.init(config=self.args, project="multi_fusion")

    def init_model(self):
        file_name = "model"
        class_name = self.args.model
        modules = __import__(file_name)
        model = getattr(modules, class_name)(self.args)
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
            choices=["ssv2", "kinetics", "hmdb", "ucf", "dance"],
            default="ssv2",
            help="Dataset to use.",
        )
        parser.add_argument("--learning_rate",
                            "-lr",
                            type=float,
                            default=0.001,
                            help="Learning rate.")
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
        parser.add_argument("--test_model_path", type=str)
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
        parser.add_argument("--way",
                            type=int,
                            default=5,
                            help="Way of each task.")
        parser.add_argument("--shot",
                            type=int,
                            default=5,
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
            default=[5000, 10000, 15000, 20000, 25000, 30000],
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
        parser.add_argument("--seq_len",
                            type=int,
                            default=8,
                            help="Frames per video.")
        parser.add_argument("--num_workers",
                            type=int,
                            default=10,
                            help="Num dataloader workers.")
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
        parser.add_argument("--opt",
                            choices=["adam", "sgd"],
                            default="sgd",
                            help="Optimizer")
        parser.add_argument("--trans_dropout",
                            type=int,
                            default=0.1,
                            help="Transformer dropout")
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
        parser.add_argument("--split",
                            type=int,
                            default=7,
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
            default=False,
            help="Only testing the model from the given checkpoint",
        )
        parser.add_argument(
            "--fixed_test_eposide",
            type=str,
            default="splits/hmdb_ARN/fixed_test.json",
            help="fixed test episodes",
        )
        parser.add_argument("--a", type=float, default=1.0)
        parser.add_argument("--b", type=float, default=0.3)
        parser.add_argument("--c", type=float, default=0.3)

        parser.add_argument("--m1", type=str, default="rgb")
        parser.add_argument("--m2", type=str, default="skeleton")
        parser.add_argument("--m3", type=str, default="flow")
        parser.add_argument("--m4", type=str, default="depth")
        parser.add_argument("--m5", type=str, default="TG")
        parser.add_argument("--model", type=str, default="ThreeStrm")
        parser.add_argument("--shirt_num", type=int, default=1)
        parser.add_argument("--trans_num", type=int, default=4)
        parser.add_argument("--demo", type=bool, default=False)
        parser.add_argument("--extract", type=bool, default=False)
        parser.add_argument("--base_model", type=str, default="TRX")

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
                args.scratch,
                "video_datasets/splits/somethingsomethingv2TrainTestlist")
            args.path = os.path.join(
                args.scratch,
                "video_datasets/data/somethingsomethingv2_256x256q5_7l8.zip",
            )
        elif args.dataset == "kinetics":
            args.traintestlist = os.path.join(
                args.scratch, "video_datasets/splits/kineticsTrainTestlist")
            args.path = os.path.join(args.scratch,
                                     "video_datasets/data/kinetics/rgb_l8")

        elif args.dataset == "ucf":
            args.traintestlist = os.path.join(
                args.scratch, "video_datasets/splits/ucfTrainTestlist")
            args.path = os.path.join(args.scratch,
                                     "video_datasets/data/ucf/rgb_l8")
        elif args.dataset == "hmdb":
            args.traintestlist = os.path.join(
                args.scratch, "video_datasets/splits/hmdb51TrainTestlist")
            args.path = os.path.join(args.scratch,
                                     "video_datasets/data/hmdb/rgb_l8")
        elif args.dataset == "dance":
            args.traintestlist = os.path.join(
                args.scratch, "video_datasets/splits/danceTrainTestlist")
            args.path = os.path.join(args.scratch,
                                     "video_datasets/data/dance/all_view_rgb_l8/rgb_Camera_0")
        args.feature_save_path = f"imp_datasets/video_datasets/data/{args.dataset}_{args.method}_{args.base_model}_feature"
        # args.path = os.path.join(args.scratch, "video_datasets/data/hmdb51_jpegs_256.zip")

        with open("args.pkl", "wb") as f:
            pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            train_accuracies = []
            losses = []
            total_iterations = self.args.training_iterations

            iteration = self.start_iteration

            if self.args.demo:
                self.demo()
                exit()

            if self.args.extract:
                print("Model being extract feature")
                self.load_checkpoint()
                self.extract(session, 1)
                exit()
                
            if self.args.test_model_only:
                print("Model being tested at path: " +
                      self.args.test_model_path)
                self.load_checkpoint()
                logging.info("开始测试")
                accuracy_dict = self.test(session, 1)
                print(accuracy_dict)
                self.logfile.close()
                return

            for task_dict in self.video_loader:
                if iteration >= total_iterations:
                    break
                iteration += 1
                torch.set_grad_enabled(True)

                task_loss, task_accuracy = self.train_task(task_dict)
                train_accuracies.append(task_accuracy)
                losses.append(task_loss)

                # optimize
                if ((iteration + 1) % self.args.tasks_per_batch
                        == 0) or (iteration == (total_iterations - 1)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.scheduler.step()
                if (iteration + 1) % self.args.print_freq == 0:
                    # print training stats
                    print_and_log(
                        self.logfile,
                        "Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}"
                        .format(
                            iteration + 1,
                            total_iterations,
                            torch.Tensor(losses).mean().item(),
                            torch.Tensor(train_accuracies).mean().item(),
                        ),
                    )
                    train_logger.info(
                        "For Task: {0}, the training loss is {1} and Training Accuracy is {2}"
                        .format(
                            iteration + 1,
                            torch.Tensor(losses).mean().item(),
                            torch.Tensor(train_accuracies).mean().item(),
                        ))
                    lr = self.get_lr(self.optimizer)
                    # wandb.log({"loss":torch.Tensor(losses).mean().item(), "lr":lr})
                    avg_train_acc = torch.Tensor(
                        train_accuracies).mean().item()
                    avg_train_loss = torch.Tensor(losses).mean().item()

                    train_accuracies = []
                    losses = []

                if ((iteration + 1) % self.args.save_freq
                        == 0) and (iteration + 1) != total_iterations:
                    self.save_checkpoint(iteration + 1)

                if ((iteration + 1) in self.args.test_iters
                    ) and (iteration + 1) != total_iterations:
                    accuracy_dict = self.test(session, iteration + 1)
                    print(accuracy_dict)
                    self.test_accuracies.print(self.logfile, accuracy_dict)

            # save the final model
            torch.save(self.model.state_dict(), self.checkpoint_path_final)

        self.logfile.close()

    def train_task(self, task_dict):
        (
            context_fea,
            target_fea,
            context_labels,
            target_labels,
            real_target_labels,
            batch_class_list,
        ) = self.prepare_task_for_fusion(task_dict)
        model_dict = self.model(context_fea, context_labels, target_fea)
        target_logits = model_dict["logits"]
        target_logits = target_logits.to(self.device)
        target_labels = target_labels.to(self.device)
        # print(target_logits.shape, target_labels)
        # print(target_logits.shape, target_labels.shape)
        task_loss = (self.loss(target_logits, target_labels, self.device) /
                     self.args.tasks_per_batch)
        # loss = torch.nn.CrossEntropyLoss()
        # task_loss = loss(target_logits, target_labels)

        # Add the logits before computing the accuracy
        target_logits = target_logits
        task_accuracy = self.accuracy_fn(target_logits, target_labels)
        task_loss.backward(retain_graph=False)
        return task_loss, task_accuracy

    def test(self, session, num_episode):
        self.model.eval()
        with torch.no_grad():
            self.video_loader.dataset.train = False
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
                target_logits = self.model(context_fea, context_labels,
                                           target_fea)['logits'].to(
                                               self.device)
                target_labels = target_labels.to(self.device)
                accuracy = self.accuracy_fn(target_logits, target_labels)

                # Loss using the new distance metric after  patch-level enrichment

                eval_logger.info(
                    f"For Task: {iteration + 1}, Testing Accuracy is {round(accuracy.item(), 3)}, Mean Accuracy is {np.array(accuracies).mean() * 100.0}"
                )
                
                # wandb.log({"acc": round(accuracy.item(), 3)})
                
                accuracies.append(accuracy.item())
                del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / \
                    np.sqrt(len(accuracies))
            accuracy_dict = {item: {"accuracy": accuracy, "confidence": confidence}}
            eval_logger.info("For Task: {0},  Testing Accuracy is {1}".format(
                num_episode, accuracy))
            print_and_log(
                self.logfile, "For Task: {0},  Testing Accuracy is {1}".format(
                    num_episode, accuracy))
            self.video_loader.dataset.train = True

        self.model.train()

        return accuracy_dict

    def extract(self, session, num_episode):
        self.model.eval()
        l = []
        d = {}
        f = open("run.log", "w")
        with torch.no_grad():
            self.video_loader.dataset.train = False
            accuracies = []
            accuracies_1 = []
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
                    support_c_v,
                    target_c_v,
                    multi_support_feature,
                    multi_target_feature
                ) = self.prepare_c_v_task(task_dict)
                print("开始提取特征")
                fusion_context, fusion_target = self.model.extract_task_feature(context_fea, context_labels,
                                           target_fea)
                prefix = os.path.join('/'.join(self.args.path.split('/')[:-2]), f'{self.args.dataset}_feature', 'multi_feature')
                l_s = [os.path.join(i[0][0], i[1][0]) for i in support_c_v]
                for i, s in enumerate(l_s):
                    d[s] = d.get(s, [])
                    d[s].append(l_s)
                    if len(d[s]) != 1:
                        path = os.path.join(prefix, s, 'feature.npy')
                        print(np.load(path), np.load(path).shape)
                        print('上面是第一次提取的数据--------------------下面是第二次提取的数据')
                        print(fusion_context.cpu().numpy()[i], fusion_context.cpu().numpy()[i].shape)
                        print(path, len(l_s), i, s)
                        print(d[s][0])
                        print(d[s][1])
                        exit()

                l_q = [os.path.join(i[0][0], i[1][0]) for i in target_c_v]
                for i in l_q:
                    d[i] = d.get(i, [])
                    d[i].append(l_q)
                
                target_logits = self.model.feature_test(fusion_context, context_labels, fusion_target)['logits'].to(self.device)
                target_logits_1 = self.model.feature_test(multi_support_feature, context_labels, multi_target_feature)['logits'].to(self.device)

                for i, data in enumerate(support_c_v):
                    c_v = '/'.join(list(data[0])+list(data[1]))
                    path = os.path.join(prefix, c_v)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    np.save(os.path.join(path, "feature.npy"), fusion_context.cpu().numpy()[i])

                for i, data in enumerate(target_c_v):
                    c_v = '/'.join(list(data[0])+list(data[1]))
                    path = os.path.join(prefix, c_v)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    np.save(os.path.join(path, "feature.npy"), fusion_target.cpu().numpy()[i])

                target_labels = target_labels.to(self.device)
                accuracy = self.accuracy_fn(target_logits, target_labels)
                accuracy_1 = self.accuracy_fn(target_logits_1, target_labels)

                # Loss using the new distance metric after  patch-level enrichment

                eval_logger.info(
                    f"For Task: {iteration + 1}, Testing Accuracy is {round(accuracy.item(), 3)}, Mean Accuracy is {np.array(accuracies).mean() * 100.0}"
                )
                eval_logger.info(
                    f"For Task: {iteration + 1}, * Testing Accuracy is {round(accuracy_1.item(), 3)},* Mean Accuracy is {np.array(accuracies_1).mean() * 100.0}"
                )

                accuracies.append(accuracy.item())
                accuracies_1.append(accuracy_1.item())
                del target_logits

            accuracy = np.array(accuracies).mean() * 100.0
            accuracy_1 = np.array(accuracies_1).mean() * 100.0
            confidence = (196.0 * np.array(accuracies).std()) / \
                        np.sqrt(len(accuracies))
            accuracy_dict = {item: {"accuracy": accuracy, "confidence": confidence}}
            eval_logger.info("For Task: {0},  Testing Accuracy is {1}".format(
                num_episode, accuracy))
            eval_logger.info("For Task: {0},  Testing1 Accuracy is {1}".format(
                num_episode, accuracy_1))
            print_and_log(
                self.logfile, "For Task: {0},  Testing Accuracy is {1}".format(
                    num_episode, accuracy))
            self.video_loader.dataset.train = True
        f.close
        self.model.train()
        return

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def demo(self, prefix="/media/disk4/zp/dataset/kinetics"):
        self.model.eval()
        with torch.no_grad():
            self.video_loader.dataset.train = False
            accuracy_dict = {}
            accuracies = []
            iteration = 0
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

                target_logits = self.model(context_fea, context_labels,
                                           target_fea)['logits'].to(
                                               self.device)
                                               
                target_labels = target_labels.to(self.device)
                averaged_predictions = torch.logsumexp(target_logits, dim=0)
                predict = torch.argmax(averaged_predictions, dim=-1)
                print(predict, target_labels)
                accuracy = self.accuracy_fn(target_logits, target_labels)
                target_labels = target_labels.cpu().numpy().tolist()
                predict = predict.cpu().numpy().tolist()
                print(task_dict['target_c_v'])
                # Loss using the new distance metric after  patch-level enrichment
                d = {}
                support_path = []
                for data in task_dict['support_c_v']:
                    c, v = data[0][0], data[1][0] + '.mp4'
                    path = os.path.join(self.args.dataset, c, v)
                    support_path.append(path)
                d["support"] = support_path

                target_path = []
                index2class = {}
                real_class = []
                for i, data in enumerate(task_dict['target_c_v']):
                    c, v = data[0][0], data[1][0] + '.mp4'
                    path = os.path.join(self.args.dataset, c, v)
                    # predict_c, predict_v = task_dict['support_c_v'][data+i*5][0][0], task_dict['support_c_v'][data+i*5][1][0]+'.mp4'
                    # predict_path = os.path.join(self.args.dataset, predict_c, predict_v)
                    target_path.append(path)
                    index2class[target_labels[i]] = c
                    real_class.append(c)
                d['target'] = target_path
                predict = [index2class[i] for i in predict]
                d['predict'] = {'predict': predict, 'real': real_class}
                print(f"Testing Accuracy is {round(accuracy.item(), 3)}")
                with open('lib/demo.yaml', 'w') as f:
                    yaml.dump(d, f)

                def web():
                    try:
                        with open('lib/demo.yaml') as f:
                            d = yaml.load(f, Loader=yaml.FullLoader)
                    except Exception as e:
                        print(e)
                    pywebio.session.set_env(output_max_width="1600px")
                    # scope = pywebio.output.put_scope('scope')
                    dd = defaultdict(list)
                    for i in d['support']:
                        html = pywebio.output.put_html(
                            f'<video controls="controls" width="250" height="250" src="{i}"></video>'
                        )
                        c = i.split('/')[-2]
                        dd[c].append(html)
                    # pywebio.output.put_html(f'<video controls="controls" src="{i}"></video>'.format(url=i))

                    l = []
                    query_video = [
                        pywebio.output.put_html(
                            '<video controls="controls" width="250" height="250" src="{}"></video>'
                            .format(i)) for i in d['target']
                    ]

                    put_table([['query'] + query_video,
                               ['predict'] + d['predict']['predict'],
                               ['real'] + d['predict']['real']]).style("margin-left: 50px")

                    out = [[
                        span('action\support'), 'support1', 'support2',
                        'support3', 'support4', 'support5'
                    ]] + [[key] + dd[key] for key in dd]

                    with use_scope("scope") as scope:
                        put_table(out)

                pywebio.start_server(web, port=8089)

                del target_logits
                break

        self.model.train()

        return accuracy_dict

    def prepare_task_for_fusion(self, task_dict, images_to_device=True):
        context_fea, context_labels = (
            task_dict["support_fea"],
            task_dict["support_labels"][0],
        )
        
        target_fea, target_labels = (
            task_dict["target_fea"],
            task_dict["target_labels"][0],
        )

        real_target_labels = task_dict["real_target_labels"][0]
        batch_class_list = task_dict["batch_class_list"][0]

        if images_to_device:
            for i, key in enumerate(context_fea):
                context_fea[key] = context_fea[key].to(self.device)
            for i, key in enumerate(target_fea):
                target_fea[key] = target_fea[key].to(self.device)
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

    def prepare_task(self, task_dict, images_to_device=True):
        context_fea, context_labels = (
            task_dict["support_fea"],
            task_dict["support_labels"][0],
        )
        target_fea, target_labels = (
            task_dict["target_fea"],
            task_dict["target_labels"][0],
        )
        real_target_labels = task_dict["real_target_labels"][0]
        batch_class_list = task_dict["batch_class_list"][0]

        if images_to_device:
            for i, key in enumerate(context_fea):
                context_fea[key] = context_fea[key].cuda(i %
                                                         self.args.num_gpus)
            for i, key in enumerate(target_fea):
                target_fea[key] = target_fea[key].cuda(i % self.args.num_gpus)
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

    def prepare_c_v_task(self, task_dict, images_to_device=True):
        context_fea, context_labels = (
            task_dict["support_fea"],
            task_dict["support_labels"][0],
        )
        target_fea, target_labels = (
            task_dict["target_fea"],
            task_dict["target_labels"][0],
        )
        real_target_labels = task_dict["real_target_labels"][0]
        batch_class_list = task_dict["batch_class_list"][0]
        support_c_v = task_dict['support_c_v']
        target_c_v = task_dict['target_c_v']
        multi_support_feature = task_dict["multi_support_feature"][0].to(self.device)
        multi_target_feature = task_dict["multi_target_feature"][0].to(self.device)
        if images_to_device:
            for i, key in enumerate(context_fea):
                context_fea[key] = context_fea[key].cuda(i %
                                                         self.args.num_gpus)
            for i, key in enumerate(target_fea):
                target_fea[key] = target_fea[key].cuda(i % self.args.num_gpus)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return (
            context_fea,
            target_fea,
            context_labels,
            target_labels,
            real_target_labels,
            batch_class_list,
            support_c_v,
            target_c_v,
            multi_support_feature,
            multi_target_feature
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
            d,
            os.path.join(self.checkpoint_dir,
                         "checkpoint{}.pt".format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, "checkpoint.pt"))

    def load_checkpoint(self):
        if self.args.test_model_only:
            checkpoint = torch.load(self.args.test_model_path)
        elif self.args.resume_from_checkpoint is True:
            checkpoint = torch.load(
                os.path.join(self.checkpoint_dir, "checkpoint.pt"))
        elif self.args.extract == True:
            checkpoint = torch.load(self.args.test_model_path)
            #     rgb_checkpoint = torch.load(self.args.rgb_test_model_path)
            #     skeleton_checkpoint = torch.load(self.args.skeleton_test_model_path)
            #     flow_checkpoint = torch.load(self.args.flow_test_model_path)
            self.start_iteration = checkpoint["iteration"]
        #     self.model.rgb_branch.load_state_dict(f(rgb_checkpoint["model_state_dict"]), strict=False)
        #     self.model.skeleton_branch.load_state_dict(f(skeleton_checkpoint["model_state_dict"]), strict=False)
        #     self.model.flow_branch.load_state_dict(f(flow_checkpoint["model_state_dict"]), strict=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_iteration = checkpoint['iteration']


if __name__ == "__main__":
    main()
