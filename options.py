import argparse
import os
import sys
import torch


def parse_common_args(parser):  
        gpu_device = 'cuda'
        device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
    
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")  #5
        parser.add_argument("--shot", type=int, default=5, help="Shots per class.")  #5
        parser.add_argument("--query_per_class", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")  #这里改成了1
        parser.add_argument("--query_per_class_test", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        

        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--print_freq", type=int, default=10, help="print and log every n iterations.") #10次打印一次
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.") #不改
        parser.add_argument("--num_workers", type=int, default=1, help="Num dataloader workers.")  #不改 10
        parser.add_argument("--trans_linear_out_dim", type=int, default=1152, help="Transformer linear_out_dim")  #不改，transfomer那边的线性层参数
        parser.add_argument("--trans_linear_in_dim", type=int, default=2048, help="Transformer linear_in_dim")  #不改，transfomer那边的线性层参数
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")  #不改
        parser.add_argument('--temp_set', nargs='+', type=int, help='cardinalities e.g. 2,3 is pairs and triples', default=[2])   #参数输入，用trm就使用2！！
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")  #不改
        parser.add_argument("--save_freq", type=int, default=10000, help="Number of iterations between checkpoint saves.")  #5000保存不改
        parser.add_argument("--split", type=int, default=3, help="Dataset split.")  #默认3，都用的3
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[20000,40000])   #学习率衰减
        parser.add_argument("--num_test_tasks", type=int, default=5000, help="number of random tasks to test on.") 
        parser.add_argument("--device",  default=device, help="device")
        
        

        parser.add_argument("--method", choices=["resnet18", "resnet34", "resnet50"], default="resnet18", help="method")  #backbone用18还是50
        parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to split the ResNet over")  #参数输入用几块卡!!
        parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf"], default="kinetics", help="Dataset to use.")  #选择数据集  ！！
        parser.add_argument("--mode",  default='KD_KL_meta', help="experiment description")  #本次实验的简单描述，搞清楚这个实验干了什么
        parser.add_argument("--debug", type=bool, default=False, help="debug mode")  #启用debug模式，降不存储模型，不使用日志等功能，仅仅看程序能否运行
        parser.add_argument("--distill_name",  default='KD', help="distill experiment name")  #本次实验的简单描述，搞清楚这个实验干了什么
        parser.add_argument("--model_backbone",  default='strm18_student', help="backbone name")
        parser.add_argument("--model_classifier",  default='TRX', help="classifier name")
        parser.add_argument("--model_teacher", choices=["cos",'e_dist','e_dist_fc2sup','e_dist_fc2_sup',"train_teacher","test_teacher","TRX_sup","TRX_sup_fixed","test_teacher_TRX_sup_fixed","test_teacher_TRX_2fcsup_fixed"], default='test_teacher', help="teacher name")
        parser.add_argument("--teacher_checkpoint",  default='/home/zty/baseline3.0/best_teahcer/checkpoint25000.pt', help="classifier name")
        parser.add_argument("--test_model", choices=["teacher", "student","extract_feature"], default='teacher', help="test who")
        
        
        parser.add_argument("--soft_loss_weight",  default=1, help="experiment Hyperparameter")  #本次实验的简单描述，搞清楚这个实验干了什么
        parser.add_argument("--hard_loss_weight",  default=1, help="experiment Hyperparameter")  #本次实验的简单描述，搞清楚这个实验干了什么
        parser.add_argument("--test", type=bool, default=False, help="experiment Hyperparameter")  #本次实验的简单描述，搞清楚这个实验干了什么
        parser.add_argument("--cfg",  default={
            'soft_loss_weight_support':1,
            'soft_loss_weight_query':1,
            'hard_loss_weight':1, 
            'soft_loss_weight':2,
            'feature_loss_weight':1,
            'temperature':4,
            'fcwsl_aerfa':0.5,
            'fcwsl_beta':1
        }, help="experiment Hyperparameter")  

        return parser

def parse_train_args(parser):  
    parser = parse_common_args(parser)
    

    parser.add_argument("--checkpoint_dir", "-c", default=None, help="Directory to save checkpoint to.")
    parser.add_argument("--training_iterations", "-i", type=int, default=100010, help="Number of meta-training iterations.")  #先迭代4万次
    parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")#断训重连
    parser.add_argument('--test_iters', nargs='+', type=int, help='iterations to test at. Default is for ssv2 otam split.', default=[10,10000,15000,20000,30000,35000,40000,50000,60000,70000,80000,90000,100000])
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.0001, help="Learning rate.")  #不改定死0.0001
    parser.add_argument("--opt", choices=["adam", "sgd"], default="sgd", help="Optimizer")  #不改
   

    return parser

def parse_test_args(parser):   
    parser = parse_common_args(parser)

    parser.add_argument("--test_model_path", "-m", default='/home/zty/baseline2.0/kinect_cheakpoint__KL_KD_meta_temp4/checkpoint20000.pt', help="Path to model to load and test.") 
      
    
    return parser

def args_cheak(args): 
    
    if args.checkpoint_dir == None:  
            print("need to specify a checkpoint dir")
            exit(1)
            
    if args.method == "resnet50":
        args.trans_linear_in_dim = 2048
    else:
        args.trans_linear_in_dim = 512
        
    if args.mode=="train_res18_trx":
            args.method = "resnet18"
            args.trans_linear_in_dim = 2048
            
    verify_checkpoint_dir(args.checkpoint_dir,args.resume_from_checkpoint, False)  
            
    return args
        
def verify_checkpoint_dir(checkpoint_dir, resume, test_mode):  
        if resume:  # verify that the checkpoint directory and file exists  
            if not os.path.exists(checkpoint_dir):
                print("Can't resume for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
                sys.exit()

            checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
            if not os.path.isfile(checkpoint_file):
                print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
                sys.exit()
        else:
            if os.path.exists(checkpoint_dir):
                print("Checkpoint directory ({}) already exits.".format(checkpoint_dir), flush=True)
                print("If starting a new training run, specify a directory that does not already exist.", flush=True)
                print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
                sys.exit()
            else:
                os.makedirs(checkpoint_dir)
        

def get_data_path(args):   

    if args.dataset == "ssv2":
            args.traintestlist = os.path.join(args.scratch, "video_datasets/splits/somethingsomethingv2TrainTestlist")
            args.path = os.path.join(args.scratch, "video_datasets/data/somethingsomethingv2_256x256q5_7l8.zip")
            
    elif args.dataset == "kinetics":
        
            args.traintestlist=os.path.join("data", "kinetics/splits/kineticsTrainTestlist")
            args.RGB_path =os.path.join("data", "kinetics/l8/rgb_l8")
            args.teacher_path = os.path.join("/home/zty/204/data/kinetics","feature/multi_feature_deit")
            
            
    elif args.dataset == "ucf":
        
            args.traintestlist = os.path.join("data", "ucf101/splits/ucf_ARN")
            args.RGB_path = os.path.join("data", "ucf101/l8/rgb_l8")
            args.teacher_path = os.path.join("/home/zty/204/data","ucf101/feature/multi_feature")
            
            
    elif args.dataset == "hmdb":
        
            args.traintestlist = os.path.join("data", "hmdb/splits/hmdb_ARN")
            args.RGB_path = os.path.join("data", "hmdb/l8/rgb_l8")
            args.teacher_path = os.path.join("data", "hmdb/feature/new_feature/multi_feature")

            
            
    return args

def get_train_args():  
    
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    
    
    return args

def get_test_args():  
    
    parser = argparse.ArgumentParser()
    parser = parse_test_args(parser)
    args = parser.parse_args()
   
    return args

def prepare_train_args():  

    args = get_train_args()
    args = args_cheak(args)
    args = get_data_path(args)
    args.debug=False
    return args

def prepare_train_args_wandb():  
    args = get_train_args()
    args = get_data_path(args)
    args.debug=False
    return args

def prepare_test_args():  
    args = get_test_args()
    args = get_data_path(args)
    return args

if __name__ == '__main__':
    train_args = prepare_train_args()
    test_args = prepare_test_args()