import os
import numpy as np
import torch
import sys
sys.path.append('model')
sys.path.append('model/classifers')
import distillers as distillers
import video_reader
from log import logs
from model.model_select import Student, Teacher, select_test
from options import prepare_test_args, prepare_train_args
from utils import TestAccuracies, aggregate_accuracy, loss
import pandas as pd
from thop import profile

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

class Evaluator:
    def __init__(self):
        
        self.args = prepare_test_args()
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')   
        print(self.args.test_model)
        self.model = Student(self.args)   
        self.model = self.model.to(self.device)
        if self.args.num_gpus > 1:
            self.model.distribute_model()  
        self.train_set, self.validation_set, self.test_set = self.init_data()  
        self.model.eval()
#-----------------------------------------------------------------------------------------------------
        self.vd = video_reader.VideoDataset(self.args)   
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.args.num_workers)  
#---------------------------------------------------------------------------------------------------------------------------        
        self.accuracy_fn = aggregate_accuracy  
        self.test_accuracies = TestAccuracies(self.test_set)
#-----------------------------------------------------------------------------------------------------                  
        self.loss = loss
          

    def test(self):
  
        self.model.eval()
        with torch.no_grad():
            self.video_loader.dataset.train = False  
            iteration = 0     
            item = self.args.dataset  
            for task_dict in self.video_loader:  
                if iteration >= self.args.num_test_tasks:
                    break
                iteration += 1

                context_images,target_images,\
                    context_teacher_feature,target_teacher_feature, \
                                    context_labels, target_labels, real_target_labels,\
                                        batch_class_list  = self.prepare_task(task_dict)  
                macs, params = profile(self.model, inputs=(context_images,target_labels,target_images,))
                print(macs)
                print(params)
                break
       
        

    def prepare_task(self, task_dict, images_to_device = True):
        
        
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]

        context_teacher_feature = task_dict['support_set_feature_teacher'][0]
        target_teacher_feature = task_dict['target_set_feature_teacher'][0]
        

        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]
        

        if images_to_device:

            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)

            context_teacher_feature = context_teacher_feature.to(self.device)
            target_teacher_feature = target_teacher_feature.to(self.device)

        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images,target_images,context_teacher_feature,target_teacher_feature,context_labels, target_labels, real_target_labels, batch_class_list 


    def init_data(self):
        
        train_set = [self.args.dataset]  
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        
        return train_set, validation_set, test_set
    

def eval_main():
    evaluator = Evaluator()
    accuracy_dict=evaluator.test()


if __name__ == '__main__':
    eval_main()
