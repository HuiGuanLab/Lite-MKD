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
from torchstat import stat

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

class Evaluator:
    def __init__(self):
        

        self.args = prepare_test_args()
#------------------------------------------------------
        

        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')  
#-------------------------------------------------------------------------------------------------

        if self.args.debug==False:
            

            self.logger=logs(self.args)       
#----------------------------------------------------------------------------------------------


            self.logger.log_args(vars(self.args))
            self.logger.error("test checkpoint Directory: %s\n" % self.args.test_model_path)  
#-------------------------------------------------------------------------------------------------   


        print(self.args.test_model)
        self.model = select_test(self.args)   
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
        badcase_logit_list=[]
        badcase_class_list=[]
        badcase_class_list_real=[]
        class_dict_total={}
        class_dict_TP={}
        class_dict_TN={}
        class_dict_FP={}
        class_dict_FN={}
        self.model.eval()
        with torch.no_grad():
            self.video_loader.dataset.train = False  
            
            accuracy_dict ={}   
            accuracies = []  
            
            accuracies_ce = []
            accuracies_kd = []
            accuracies_55 = []
            accuracies_19 = []
            accuracies_28 = []
            accuracies_37 = []
            accuracies_46 = []
            accuracies_64 = []
            accuracies_73 = []
            accuracies_82 = []
            accuracies_91 = []
            accuracies_1020 = []
            
            iteration = 0   
            item = self.args.dataset  #item
            
            for task_dict in self.video_loader:  
                if iteration >= self.args.num_test_tasks:
                    break
                iteration += 1

                context_images,target_images,\
                    context_teacher_feature,target_teacher_feature, \
                                    context_labels, target_labels, real_target_labels,\
                                        batch_class_list  = self.prepare_task(task_dict) 
                
                if self.args.test_model =='student' :                         
                    model_dict=self.model(context_images,context_labels,target_images) 
                elif self.args.test_model =='teacher' : 
                    model_dict=self.model(context_teacher_feature,context_labels,target_teacher_feature)
                    
                #---------------------------------fc2----------------------------------    
                model_logits=model_dict['logits']   
                model_logits=model_logits.to(self.device)
                # model_logits_fc1=model_dict['logits']['fc_1']
                # model_logits_fc2=model_dict['logits']['fc_2']
                
                # model_logits_fc1=model_logits_fc1.to(self.device)
                # model_logits_fc2=model_logits_fc2.to(self.device)
                
                # model_logits=model_logits_fc1+model_logits_fc2
                
                # model_logits_ce=model_logits_fc1
                # model_logits_kd=model_logits_fc2
                
                # model_logits_55=0.5*model_logits_fc1+0.5*model_logits_fc2
                
                # model_logits_19=0.1*model_logits_fc1+0.9*model_logits_fc2
                # model_logits_28=0.2*model_logits_fc1+0.8*model_logits_fc2
                # model_logits_37=0.3*model_logits_fc1+0.7*model_logits_fc2
                # model_logits_46=0.4*model_logits_fc1+0.6*model_logits_fc2
                # model_logits_64=0.6*model_logits_fc1+0.4*model_logits_fc2
                # model_logits_73=0.7*model_logits_fc1+0.3*model_logits_fc2
                # model_logits_82=0.8*model_logits_fc1+0.2*model_logits_fc2
                # model_logits_91=0.9*model_logits_fc1+0.1*model_logits_fc2
                
                # model_logits_1020=model_logits_fc1+2*model_logits_fc2
                
                
                target_labels = target_labels.to(self.device)
                
                # print('shape')
                # print(model_logits.shape)
                # print(target_labels.shape)
                # print(batch_class_list.shape)
                # print('label')
                # print(model_logits)
                # print(target_labels)
                # print(batch_class_list)
                support_label=context_labels.reshape(5,5)
                # print(context_labels)
                # print(support_label)
                
                accuracy = self.accuracy_fn(model_logits, target_labels)
                
                # accuracy_ce = self.accuracy_fn(model_logits_ce, target_labels)
                # accuracy_kd = self.accuracy_fn(model_logits_kd, target_labels)
                # accuracy_55 = self.accuracy_fn(model_logits_55, target_labels)
                # accuracy_19 = self.accuracy_fn(model_logits_19, target_labels)
                # accuracy_28 = self.accuracy_fn(model_logits_28, target_labels)
                # accuracy_37 = self.accuracy_fn(model_logits_37, target_labels)
                # accuracy_46 = self.accuracy_fn(model_logits_46, target_labels)
                # accuracy_64 = self.accuracy_fn(model_logits_64, target_labels)
                # accuracy_73 = self.accuracy_fn(model_logits_73, target_labels)
                # accuracy_82 = self.accuracy_fn(model_logits_82, target_labels)
                # accuracy_91 = self.accuracy_fn(model_logits_91, target_labels)
                # accuracy_1020 = self.accuracy_fn(model_logits_1020, target_labels)
 

                print(iteration)
                print('任务准确率：')
                print(accuracy)

                
                #---------------------badcase----------------------------------
                # if accuracy< 0.8:
                #      badcase_class_list.append(batch_class_list)
                #      real_batch=[]
                #      for bcl in batch_class_list:
                #          real_batch.append(real_class[int(bcl)])

                #      badcase_class_list_real.append(real_batch)
                    
                
                # for idex,i in enumerate(model_logits) :  #每个任务有5way，这里依次遍历，就是说5个分类子任务，每个子任务是个分类问题
                #     badcasedict(class_dict_total,real_class[int(batch_class_list[target_labels[idex]])][0])  #记录下序号一的标签的实际类，target_labels[idex]是第一个标签的序号类
                #     if i.argmax()== target_labels[idex]:
                #         # print("相等")
                #         badcasedict(class_dict_TP,real_class[int(batch_class_list[target_labels[idex]])][0])  #相等，存相等的实际类
                #         for idex2,j in enumerate(support_label[idex]): #第一个（idex）分类问题的supportset的标签，j是其中每个的序号类
                #             if idex2!=i.argmax(): #序号类的序号不等于相等的类
                #                 badcasedict(class_dict_TN,real_class[int(batch_class_list[int(j)])][0])
                        
                #     else:
                #         # print('不相等')
                #         badcasedict(class_dict_FP,real_class[int(batch_class_list[i.argmax()])][0])  #假如错了，记录选则类
                #         for idex2,j in enumerate(support_label[idex]): #第一个（idex）分类问题的supportset的标签，j是其中每个的序号类
                #             if idex2!=i.argmax(): #序号类的序号不等于相等的类
                #                 badcasedict(class_dict_FN,real_class[int(batch_class_list[int(j)])][0])
                        # badcase_logit_list.append(i)
                        # badcase_class_list.append(batch_class_list)
                            
                #--------------------------------------------------------------
                # print('我的list')
                # print(badcase_class_list)
                # print(class_dict_total)
                # print(class_dict_TP)
                # print(class_dict_TN)
                # print(class_dict_FP)
                
                
                
                # for idex,i in enumerate(model_logits) :

                #     if i.argmax()== target_labels[idex]:
                #         print("相等")
                #     else:
                #         print('不相等')
                #         while i.argmax()!= target_labels[idex]:
                #             print(model_logits[idex][target_labels[idex]])
                #             model_logits[idex][target_labels[idex]]=model_logits[idex][target_labels[idex]]+1
                
                
                # for idex,i in enumerate(model_logits) :
                #     if i.argmax()== target_labels[idex]:
                #         print("相等")
                #     else:
                #         while i.argmax()!= target_labels[idex]:
                #             model_logits[idex][target_labels[idex]]=model_logits[idex][target_labels[idex]]+1

                       
                if self.args.debug==False:
                    self.logger.info("For Task: {0},  Testing Accuracy is {1}".format(self.args.mode ,
                            accuracy.item()))  
                    
                accuracies.append(accuracy.item())
                
                # accuracies_ce.append(accuracy_ce.item())
                # accuracies_kd.append(accuracy_kd.item())
                # accuracies_19.append(accuracy_19.item())
                # accuracies_28.append(accuracy_28.item())
                # accuracies_37.append(accuracy_37.item())
                # accuracies_46.append(accuracy_46.item())
                # accuracies_55.append(accuracy_55.item())
                # accuracies_64.append(accuracy_64.item())
                # accuracies_73.append(accuracy_73.item())
                # accuracies_82.append(accuracy_82.item())
                # accuracies_91.append(accuracy_91.item())
                # accuracies_1020.append(accuracy_1020.item())


                del model_logits
                
                # del model_logits_ce
                # del model_logits_kd
                
                # del model_logits_55
                
                # del model_logits_19
                # del model_logits_28
                # del model_logits_37
                # del model_logits_46

                # del model_logits_64
                # del model_logits_73
                # del model_logits_82
                # del model_logits_91
                # del model_logits_1020

            
            accuracy = np.array(accuracies).mean() * 100.0
            # accuracy_ce = np.array(accuracies_ce).mean() * 100.0
            # accuracy_kd = np.array(accuracies_kd).mean() * 100.0
            # accuracy_55 = np.array(accuracies_55).mean() * 100.0
            # accuracy_19 = np.array(accuracies_19).mean() * 100.0
            # accuracy_28 = np.array(accuracies_28).mean() * 100.0
            # accuracy_37 = np.array(accuracies_37).mean() * 100.0
            # accuracy_46 = np.array(accuracies_46).mean() * 100.0
            # accuracy_64 = np.array(accuracies_64).mean() * 100.0
            # accuracy_73 = np.array(accuracies_73).mean() * 100.0
            # accuracy_82 = np.array(accuracies_82).mean() * 100.0
            # accuracy_91 = np.array(accuracies_91).mean() * 100.0
            # accuracy_1020 = np.array(accuracies_1020).mean() * 100.0

            confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
            
            print(accuracy)
            # print(accuracy_ce)
            # print(accuracy_kd)
            # print(accuracy_55)
            # print(accuracy_19)
            # print(accuracy_28)
            # print(accuracy_37)
            # print(accuracy_46)
            # print(accuracy_64)
            # print(accuracy_73)
            # print(accuracy_82)
            # print(accuracy_91)
            # print(accuracy_1020)
            print(confidence)

            
            
            # Tp=pd.DataFrame(pd.Series(class_dict_TP))
            # Fp=pd.DataFrame(pd.Series(class_dict_FP))
            # Tn=pd.DataFrame(pd.Series(class_dict_TN))
            # Fn=pd.DataFrame(pd.Series(class_dict_FN))
            # Tp.to_csv('tp.csv')
            # Fp.to_csv('fp.csv')
            # Tn.to_csv('tn.csv')
            # Fn.to_csv('fn.csv')
            
            # badcase=pd.DataFrame(badcase_class_list_real)
            # badcase.to_csv('badcase.csv')
            
    
            accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence, "loss": loss}
            self.logger.info("For Task: {0},  and Testing Accuracy is {1}".format(self.args.mode, accuracy))

        

    def prepare_task(self, task_dict, images_to_device = True):
        
        '''
        TODO：
        这一块可以做一些封装
        对于不同的数据输入情况做一些映射
        
        现在就是把所有需要读到的数据全部传了进来
        有rgb原始，老师特征，rgb，flow，depth的特征，还有标签并把他们放到cuda上
        '''
        
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]

        context_teacher_feature = task_dict['support_set_feature_teacher'][0]
        target_teacher_feature = task_dict['target_set_feature_teacher'][0]
        
        
        # context_feature_rgb = task_dict['support_set_rgb'][0]
        # target_feature_rgb = task_dict['target_set_rgb'][0]
        
        
        # context_feature_flow = task_dict['support_set_flow'][0]
        # target_feature_flow = task_dict['target_set_flow'][0]
        
        
        # context_feature_depth = task_dict['support_set_depth'][0]
        # target_feature_depth = task_dict['target_set_depth'][0]
        

        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]
        
        # real_class=task_dict['real_class']
        if images_to_device:

            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)

            context_teacher_feature = context_teacher_feature.to(self.device)
            target_teacher_feature = target_teacher_feature.to(self.device)
            
            # context_feature_rgb = context_feature_rgb.to(self.device)
            # target_feature_rgb = target_feature_rgb.to(self.device)
            
            # context_feature_flow = context_feature_flow.to(self.device)
            # target_feature_flow = target_feature_flow.to(self.device)
            
            # context_feature_depth = context_feature_depth.to(self.device)
            # target_feature_depth = target_feature_depth.to(self.device)

        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images,target_images,context_teacher_feature,target_teacher_feature,context_labels, target_labels, real_target_labels, batch_class_list 


    def init_data(self):
        
        train_set = [self.args.dataset]  
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        
        return train_set, validation_set, test_set

        

def badcasedict(baddict,num):
        if num in baddict :
            baddict[num]=baddict[num]+1
        else:
            baddict[num]=1
            
        return baddict       

def eval_main():
    evaluator = Evaluator()
    accuracy_dict=evaluator.test()
    print(accuracy_dict)


if __name__ == '__main__':
    eval_main()
