import torch
import numpy as np
import argparse
import os
import pickle
from utils import  TestAccuracies,aggregate_accuracy
# from model import KD_KL_student_50, KD_fusion,KD_KL_student,KD_KL_teacher,ThreeTRXShiftLoopTime,KD_KL_student2
import threading
from torch.optim.lr_scheduler import MultiStepLR
import sys
sys.path.append('model')
sys.path.append('model/classifers')
print(sys.path)
import torchvision
import video_reader
import random 
import logging
from log import logs
from options import prepare_train_args_wandb
from torch.cuda.amp import autocast as autocast
import distillers as distiller
from model.model_select import Student,Teacher
import argparse
import time
import wandb

#setting up seeds
manualSeed = random.randint(1, 10000)  
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)


args = prepare_train_args_wandb()  
debug=False
aerfa=0.5
def model_pipeline(hyperparameters):
     with wandb.init(project="experience-6", config=hyperparameters,name='hmdb-KD'):  
         
         config=wandb.config
         '''参数设置'''
         gpu_device = 'cuda'
         device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
         config.update({"device":device})

         #----------------------------------------------------------------------------------------------------   
         student,teacher,video_loader,distillers,accuracy_fn,test_accuracies,optimizer,scheduler=make(config)
         
         wandb.watch(student)
         train(student,teacher,video_loader,distillers,optimizer,scheduler,accuracy_fn,config)
         
     return student
                 
def init_model(config):
    '''
    The backbone inputs raw video, and outputs the features of the student
    The classifier inputs the features output by the backbone, and outputs the score of the student.
    The teacher inputs the fixed features of the teacher, and outputs the score of the teacher.
    '''
    student=Student(config)
    teacher=Teacher(config)
    student=student.to(config.device)
    teacher=teacher.to(config.device)
            
    return student,teacher

def init_data(config):
        
    train_set = [config.dataset]  
    validation_set = [config.dataset]
    test_set = [config.dataset]
        
    return train_set, validation_set, test_set     

def make(config):
    
     '''Initialization of the model and data'''
     student,teacher= init_model(config)   
     train_set,  validation_set, test_set = init_data(config)  
    #-----------------------------------------------------------------------------------------------------
         
     '''Initialization of dataset '''
     
     vd = video_reader.VideoDataset(config)   
     video_loader = torch.utils.data.DataLoader(vd, batch_size=1, num_workers=config.num_workers)  
    #---------------------------------------------------------------------------------------------------------------------------      
        
     '''Initialization of distillers'''
     distillers=distiller.Distiller(config.distill_name,config.cfg,config.device)
    #-----------------------------------------------------------------------------------------------------   
        
     '''Calculator class for accuracy'''
     accuracy_fn = aggregate_accuracy  
     test_accuracies = TestAccuracies(test_set)
    #-----------------------------------------------------------------------------------------------------   
    
     ''' optimizer '''
     if config.opt == "adam":
        optimizer = torch.optim.Adam(student.parameters(), lr=config.learning_rate)
     elif config.opt == "sgd":  
        optimizer = torch.optim.SGD(student.parameters(), lr=config.learning_rate)
     scheduler = MultiStepLR(optimizer, milestones=config.sch, gamma=0.1)    
     optimizer.zero_grad()  
#------------------------------------------------------------------------------------------------
         
     return student,teacher,video_loader,distillers,accuracy_fn,test_accuracies,optimizer,scheduler
  
def train(student,teacher,video_loader,distiller,optimizer,scheduler,accuracy_fn,config):
        
        train_accuracies = []
        losses = []  
        
        task_train_accuracies = []
        task_losses = [] 
        
        total_iterations = config.training_iterations  
        iteration = 0

        for task_dict in video_loader:
            if iteration >= total_iterations:
                            break
            iteration += 1
            with autocast():
                        
                torch.set_grad_enabled(True)  

                task_loss, task_accuracy,task_accuracy_dict = train_task(task_dict,student,teacher,distiller,accuracy_fn,config)  #训练
                print("accuracy'{}".format(iteration+1))
                print(task_accuracy_dict['accuracy'])

                
                train_accuracies.append(task_accuracy) 
                losses.append(task_loss)
                task_losses.append(task_loss)
                task_train_accuracies.append(task_accuracy)

                '''Print or perform gradient descent operations according to iteration numbers '''
                if ((iteration + 1) % config.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                    optimizer.step()
                    optimizer.zero_grad()
                    
                scheduler.step()  
    
    #------------------------------------------------------------------------------------------------------------ 

                '''Print and logging'''           
                if (iteration + 1) % config.print_freq == 0:
                    print('Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                .format(iteration + 1, total_iterations, torch.Tensor(task_losses).mean().item(),
                                                    torch.Tensor(task_train_accuracies).mean().item()))
                    

                avg_train_acc = torch.Tensor(train_accuracies).mean().item()
                avg_train_loss = torch.Tensor(losses).mean().item()
                
                
                if debug==False:
                    wandb.log({
                        'avg_train_acc':avg_train_acc,
                        'avg_train_loss':avg_train_loss,
                        'task_losses':task_losses
                        })
                                
                task_train_accuracies = []  
                task_losses = []  
    #------------------------------------------------------------------------------------------------------------                
                    
                '''Save parameters '''
                if ((iteration + 1) % config.save_freq == 0) and (iteration + 1) != total_iterations:
                    checkpoint_dict = {
                    'iteration': iteration, 
                    'model_state_dict': student.state_dict(), 
                    }
                    current_time=time.strftime('%Y%m%d%H%M',time.localtime(time.time() ))
                    torch.save(checkpoint_dict,'model_save/{}{}{}.pt'.format(current_time,config.mode,iteration) ) 
                    if debug==False:
                        wandb.save('model_save/{}{}{}.pt'.format(current_time,config.mode,iteration))
                        
    #------------------------------------------------------------------------------------------------------------
                        

                '''Call testing, or validation, but here they use the testing set for validation. '''
                if ((iteration + 1) in config.test_iters) and (iteration + 1) != total_iterations:
                    accuracy_dict = test(student,video_loader,accuracy_fn,config)
                    print(accuracy_dict)
                
def train_task(task_dict,student,teacher,distiller,accuracy_fn,config):  
        
    '''
        This is the default for distillation, that is, initialize two models
        
    '''
    context_images,target_images, \
        context_teacher_feature,target_teacher_feature,\
                        context_labels, target_labels, \
                            real_target_labels, batch_class_list \
                                = prepare_task(task_dict,config.device)  
    model_dict = student(context_images, context_labels, target_images)  

    teacher_model_dict = teacher(context_teacher_feature,context_labels,target_teacher_feature)

        
    target_logits = model_dict['logits']   
    teacher_logits = teacher_model_dict['logits']  

#-------------------------------------Prepare for experiments that require feature.-----------------------------------------------  
    if config.distill_name=='KL_feature':
            
        context_stduent_feature =model_dict['context_features']
        target_stduent_feature =model_dict['target_features']
            
        Student_feature =torch.cat([context_stduent_feature,target_stduent_feature],0)
        teacher_feature =torch.cat([context_teacher_feature,target_teacher_feature],0)
  
        target_logits = {
            'logits':target_logits,
            'feature':Student_feature
        }
            
        teacher_logits = {
            'logits':teacher_logits,
            'feature':teacher_feature
        }       

 #------------------------------------------------------------------------------------------------     
 
    target_labels = target_labels.to(config.device)
    loss=getattr(distiller, config.distill_name)(target_logits, teacher_logits, target_labels)
    task_loss= loss['loss'] 
    
    if debug==False:
        wandb.log({
            # 'soft_loss':loss['soft_loss'],                           
            # 'hard_loss':loss['hard_loss'],                                
            "task_loss":task_loss,
            # 'weight':loss['aerfa']                                       
        })
#---------------------------For some different distillation experiment settings, generally please select 'fc_2_sup_dist'
    if config.distill_name=='support_sim':
        target_logits=target_logits['query'].to(config.device)
    elif config.distill_name=='KL_feature':
        target_logits=target_logits['logits'].to(config.device)       

    elif config.distill_name=='fc_2_sup_dist':
        
        target_logits_fc1=target_logits['kl'].to(config.device)
        target_logits_fc2=target_logits['ce'].to(config.device)
        
        task_accuracy_ce = accuracy_fn(target_logits_fc1, target_labels)
        task_accuracy_kl = accuracy_fn(target_logits_fc2, target_labels)
        
        target_logits = target_logits_fc1+target_logits_fc2

        target_logits = target_logits_fc1+target_logits_fc2

    elif config.distill_name=='strm_fc_2_sup_dist':
        target_logits_pat=target_logits['pat'].to(config.device)
        target_logits_fr1=target_logits['fr1'].to(config.device)
        target_logits_fr2=target_logits['fr2'].to(config.device)
        target_logits = 0.2*target_logits_pat+target_logits_fr1+target_logits_fr2        
     
    elif config.distill_name=='strm':
        target_logits_pat=target_logits['pat'].to(config.device)
        target_logits_fr=target_logits['fr'].to(config.device)
        target_logits = 0.1*target_logits_pat+target_logits_fr

    elif config.distill_name=='strm_KD':
        target_logits_pat=target_logits['pat'].to(config.device)
        target_logits_fr=target_logits['fr'].to(config.device)
        target_logits = 0.1*target_logits_pat+target_logits_fr
    else:
        target_logits=target_logits.to(config.device)
        
        
    task_accuracy = accuracy_fn(target_logits, target_labels)
    task_accuracy_dict={
        'accuracy':task_accuracy,
        # 'accuracy_ce':task_accuracy_ce,                                               
        # 'accuracy_kl':task_accuracy_kl,                                              
    }
            
    task_loss.backward(retain_graph=False)

    return task_loss, task_accuracy ,task_accuracy_dict
           
def test_task(task_dict,model,accuracy_fn,config):
    
        model=model 

        context_images,target_images, \
                context_teacher_feature,target_teacher_feature,\
                                context_labels, target_labels, \
                                    real_target_labels, batch_class_list \
                                        = prepare_task(task_dict,config.device)  

        if config.test_model =='student' :                         
            model_dict=model(context_images,context_labels,target_images) 
            
        elif config.test_model =='teacher' : 
            model_dict=model(context_teacher_feature,context_labels,target_teacher_feature) 
            
        if config.distill_name=='strm':
            model_logits_pat=model_dict['logits']['pat']
            model_logits_fr=model_dict['logits']['fr']
            model_logits=0.1*model_logits_pat+model_logits_fr
            model_logits=model_logits.to(config.device)
            
        elif config.distill_name=='strm_KD':
            model_logits_pat=model_dict['logits']['pat']
            model_logits_fr=model_dict['logits']['fr']
            model_logits=0.1*model_logits_pat+model_logits_fr
            model_logits=model_logits.to(config.device)
            
        elif config.distill_name=='fc_2_sup_dist':
        
            model_logits_fc1=model_dict['logits']['kl']
            model_logits_fc2=model_dict['logits']['ce']
            
            model_logits_fc1=model_logits_fc1.to(config.device)
            model_logits_fc2=model_logits_fc2.to(config.device)
            model_logits=model_logits_fc1+model_logits_fc2
            
        elif config.distill_name=='strm_fc_2_sup_dist':
            
            model_logits_pat=model_dict['logits']['pat']
            model_logits_fr1=model_dict['logits']['fr1']
            model_logits_fr2=model_dict['logits']['fr2']
        
            model_logits_pat=model_logits_pat.to(config.device)
            model_logits_fr1=model_logits_fr1.to(config.device)
            model_logits_fr2=model_logits_fr2.to(config.device)

            model_logits=0.2*model_logits_pat+model_logits_fr1+model_logits_fr2

        elif config.distill_name=='support_sim':
            
            model_logits=model_logits['query'].to(config.device)
            model_logits=model_logits.to(config.device)
            
        else:
            model_logits=model_dict['logits'].to(config.device)
            
        target_labels = target_labels.to(config.device)
        
        test_accuracy_dict={
            'test_accuracy':accuracy_fn(model_logits, target_labels),
            # 'test_accuracy_ce':ce_model_logits,  #sup                            
            # 'test_accuracy_kl':kl_model_logits,  #                  
        }
        # del target_logits
        del model_logits
        # del target_logits_ce
        # del target_logits_kl
        return test_accuracy_dict
      
def test(model,video_loader,accuracy_fn,config):
        
        '''
        Test the training model, which is designed according to the pattern of distillation. 
        By default, the student model is tested, and if you want to test another model, 
        you need to modify the model passed to the 'test_task' argument.
        '''
        model.eval()
        print("开始测试")
        
        with torch.no_grad():

                video_loader.dataset.train = False  
                
                accuracy_dict ={}
                accuracies = []
                accuracies_ce= []
                accuracies_kl= []
                
                iteration = 0
                item = config.dataset
                
                for task_dict in video_loader:  
                    if iteration >= config.num_test_tasks:
                        break
                    iteration += 1
                    
                    task_accuracy_dict = test_task(task_dict,model,accuracy_fn,config)  
                    
                    if ( iteration % 1000 )== 0:
                        print('test_accuracy ：', iteration ,task_accuracy_dict['test_accuracy'])
                        # print('测试ce ：', iteration ,task_accuracy_dict['test_accuracy_ce'])                       
                        # print('测试kl ：', iteration ,task_accuracy_dict['test_accuracy_kl'])                       
                        
                    accuracies.append(task_accuracy_dict['test_accuracy'].item())
                    # accuracies_ce.append(task_accuracy_dict['test_accuracy_ce'].item())                             
                    # accuracies_kl.append(task_accuracy_dict['test_accuracy_kl'].item())                            
                    
                '''准确率的计算'''    
                accuracy = np.array(accuracies).mean() * 100.0
                # accuracy_ce = np.array(accuracies_ce).mean() * 100.0                                             
                # accuracy_kl = np.array(accuracies_kl).mean() * 100.0                                             
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))
                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
                
                if debug==False:
                    
                    wandb.log({
                        "test_accuracy":accuracy,
                        # "test_accuracy_ce":accuracy_ce,                                       
                        # "test_accuracy_kl":accuracy_kl                                          
                        
                    })
                
                ''' 变回训练采样模式'''
                video_loader.dataset.train = True
                
        model.train()   
        return accuracy_dict

def prepare_task(task_dict, device,images_to_device = True):

    context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
    target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]

    context_teacher_feature = task_dict['support_set_feature_teacher'][0]
    target_teacher_feature = task_dict['target_set_feature_teacher'][0]


    real_target_labels = task_dict['real_target_labels'][0]
    batch_class_list = task_dict['batch_class_list'][0]

    if images_to_device:

        context_images = context_images.to(device)
        target_images = target_images.to(device)

        context_teacher_feature = context_teacher_feature.to(device)
        target_teacher_feature = target_teacher_feature.to(device)
            

    context_labels = context_labels.to(device)
    target_labels = target_labels.type(torch.LongTensor).to(device)

    return context_images,target_images,context_teacher_feature,target_teacher_feature,context_labels, target_labels, real_target_labels, batch_class_list  

def main_debug(hyperparameters):

    config=hyperparameters 
         
    student,teacher,video_loader,distillers,accuracy_fn,test_accuracies,optimizer,scheduler=make(config)
         
    train(student,teacher,video_loader,distillers,optimizer,scheduler,accuracy_fn,config)
         

if __name__ == "__main__":
    
    if debug:
        main_debug(args)
    else:
        mdeol=model_pipeline(args)
