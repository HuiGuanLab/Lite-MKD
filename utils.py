import torch
import torch.nn.functional as F
import os
import math
from enum import Enum
import sys


class TestAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets and we deem
    the evaluation to be better if more than half of the validation accuracies on the individual validation datsets
    are better than the previous best.
    """

    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)

    def print(self, logfile, accuracy_dict):
        print_and_log(logfile, "")  # add a blank line
        print_and_log(logfile, "Test Accuracies:")
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")  # add a blank line




def print_and_log(log_file, message):
    """
    Helper function to print to the screen and the cnaps_layer_log.txt file.
    """
    print(message, flush=True)
    log_file.write(message + '\n')


def get_assist_log(checkpoint_dir):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return logfile


def stack_first_dim(x):
    """
    Method to combine the first two dimension of an array
    """
    x_shape = x.size()
    new_shape = [x_shape[0] * x_shape[1]]
    if len(x_shape) > 2:
        new_shape += x_shape[2:]
    return x.view(new_shape)


def split_first_dim_linear(x, first_two_dims):
    """
    Undo the stacking operation  
    """
    x_shape = x.size()  #20,5
    new_shape = first_two_dims  #[1,20]
    if len(x_shape) > 1:  #2
        new_shape += [x_shape[-1]]  #5
    return x.view(new_shape)


def sample_normal(mean, var, num_samples):
    """
    Generate samples from a reparameterized normal distribution
    :param mean: tensor - mean parameter of the distribution
    :param var: tensor - variance of the distribution
    :param num_samples: np scalar - number of samples to generate
    :return: tensor - samples from distribution of size numSamples x dim(mean)
    """
    sample_shape = [num_samples] + len(mean.size())*[1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()

def KDloss(teacher_feature,student_feature,device):
    loss=F.mse_loss(teacher_feature,student_feature).to(device)
    return loss

def KLDivloss(teacher_logits,student_logits,device):
    temp=4
    criterion=torch.nn.KLDivLoss(reduction='sum')
    x=torch.nn.functional.log_softmax(student_logits/temp,dim=-1)
    y=torch.nn.functional.softmax(teacher_logits/temp,dim=-1)
    klloss=criterion(x,y)
    return klloss

def loss(test_logits_sample, test_labels, device):
    """
    Compute the classification loss.
    """
    size = test_logits_sample.size()  #1,20
    sample_count = size[0]  # scalar for the loop counter  1
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)  #1

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)  #1,20
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)  #在batch为1时没有任何用，
    return -torch.sum(score, dim=0)


def aggregate_accuracy(test_logits_sample, test_labels):
    """
    Compute classification accuracy.
    """
    averaged_predictions = test_logits_sample
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())

def task_confusion(test_logits, test_labels, real_test_labels, batch_class_list):
    preds = torch.argmax(torch.logsumexp(test_logits, dim=0), dim=-1)
    real_preds = batch_class_list[preds]
    return real_preds

def linear_classifier(x, param_dict):
    """
    Classifier.
    """
    return F.linear(x, param_dict['weight_mean'], param_dict['bias_mean'])


'''Abandoned ideas'''
# def wslloss(teacher_logits,student_logits,test_labels,device):

#     temp=2
#     teacher_logits=teacher_logits[0] #20,5
#     student_logits=student_logits[0] #20,5
    
    
#     teacher_T_softmax=torch.nn.functional.softmax(teacher_logits/temp,dim=-1)
#     student_T_softmax=torch.nn.functional.log_softmax(student_logits/temp,dim=-1) #20,5
#     teahcer_student_loss=torch.squeeze(-torch.sum(teacher_T_softmax*student_T_softmax,-1,keepdim=True))  #20
#     # print(teahcer_student_loss.shape)
    
#     teacher_logits_nograd=teacher_logits.detach()
#     student_logits_nograd=student_logits.detach()
    
#     log_softmax_t = torch.nn.functional.log_softmax(teacher_logits_nograd,dim=-1) #20,5
#     log_softmax_s = torch.nn.functional.log_softmax(student_logits_nograd,dim=-1) #logsoftmax
    
#     one_hot_label = F.one_hot(test_labels, num_classes=5).float() #20,5

    
#     softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, -1, keepdim=True)  #20,1
#     softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, -1, keepdim=True)
    
    
#     focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
#     ratio_lower = torch.zeros(1).cuda()
#     focal_weight = torch.max(focal_weight, ratio_lower)
#     focal_weight = 1 - torch.exp(- focal_weight)
    
#     softmax_loss = focal_weight * teahcer_student_loss
#     soft_loss = (temp ** 2) * torch.mean(softmax_loss)
#     return soft_loss




# def WSLDistiller(student_logits, target_label,teacher_logits):

#         T=2
#         alpha=2.5
        
#         s_input_for_softmax = student_logits / T  #温度
#         t_input_for_softmax = teacher_logits / T  #温度

#         t_soft_label = torch.nn.functional.softmax(t_input_for_softmax)  #教师过softmax

#         softmax_loss = - torch.sum(t_soft_label * torch.nn.functional.logsoftmax(s_input_for_softmax), 1, keepdim=True)  #学生老师交叉熵

#         fc_s_auto = fc_s.detach()
#         fc_t_auto = fc_t.detach()#去除梯度
#         log_softmax_s = self.logsoftmax(fc_s_auto)
#         log_softmax_t = self.logsoftmax(fc_t_auto) #logsoftmax
#         one_hot_label = F.one_hot(label, num_classes=1000).float()
#         softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
#         softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

#         focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
#         ratio_lower = torch.zeros(1).cuda()
#         focal_weight = torch.max(focal_weight, ratio_lower)
#         focal_weight = 1 - torch.exp(- focal_weight)
#         softmax_loss = focal_weight * softmax_loss

#         soft_loss = (self.T ** 2) * torch.mean(softmax_loss)

#         hard_loss = self.hard_loss(fc_s, label)

#         loss = hard_loss + self.alpha * soft_loss

#         return fc_s, loss
    
    
    
# def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    
#     logits_student=logits_student[0] #20,5
#     logits_teacher=logits_teacher[0] #20,5
    
#     gt_mask = _get_gt_mask(logits_student, target)    #目标类的mask
#     other_mask = _get_other_mask(logits_student, target)  #非目标类的mask
#     pred_student = F.softmax(logits_student / temperature, dim=1)  #学生预测
#     pred_teacher = F.softmax(logits_teacher / temperature, dim=1)  #教师预测
#     pred_student = cat_mask(pred_student, gt_mask, other_mask)    
#     pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)    
#     log_pred_student = torch.log(pred_student)     #学生log
#     tckd_loss = (
#         F.kl_div(log_pred_student, pred_teacher, size_average=False)  
#         * (temperature**2)
#         / target.shape[0]
#     )
#     pred_teacher_part2 = F.softmax(
#         logits_teacher / temperature - 1000.0 * gt_mask, dim=1
#     )
#     log_pred_student_part2 = F.log_softmax(
#         logits_student / temperature - 1000.0 * gt_mask, dim=1
#     )
#     nckd_loss = (
#         F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
#         * (temperature**2)
#         / target.shape[0]
#     )
#     return alpha * tckd_loss + beta * nckd_loss


# def _get_gt_mask(logits, target):
#     target = target.reshape(-1)
#     mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()  #把gt位置都填1，其他为0
#     return mask


# def _get_other_mask(logits, target):
#     target = target.reshape(-1)
#     mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()   #把非GT位置填1，GT位置填0
#     return mask


# def cat_mask(t, mask1, mask2):
#     t1 = (t * mask1).sum(dim=1, keepdims=True)
#     t2 = (t * mask2).sum(1, keepdims=True)
#     rt = torch.cat([t1, t2], dim=1)
#     return rt
#---------------------------------------------------------------------------------------



def prepare_task(self, task_dict, images_to_device = True):
        
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]

        context_teacher_feature = task_dict['support_set_feature_teacher'][0]
        target_teacher_feature = task_dict['target_set_feature_teacher'][0]
        
        
        context_feature_rgb = task_dict['support_set_rgb'][0]
        target_feature_rgb = task_dict['target_set_rgb'][0]
        
        
        context_feature_flow = task_dict['support_set_flow'][0]
        target_feature_flow = task_dict['target_set_flow'][0]
        
        
        context_feature_depth = task_dict['support_set_depth'][0]
        target_feature_depth = task_dict['target_set_depth'][0]
        

        real_target_labels = task_dict['real_target_labels'][0]
        batch_class_list = task_dict['batch_class_list'][0]

        if images_to_device:

            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)

            context_teacher_feature = context_teacher_feature.to(self.device)
            target_teacher_feature = target_teacher_feature.to(self.device)
            
            context_feature_rgb = context_feature_rgb.to(self.device)
            target_feature_rgb = target_feature_rgb.to(self.device)
            
            context_feature_flow = context_feature_flow.to(self.device)
            target_feature_flow = target_feature_flow.to(self.device)
            
            context_feature_depth = context_feature_depth.to(self.device)
            target_feature_depth = target_feature_depth.to(self.device)

        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images,target_images,context_teacher_feature,target_teacher_feature, context_feature_rgb,target_feature_rgb,context_feature_flow,target_feature_flow,context_feature_depth,target_feature_depth,context_labels, target_labels, real_target_labels, batch_class_list  