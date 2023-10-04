import torch
import torch.nn.functional as F
import os
import math
import sys

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    # print(log_pred_student)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    # print(pred_teacher)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    # print(loss_kd)
    loss_kd *= temperature**2
    return loss_kd
#------------------------------Dist----------------------------------        

def cosine_similarity(x, y, eps=1e-8):
        return (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)


def pearson_correlation(x, y, eps=1e-8):
        return cosine_similarity(x - x.mean(1).unsqueeze(1), y - y.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    y_s=y_s.softmax(dim=1)
    y_t=y_t.softmax(dim=1)

    return 1 - pearson_correlation(y_s, y_t).mean()
    
    

class Distiller(object):
    def __init__(self, distill_name,distill_cfg,device):
        self.distill_name = distill_name
        self.distill_dict = distill_cfg
        # self.hard_weight=hard_weight
        # self.soft_weight=soft_weight
        self.device=device
        
    def KD(self,student_logits,teacher_logits,test_labels):  #shape  20,5   20,5  20
        
        student_logits=student_logits.to(self.device)
        teacher_logits=teacher_logits.to(self.device)


        #--------------------旧版KA-------------------------------------
        # for idex,i in enumerate(teacher_logits):
        #     if i.argmax() != test_labels[idex]:
        #         while i.argmax()!= test_labels[idex]:
        #             teacher_logits[idex][test_labels[idex]]=teacher_logits[idex][test_labels[idex]]+1

        #---------------新版KA--------------------------------

        # student_logits=F.softmax(student_logits)
        # # print(student_logits.shape)

        # label=F.one_hot(test_labels)
        # # print(label.shape)
        # # print(student_logits+label)



        # log_pred_student = F.log_softmax(student_logits, dim=1)
        # pred_teacher = F.softmax(teacher_logits, dim=1) 
        # loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        # kl_loss = loss_kd*16       


        ce_loss=self.distill_dict['hard_loss_weight']*F.cross_entropy(student_logits, test_labels,) /16#20,5
        kl_loss=self.distill_dict['soft_loss_weight']*kd_loss(student_logits,teacher_logits,self.distill_dict['temperature'])

        return {'hard_loss': ce_loss, 'soft_loss': kl_loss, 'loss': ce_loss + kl_loss}
    
    def wsl(self,student_logits,teacher_logits,test_labels):
        
        student_logits=student_logits.to(self.device)
        teacher_logits=teacher_logits.to(self.device)
             
        teahcer_student_loss = kd_loss(student_logits,teacher_logits,self.distill_dict['temperature'])  #1


        teacher_logits_nograd = teacher_logits.detach()
        student_logits_nograd = student_logits.detach()

        softmax_loss_t = F.cross_entropy(teacher_logits_nograd, test_labels) #1
        softmax_loss_s = F.cross_entropy(student_logits_nograd, test_labels) #1

        focal_weight =  softmax_loss_s/ ( softmax_loss_t+ 1e-8)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        # print(focal_weight)
        soft_loss = focal_weight * teahcer_student_loss
        hard_loss = F.cross_entropy(student_logits, test_labels)/16

        return {'soft_loss': self.distill_dict['soft_loss_weight']*soft_loss, 'hard_loss': self.distill_dict['hard_loss_weight'] * hard_loss, 'loss': self.distill_dict['soft_loss_weight'] * soft_loss + self.distill_dict['hard_loss_weight'] * hard_loss}
    
    def ce(self,student_logits,teacher_logits,test_labels):
        
        student_logits=student_logits.to(self.device)
        # teacher_logits=teacher_logits['kl']
        # teacher_logits=teacher_logits.to(self.device)
        
        ce_loss=F.cross_entropy(student_logits, test_labels,)/16  #20,5

        return {'loss': ce_loss}
        
    def support_sim(self,student_logits,teacher_logits,test_labels):
        
        sim_studnet=student_logits['support_set'].to(self.device).reshape(20,25)
        sim_teacher=teacher_logits['support_set'].to(self.device).reshape(20,25)
        
        ori_student=student_logits['query'].to(self.device)
        ori_teacher=teacher_logits['query'].to(self.device)
        
        
        support_kl_loss=self.distill_dict['soft_loss_weight_support']*kd_loss(sim_studnet,sim_teacher,self.distill_dict['temperature'])
        ori_kl_loss=self.distill_dict['soft_loss_weight_query']*kd_loss(ori_student,ori_teacher,self.distill_dict['temperature'])
        
        ce_loss=self.distill_dict['hard_loss_weight']*F.cross_entropy(ori_student, test_labels,)/16  #20,5
        
        return {'hard_loss': ce_loss, 'soft_support_loss': support_kl_loss,'soft_query_loss':ori_kl_loss, 'loss': ce_loss + support_kl_loss + ori_kl_loss}
        
    def KL_feature(self,student_logits,teacher_logits,test_labels):
        
        student_logit = student_logits['logits'].to(self.device)
        teacher_logit = teacher_logits['logits'].to(self.device)  
        
        student_feature = student_logits['feature']
        teacher_feature = teacher_logits['feature']
        
        
        
        ce_loss=self.distill_dict['hard_loss_weight']*F.cross_entropy(student_logit, test_labels,) /16#20,5
        kl_loss=self.distill_dict['soft_loss_weight']*kd_loss(student_logit,teacher_logit,self.distill_dict['temperature'])
        feature_loss= self.distill_dict['feature_loss_weight']*F.mse_loss(student_feature,teacher_feature)
        
        # print('ce_loss')
        # print(ce_loss)
        
        # print('kl_loss')
        # print(kl_loss)
        
        # print('feature_loss')
        # print(feature_loss)
        
        
        return {'hard_loss': ce_loss, 'soft_loss': kl_loss, 'feature_loss':feature_loss,'loss': ce_loss + kl_loss + feature_loss}
    
    def fc_2(self,student_logits,teacher_logits,test_labels):
         teacher_logit = teacher_logits.to(self.device)  
        
         fc1_logits=student_logits['fc_1'].to(self.device)
         fc2_logits=student_logits['fc_2'].to(self.device)
         
         ce_loss=self.distill_dict['hard_loss_weight']*F.cross_entropy(fc1_logits, test_labels,) /16
         kl_loss=self.distill_dict['soft_loss_weight']*kd_loss(fc2_logits,teacher_logit,self.distill_dict['temperature'])
         
         return {'hard_loss': ce_loss, 'soft_loss': kl_loss, 'loss': ce_loss + kl_loss}
     
    def fc_2_wsl(self,student_logits,teacher_logits,test_labels):
         
        
        teacher_logit = teacher_logits.to(self.device)    #教师分数
        
        fc1_logits_ce=student_logits['fc_1'].to(self.device)  #学生ce分支分数
        fc2_logits_kd=student_logits['fc_2'].to(self.device)  #学生kd分支分数
        
        # student_logits=0.6*fc1_logits_ce+0.4*fc2_logits_kd

        
        # print('aerfa:')
        # print(aerfa)
        teacher_kd_loss = kd_loss(fc2_logits_kd,teacher_logit,self.distill_dict['temperature'])  #
        label_ce_loss = F.cross_entropy(fc1_logits_ce,test_labels)/16

        teacher_logit_nograd = teacher_logit.detach()
        fc1_logits_ce_nograd = fc1_logits_ce.detach()
        fc2_logits_kd_nograd = fc2_logits_kd.detach()
        # student_logits_nograd = student_logits.detach()


        softmax_loss_t = F.cross_entropy(fc2_logits_kd_nograd, test_labels) #
        softmax_loss_s_ce = F.cross_entropy(fc1_logits_ce_nograd, test_labels) #
        

        focal_weight =  softmax_loss_s_ce/ ( softmax_loss_t+ 1e-8)  
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        

        soft_loss = (1+focal_weight) * teacher_kd_loss
        hard_loss = (2-focal_weight) * label_ce_loss
        loss=soft_loss+hard_loss
        self.distill_dict['fcwsl_aerfa']=focal_weight
        
        
        return {'hard_loss': hard_loss, 'soft_loss': soft_loss, 'loss': loss,'aerfa':focal_weight}
        
    def strm(self,student_logits,teacher_logits,test_labels):
        
         teacher_logit = teacher_logits.to(self.device)  
        
         pat_logits=student_logits['pat'].to(self.device)
         fr_logits=student_logits['fr'].to(self.device)
         
         pat_loss=F.cross_entropy(pat_logits, test_labels,) /16
         fr_loss=F.cross_entropy(fr_logits, test_labels,) /16
         
         return {'pat_loss': pat_loss, 'fr_loss': fr_loss, 'loss': 0.1*pat_loss+ fr_loss}
     
    def strm_KD(self,student_logits,teacher_logits,test_labels):
        
         teacher_logit = teacher_logits.to(self.device)  
        
         pat_logits=student_logits['pat'].to(self.device)
         fr_logits=student_logits['fr'].to(self.device)
         
         kl_loss=self.distill_dict['soft_loss_weight']*kd_loss(fr_logits,teacher_logit,self.distill_dict['temperature'])
         
         pat_loss=F.cross_entropy(pat_logits, test_labels,) /16
         fr_loss=F.cross_entropy(fr_logits, test_labels,) /16
         
         return {'pat_loss': pat_loss, 'fr_loss': fr_loss,'softloss':kl_loss, 'loss': 0.1*pat_loss+ fr_loss+kl_loss}
     
    def fc_2_sup(self,student_logits,teacher_logits,test_labels):

        
        student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_ce = student_logits['ce'].to(self.device)
        
        # print(student_logits_sup.shape)
        # print(student_logits_kl.shape)
        
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        
        # teacher_logits_sup = teacher_logits_sup.fill_diagonal_(-100000)
        
        student_logits_kl_nograd = student_logits_kl.detach()
        student_logits_sup_nograd = student_logits_sup.detach()
        student_logits_ce_nograd = student_logits_ce.detach()
        teacher_logits_kl_nograd = teacher_logits_kl.detach()
        teacher_logits_sup_nograd = teacher_logits_sup.detach()


        softmax_loss_t = F.cross_entropy(student_logits_kl_nograd, test_labels) #
        softmax_loss_s_ce = F.cross_entropy(student_logits_ce_nograd, test_labels) #
        

        focal_weight =  softmax_loss_s_ce/ ( softmax_loss_t+ 1e-8)  
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        
        
        
        
        
        kl_loss = kd_loss(student_logits_kl,teacher_logits_kl,self.distill_dict['temperature'])  #
   
        sup_loss = kd_loss(student_logits_sup,teacher_logits_sup,self.distill_dict['temperature']) /16
        
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss = F.cross_entropy(student_logits_ce,test_labels)/16


        loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        # print('sup_loss')
        # print(0.01*sup_loss)
        
        # print('kl_loss')
        # print(kl_loss)
        
        # print('ce_sup_loss')
        # print(ce_sup_loss)
        
        return {'soft_loss': kl_loss, 'hard_loss': 0.01*sup_loss+ce_sup_loss,'loss': loss}
    
    def Dist_KD(self,student_logits,teacher_logits,test_labels):
        
        student_logits=student_logits.to(self.device)
        teacher_logits=teacher_logits.to(self.device)
        ce_loss=self.distill_dict['hard_loss_weight']*F.cross_entropy(student_logits, test_labels,) /16#20,5
        dist_loss=self.distill_dict['soft_loss_weight']*inter_class_relation(student_logits,teacher_logits)
        loss = ce_loss + dist_loss
        return {'soft_loss': dist_loss, 'hard_loss': ce_loss,'loss': loss}
    
    def fc_2_sup_dist(self,student_logits,teacher_logits,test_labels):
        
        student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_ce = student_logits['ce'].to(self.device)
        
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        
        # teacher_logits_sup = teacher_logits_sup.fill_diagonal_(-100000)
        
        
        # student_logits_kl_nograd = student_logits_kl.detach()
        # student_logits_sup_nograd = student_logits_sup.detach()
        # student_logits_ce_nograd = student_logits_ce.detach()
        # teacher_logits_kl_nograd = teacher_logits_kl.detach()
        # teacher_logits_sup_nograd = teacher_logits_sup.detach()


        # softmax_loss_t = F.cross_entropy(student_logits_kl_nograd, test_labels) #
        # softmax_loss_s_ce = F.cross_entropy(student_logits_ce_nograd, test_labels) #
        

        # focal_weight =  softmax_loss_s_ce/ ( softmax_loss_t+ 1e-8)  
        # ratio_lower = torch.zeros(1).cuda()
        # focal_weight = torch.max(focal_weight, ratio_lower)
        # focal_weight = 1 - torch.exp(- focal_weight)
        
        
        
        kl_loss = kd_loss(student_logits_kl,teacher_logits_kl,self.distill_dict['temperature'])  #
   
        sup_loss = inter_class_relation(student_logits_sup,teacher_logits_sup) 
        
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss = F.cross_entropy(student_logits_ce,test_labels)/16

        # loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        loss=kl_loss+0.5*sup_loss+ce_sup_loss
        
        return {'soft_loss': kl_loss, 'hard_loss': 0.5*sup_loss+ce_sup_loss,'loss': loss}
    
    def fc_2_sup_kl(self,student_logits,teacher_logits,test_labels):
        
        student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_ce = student_logits['ce'].to(self.device)
        
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        
        # teacher_logits_sup = teacher_logits_sup.fill_diagonal_(-100000)
        
        
        # student_logits_kl_nograd = student_logits_kl.detach()
        # student_logits_sup_nograd = student_logits_sup.detach()
        # student_logits_ce_nograd = student_logits_ce.detach()
        # teacher_logits_kl_nograd = teacher_logits_kl.detach()
        # teacher_logits_sup_nograd = teacher_logits_sup.detach()


        # softmax_loss_t = F.cross_entropy(student_logits_kl_nograd, test_labels) #
        # softmax_loss_s_ce = F.cross_entropy(student_logits_ce_nograd, test_labels) #
        

        # focal_weight =  softmax_loss_s_ce/ ( softmax_loss_t+ 1e-8)  
        # ratio_lower = torch.zeros(1).cuda()
        # focal_weight = torch.max(focal_weight, ratio_lower)
        # focal_weight = 1 - torch.exp(- focal_weight)
        
        
        
        kl_loss = kd_loss(student_logits_kl,teacher_logits_kl,self.distill_dict['temperature'])  #
   
        sup_loss = kd_loss(student_logits_sup,teacher_logits_sup,self.distill_dict['temperature']) 
        
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss = F.cross_entropy(student_logits_ce,test_labels)/16


        # loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        loss=kl_loss+0.5*sup_loss+ce_sup_loss
        
        
        return {'soft_loss': kl_loss, 'hard_loss': 0.5*sup_loss+ce_sup_loss,'loss': loss}
    
    def fc_2_sup_dist_cece(self,student_logits,teacher_logits,test_labels):
        
        student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_ce = student_logits['ce'].to(self.device)
        
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        
        # teacher_logits_sup = teacher_logits_sup.fill_diagonal_(-100000)
        
        
        # student_logits_kl_nograd = student_logits_kl.detach()
        # student_logits_sup_nograd = student_logits_sup.detach()
        # student_logits_ce_nograd = student_logits_ce.detach()
        # teacher_logits_kl_nograd = teacher_logits_kl.detach()
        # teacher_logits_sup_nograd = teacher_logits_sup.detach()


        # softmax_loss_t = F.cross_entropy(student_logits_kl_nograd, test_labels) #
        # softmax_loss_s_ce = F.cross_entropy(student_logits_ce_nograd, test_labels) #
        

        # focal_weight =  softmax_loss_s_ce/ ( softmax_loss_t+ 1e-8)  
        # ratio_lower = torch.zeros(1).cuda()
        # focal_weight = torch.max(focal_weight, ratio_lower)
        # focal_weight = 1 - torch.exp(- focal_weight)
        
        
        
        kl_loss = kd_loss(student_logits_kl,teacher_logits_kl,self.distill_dict['temperature'])  #
   
        sup_loss = inter_class_relation(student_logits_sup,teacher_logits_sup) 
        
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss = F.cross_entropy(student_logits_ce,test_labels)/16
        klce_loss = F.cross_entropy(student_logits_kl,test_labels)/16


        # loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        loss=kl_loss+klce_loss+0.5*sup_loss+ce_sup_loss
        
        return {'soft_loss': kl_loss, 'hard_loss': 0.5*sup_loss+ce_sup_loss,'loss': loss}
    
    def fc_2_sup_klklcece(self,student_logits,teacher_logits,test_labels):
        
        student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_ce = student_logits['ce'].to(self.device)
        
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        
        # teacher_logits_sup = teacher_logits_sup.fill_diagonal_(-100000)
        
        
        # student_logits_kl_nograd = student_logits_kl.detach()
        # student_logits_sup_nograd = student_logits_sup.detach()
        # student_logits_ce_nograd = student_logits_ce.detach()
        # teacher_logits_kl_nograd = teacher_logits_kl.detach()
        # teacher_logits_sup_nograd = teacher_logits_sup.detach()


        # softmax_loss_t = F.cross_entropy(student_logits_kl_nograd, test_labels) #
        # softmax_loss_s_ce = F.cross_entropy(student_logits_ce_nograd, test_labels) #
        

        # focal_weight =  softmax_loss_s_ce/ ( softmax_loss_t+ 1e-8)  
        # ratio_lower = torch.zeros(1).cuda()
        # focal_weight = torch.max(focal_weight, ratio_lower)
        # focal_weight = 1 - torch.exp(- focal_weight)
        
        
        
        kl_loss = kd_loss(student_logits_kl,teacher_logits_kl,self.distill_dict['temperature'])  #
   
        sup_loss = kd_loss(student_logits_sup,teacher_logits_sup,self.distill_dict['temperature']) 
        
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss = F.cross_entropy(student_logits_ce,test_labels)/16
        klce_loss = F.cross_entropy(student_logits_kl,test_labels)/16


        # loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        loss=kl_loss+klce_loss+0.5*sup_loss+ce_sup_loss
        
        return {'soft_loss': kl_loss, 'hard_loss': 0.5*sup_loss+ce_sup_loss,'loss': loss}
    
    def fc_2_sup_distdistcece(self,student_logits,teacher_logits,test_labels):
        
        student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_ce = student_logits['ce'].to(self.device)
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        

        kl_loss = inter_class_relation(student_logits_kl,teacher_logits_kl)  
        sup_loss = inter_class_relation(student_logits_sup,teacher_logits_sup) 
        
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss = F.cross_entropy(student_logits_ce,test_labels)/16
        klce_loss = F.cross_entropy(student_logits_kl,test_labels)/16


        # loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        loss=kl_loss+klce_loss+0.5*sup_loss+ce_sup_loss
        
        return {'soft_loss': kl_loss, 'hard_loss': 0.5*sup_loss+ce_sup_loss,'loss': loss}
    
    def fc_2_sup_2(self,student_logits,teacher_logits,test_labels):
        
        student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup_kl = student_logits['sup_kl'].to(self.device)   #学生sup分支分数
        
        student_logits_ce = student_logits['ce'].to(self.device)
        student_logits_sup_ce = student_logits['sup_ce'].to(self.device)
        
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        
        # teacher_logits_sup = teacher_logits_sup.fill_diagonal_(-100000)
        
        
        # student_logits_kl_nograd = student_logits_kl.detach()
        # student_logits_sup_nograd = student_logits_sup.detach()
        # student_logits_ce_nograd = student_logits_ce.detach()
        # teacher_logits_kl_nograd = teacher_logits_kl.detach()
        # teacher_logits_sup_nograd = teacher_logits_sup.detach()


        # softmax_loss_t = F.cross_entropy(student_logits_kl_nograd, test_labels) #
        # softmax_loss_s_ce = F.cross_entropy(student_logits_ce_nograd, test_labels) #
        

        # focal_weight =  softmax_loss_s_ce/ ( softmax_loss_t+ 1e-8)  
        # ratio_lower = torch.zeros(1).cuda()
        # focal_weight = torch.max(focal_weight, ratio_lower)
        # focal_weight = 1 - torch.exp(- focal_weight)
        
        
        
        kl_loss = kd_loss(student_logits_kl,teacher_logits_kl,self.distill_dict['temperature'])  #
   
        sup_ce_loss = inter_class_relation(student_logits_sup_ce,teacher_logits_sup) 
        sup_kl_loss = inter_class_relation(student_logits_sup_kl,teacher_logits_sup) 
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_loss = F.cross_entropy(student_logits_ce,test_labels)/16


        # loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        loss=(kl_loss+sup_kl_loss)+ce_loss+sup_ce_loss
        
        return {'soft_loss': kl_loss+0.5*sup_kl_loss, 'hard_loss': ce_loss+0.5*sup_ce_loss,'loss': loss}


    def fc_2_sup_disver(self,student_logits,teacher_logits,test_labels):
        
        student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_ce = student_logits['ce'].to(self.device)
        
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        
        
        kl_loss_sup = kd_loss(student_logits_sup,teacher_logits_sup,self.distill_dict['temperature'])  #
        sup_loss_query = inter_class_relation(student_logits_kl,teacher_logits_kl) 
        
        ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss = F.cross_entropy(student_logits_ce,test_labels)/16


        # loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        loss=0.5*kl_loss_sup+sup_loss_query+ce_sup_loss+ce_kl_loss
        
        
        return {'soft_loss': kl_loss_sup, 'hard_loss': sup_loss_query+ce_sup_loss,'loss': loss}

    def fc_2_sup_dist_wsl(self,student_logits,teacher_logits,test_labels):
        
        student_logits_kl = student_logits['kl'].to(self.device)  #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_ce = student_logits['ce'].to(self.device)
        
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        
        # teacher_logits_sup = teacher_logits_sup.fill_diagonal_(-100000)
        
        
        student_logits_kl_nograd = student_logits_kl.detach()
        student_logits_sup_nograd = student_logits_sup.detach()
        student_logits_ce_nograd = student_logits_ce.detach()
        teacher_logits_kl_nograd = teacher_logits_kl.detach()
        teacher_logits_sup_nograd = teacher_logits_sup.detach()


        softmax_loss_t = F.cross_entropy(student_logits_kl_nograd, test_labels) #
        softmax_loss_s_ce = F.cross_entropy(student_logits_ce_nograd, test_labels) #
        

        focal_weight =  softmax_loss_s_ce/ ( softmax_loss_t+ 1e-8)  
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        
        kl_loss = kd_loss(student_logits_kl,teacher_logits_kl,self.distill_dict['temperature'])  #
        sup_loss = inter_class_relation(student_logits_sup,teacher_logits_sup) 
        
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss = F.cross_entropy(student_logits_ce,test_labels)/16

        loss=(0.5+focal_weight)*kl_loss+(0.5+1-focal_weight)*(0.5*sup_loss+ce_sup_loss)
        
        # loss=kl_loss+0.5*sup_loss+ce_sup_loss
        
        # print('sup_loss')
        # print(sup_loss)
        
        # print('kl_loss')
        # print(kl_loss)
        
        # print('ce_sup_loss')
        # print(ce_sup_loss)
        
        
        
        return {'soft_loss': kl_loss, 'hard_loss': 0.5*sup_loss+ce_sup_loss,'loss': loss}
    
    def strm_fc_2_sup_dist(self,student_logits,teacher_logits,test_labels):
        
        student_logits_pat = student_logits['pat'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_fr1 = student_logits['fr1'].to(self.device)
        student_logits_fr2 = student_logits['fr2'].to(self.device)
        
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        
        kl_loss_fr = kd_loss(student_logits_fr1,teacher_logits_kl,self.distill_dict['temperature'])  #
        kl_loss_pat = kd_loss(student_logits_pat,teacher_logits_kl,self.distill_dict['temperature'])  #
   
        sup_loss = inter_class_relation(student_logits_sup,teacher_logits_sup) 
        
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss_fr = F.cross_entropy(student_logits_fr2,test_labels)/16
        ce_sup_loss_pat = F.cross_entropy(student_logits_pat,test_labels)/16



        # loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        loss=kl_loss_fr+0.5*sup_loss+ce_sup_loss_fr+0.1*(kl_loss_pat+ce_sup_loss_pat)
        
        
        return {'loss': loss}
    
    def strm_1fc_sup(self,student_logits,teacher_logits,test_labels):
        student_logits_pat = student_logits['pat'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_fr = student_logits['fr'].to(self.device)

        
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
        
        kl_loss_fr = kd_loss(student_logits_fr,teacher_logits_kl,self.distill_dict['temperature'])  #
        kl_loss_pat = kd_loss(student_logits_pat,teacher_logits_kl,self.distill_dict['temperature'])  #
   
        sup_loss = inter_class_relation(student_logits_sup,teacher_logits_sup) 
        
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss_fr = F.cross_entropy(student_logits_fr,test_labels)/16
        ce_sup_loss_pat = F.cross_entropy(student_logits_pat,test_labels)/16



        # loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        loss=kl_loss_fr+0.5*sup_loss+ce_sup_loss_fr+0.1*(kl_loss_pat+ce_sup_loss_pat)
        
        
        return {'loss': loss}
    
    def fc_1_sup(self,student_logits,teacher_logits,test_labels):
        
        # student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_kl = student_logits['kl'].to(self.device)
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数

        kl_loss = kd_loss(student_logits_kl,teacher_logits_kl,self.distill_dict['temperature'])  #
        ce_loss=F.cross_entropy(student_logits_kl, test_labels,) /16#20,5
        sup_loss=0.5*inter_class_relation(student_logits_sup,teacher_logits_sup)

        return { 'loss': ce_loss + kl_loss+sup_loss}
    
    def fc_sup(self,student_logits,teacher_logits,test_labels):
        
        # student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        student_logits_kl = student_logits['kl'].to(self.device)
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数

        # kl_loss = kd_loss(student_logits_kl,teacher_logits_kl,self.distill_dict['temperature'])  #
        ce_loss=F.cross_entropy(student_logits_kl, test_labels,) /16#20,5
        sup_loss=0.5*inter_class_relation(student_logits_sup,teacher_logits_sup)

        return { 'loss': ce_loss + sup_loss}
    
    def e_dist_1fc_sup(self,student_logits,teacher_logits,test_labels):
        
        student_logits_kl = student_logits['kl'].to(self.device) #学生kl分支分数
        student_logits_sup = student_logits['sup'].to(self.device)   #学生sup分支分数
        # student_logits_ce = student_logits['ce'].to(self.device)
        
        teacher_logits_kl = teacher_logits['kl'].to(self.device)  #学生kl分支分数
        teacher_logits_sup = teacher_logits['sup'].to(self.device)  #学生sup分支分数
       
        kl_loss = kd_loss(student_logits_kl,teacher_logits_kl,self.distill_dict['temperature'])  #
   
        sup_loss = inter_class_relation(student_logits_sup,teacher_logits_sup) 
        
        # ce_kl_loss = F.cross_entropy(student_logits_kl,test_labels)/16
        ce_sup_loss = F.cross_entropy(student_logits_kl,test_labels)/16

        # loss=(1+focal_weight)*kl_loss+(1+1-focal_weight)*(0.1*sup_loss+ce_sup_loss)
        
        loss=kl_loss+0.5*sup_loss+ce_sup_loss\
        
        return {'loss': loss}





