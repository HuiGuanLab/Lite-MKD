import torch
import torch.nn as nn
import math
import numpy as np
import os
import torch.nn.functional as F  

NUM_SAMPLES=1

np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)



class SupportDK(nn.Module):
    
    def __init__(self, args):
        super(SupportDK, self).__init__()
        self.args = args
        
    def diff_norm(self,x,y):
        diff=x-y
        norm_sq = torch.norm(diff, dim=[-2,-1])**2
        distance = torch.div(norm_sq, self.args.seq_len) # 25
        distance=distance* -1
        return distance
        
    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        support_set = support_set.reshape(self.args.way,self.args.shot,self.args.seq_len,2048) #5*5*8*2048
        support_prototype = support_set.mean(dim=1) #5*8*2048
        
        new_dis = torch.zeros(5,4)
        for index,i in enumerate(support_prototype):
            m = 0
            for n in range(5):
                if index != n:
                    new_dis[index,m] = self.diff_norm(i,support_prototype[n])
                    m=m+1
 
        return_dict = {'logits': new_dis}
        return return_dict
        
class e_dist(nn.Module):
   #余弦相似度的计算
    def __init__(self, args):
        super(e_dist, self).__init__()
        self.args = args

    def forward(self, support_set, support_labels, queries):  
        # support_set : 200*2048, support_labels: 25, queries: 160*2048
        supports = support_set.reshape(-1,8,2048) #25*8*2048
        queries = queries.reshape(-1,8,2048) #20*8*2048
        
        n_queries = queries.shape[0] #4
        queries=queries.mean(dim=1)  #前两个维度聚集  4*2048
        
        
        support_labels = support_labels
        unique_labels = torch.unique(support_labels) # 5  

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        dist_all = torch.zeros(n_queries, self.args.way) # 5 x 5

        for label_idx, c in enumerate(unique_labels):
            # Select keys corresponding to this class from the support set tuples
            class_k = torch.index_select(supports, 0, self._extract_class_indices(support_labels, c)) # 5-8-2048    

            support_set_c=class_k.mean(dim=1) #5-2048

            # Calculate p-norm distance between the query embedding and the support set embedding
            distmat = torch.cdist(queries, support_set_c,p=2) # 5 ×  5

            # Average across the 28 tuples
            query_dist = distmat.mean(dim=1)  # 5

            # Make it negative as this has to be reduced.
            distance = -1.0 * query_dist  #距离变负  5
            c_idx = c.long()
            dist_all[:,c_idx] = distance # Insert into the required location.

        logits = {
            'logits':dist_all
            } 
        
        return logits
    
    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class e_dist_fc2(nn.Module):
    '''
    TRX分类器
    输入S和Q的特征  25，8，2048  20，8，2048
    输出分数   20，5
    '''

    def __init__(self, args, ):
        super(e_dist_fc2, self).__init__()
        self.train()
        self.args = args       
        self.e_dict = e_dist(args)
        
                           
    def forward(self,context_feature, context_labels, target_feature):
        
        context_feature_1 = context_feature['context_features_1']
        context_feature_2 = context_feature['context_features_2']
        target_feature_1 = target_feature['target_features_1']
        target_feature_2 = target_feature['target_features_2']

        all_logits_1=self.e_dict(context_feature_1, context_labels, target_feature_1)['logits']
        all_logits_2=self.e_dict(context_feature_2, context_labels, target_feature_2)['logits']

        
        all_logit={'fc_1': all_logits_1, 'fc_2': all_logits_2}
        
        all_logits={
            'logits':all_logit
        }
        return all_logits


class e_dist_fc2_sup(nn.Module):
    '''
    TRX分类器
    输入S和Q的特征  25，8，2048  20，8，2048
    输出分数   20，5
    '''

    def __init__(self, args, ):
        super(e_dist_fc2_sup, self).__init__()
        self.train()
        self.args = args       
        self.e_dict = e_dist(args)
        self.supportKD=SupportDK(args)
        
                           
    def forward(self,context_feature, context_labels, target_feature):

        
        context_feature_1 = context_feature['context_features_1']
        context_feature_2 = context_feature['context_features_2']
        target_feature_1 = target_feature['target_features_1']
        target_feature_2 = target_feature['target_features_2']

        all_logits_1=self.e_dict(context_feature_1, context_labels, target_feature_1)['logits']
        all_logits_2=self.e_dict(context_feature_2, context_labels, target_feature_2)['logits']
        all_logits_3=self.supportKD(context_feature_2,context_labels,target_feature_2)['logits']

        
        all_logit={'kl': all_logits_1, 'ce': all_logits_2 ,'sup':all_logits_3}
        
        all_logits={
            'logits':all_logit
        }
        return all_logits
    

class e_dist_1fc_sup(nn.Module):
    '''
    TRX分类器
    输入S和Q的特征  25，8，2048  20，8，2048
    输出分数   20，5
    '''

    def __init__(self, args, ):
        super(e_dist_1fc_sup, self).__init__()
        self.train()
        self.args = args       
        self.e_dict = e_dist(args)
        self.supportKD=SupportDK(args)
        
                           
    def forward(self,context_feature, context_labels, target_feature):

        all_logits_2=self.e_dict(context_feature, context_labels, target_feature)['logits']
        all_logits_3=self.supportKD(context_feature,context_labels,target_feature)['logits']

        all_logit={'kl': all_logits_2, 'sup':all_logits_3}
        
        all_logits={
            'logits':all_logit
        }
        return all_logits
   
    
class e_dist_fc2_sup_fixed(nn.Module):
    '''
    TRX分类器
    输入S和Q的特征  25，8，2048  20，8，2048
    输出分数   20，5
    '''

    def __init__(self, args, ):
        super(e_dist_fc2_sup_fixed, self).__init__()
        self.train()
        self.args = args       
        self.e_dict = e_dist(args)
        self.supportKD=SupportDK(args)
        
                           
    def forward(self,context_feature, context_labels, target_feature):



        all_logits_1=self.e_dict(context_feature, context_labels, target_feature)['logits']
        all_logits_2=self.supportKD(context_feature,context_labels,target_feature)['logits']

        
        all_logit={'kl': all_logits_1,'sup':all_logits_2}
        
        all_logits={
            'logits':all_logit
        }
        return all_logits
    