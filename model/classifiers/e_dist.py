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

