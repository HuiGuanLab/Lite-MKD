from multiprocessing import context
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import split_first_dim_linear
import math
import numpy as np
from itertools import combinations 

from torch.autograd import Variable

import torchvision.models as models
import os


NUM_SAMPLES=1

np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)


class PositionalEncoding(nn.Module):   #PE
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        # pe is of shape max_len(5000) x 2048(last layer of FC)
        pe = torch.zeros(max_len, d_model)
        # print(pe.shape)
        # position is of shape 5000 x 1
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape 1 x 5000 x 2048
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       y=Variable(self.pe[:, :x.size(1)], requires_grad=False)
    #    print(x.shape)
    #    print(y.shape)
       x = x + y
       return self.dropout(x)

class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=2):
        super(TemporalCrossTransformer, self).__init__()
        self.args = args
        self.temporal_set_size = temporal_set_size  #固定2

        max_len = int(self.args.seq_len * 1.5)
        
        self.pe = PositionalEncoding(2048, self.args.trans_dropout, max_len=max_len)  #固定2048，因为是融合特征，在backbone当会升维到2048

        self.k_linear = nn.Linear(2048 * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(2048 * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb).cuda() for comb in frame_combinations]
        self.tuples_len = len(self.tuples) #28
    
    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0] #20
        n_support = support_set.shape[0] #25
        
        # static pe after adding the position embedding
        support_set = self.pe(support_set)# Support set is of shape 25 x 8 x 2048 -> 25 x 8 x 2048
        queries = self.pe(queries) # Queries is of shape 20 x 8 x 2048 -> 20 x 8 x 2048

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked))
        # print(len(self.tuples))
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]

        support_set = torch.stack(s, dim=-2) # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2) # 20 x 28 x 4096

        # apply linear maps for performing self-normalization in the next step and the key map's output
        '''
            support_set_ks is of shape 25 x 28 x 1152, where 1152 is the dimension of the key = query head. converting the 5-way*5-shot x 28(tuples).
            query_set_ks is of shape 20 x 28 x 1152 covering 4 query/sample*5-way x 28(number of tuples)
        '''
        support_set_ks = self.k_linear(support_set) # 25 x 28 x 1152
        queries_ks = self.k_linear(queries) # 20 x 28 x 1152
        support_set_vs = self.v_linear(support_set) # 25 x 28 x 1152
        queries_vs = self.v_linear(queries) # 20 x 28 x 1152
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks) # 25 x 28 x 1152
        mh_queries_ks = self.norm_k(queries_ks) # 20 x 28 x 1152
        support_labels = support_labels
        mh_support_set_vs = support_set_vs # 25 x 28 x 1152
        mh_queries_vs = queries_vs # 20 x 28 x 1152       
        unique_labels = torch.unique(support_labels) # 5

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        all_distances_tensor_ori = torch.zeros(n_queries, self.args.way) # 20 x 5
        all_distances_tensor = torch.zeros(n_queries,self.tuples_len,self.args.trans_linear_out_dim, self.args.way) # 20 x 5
        cos=nn.CosineSimilarity(dim=1)
        # print(all_distances_tensor.shape)
        for label_idx, c in enumerate(unique_labels):
            # select keys and values for just this class 
            # print('mh_support_set_ks')
            # print(mh_support_set_ks.shape)
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c)) # 5 x 28 x 1152
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c)) # 5 x 28 x 1152
            k_bs = class_k.shape[0] # 5
            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim) # 20 x 5 x 28 x 28
            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3) # 20 x 28 x 5 x 28 
            
            # [For the 20 queries' 28 tuple pairs, find the best match against the 5 selected support samples from the same class
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1) # 20 x 28 x 140
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)] # list(20) x 28 x 140
            class_scores = torch.cat(class_scores) # 560 x 140 - concatenate all the scores for the tuples
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len) # 20 x 28 x 5 x 28
            class_scores = class_scores.permute(0,2,1,3) # 20 x 5 x 28 x 28
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v) # 20 x 5 x 28 x 1152 
            query_prototype = torch.sum(query_prototype, dim=1) # 20 x 28 x 1152 -> Sum across all the support set values of the corres. class
            
            
            #现在
            # print("query_prototype:")
            # print(query_prototype.shape)
            
            c_idx = c.long()
            all_distances_tensor[:,:,:,c_idx] = query_prototype # 20*28*1152*5
            
            
            #原来的分数
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype # 20 x 28 x 1152
            norm_sq = torch.norm(diff, dim=[-2,-1])**2 # 20 
            distance = torch.div(norm_sq, self.tuples_len) # 20
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor_ori[:,c_idx] = distance # 20
            
            
        #现在
        all_distances_tensor=all_distances_tensor.reshape(n_queries,-1,self.args.way)  #20*(56*1152)*5
        
        sim=cos(all_distances_tensor.unsqueeze(2),all_distances_tensor.unsqueeze(3))
        # sim=torch.matmul(all_distances_tensor.T,all_distances_tensor) *-1
        # print("sim")
        # print(sim.shape)
        # print('ori')
        # print(all_distances_tensor_ori)
        
        return_dict={
            'logits':{
                
            'support_set':sim,
            'query': all_distances_tensor_ori
            }
        }
        
        return return_dict

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


class TRX_sup(nn.Module):
    '''
    TRX分类器
    输入S和Q的特征  25，8，2048  20，8，2048
    输出分数   20，5
    '''

    def __init__(self, args, ):
        super(TRX_sup, self).__init__()
        self.train()
        self.args = args       
        self.transformers = TemporalCrossTransformer(args, 2)
                           
    def forward(self,context_feature, context_labels, target_feature):
            all_logits = self.transformers(context_feature, context_labels, target_feature)
            return all_logits
    
    
class TRX_sup_fixed(nn.Module):
    '''
    TRX分类器
    输入S和Q的特征  25，8，2048  20，8，2048
    输出分数   20，5
    '''

    def __init__(self, args, ):
        super(TRX_sup_fixed, self).__init__()
        self.train()
        self.args = args       
        self.transformers = TemporalCrossTransformer(args, 2)
        
                           
    def forward(self,context_feature, context_labels, target_feature):
        with torch.no_grad():
            all_logits = self.transformers(context_feature, context_labels, target_feature)
        return all_logits
    