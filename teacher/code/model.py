import math
from collections import OrderedDict
from itertools import combinations
from turtle import forward

import timm
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from matplotlib.style import context
from torch.autograd import Variable
from transformer import BertAttention
from torch.cuda.amp import autocast as autocast
from utils import My_Loss, split_first_dim_linear
from einops import rearrange
import torch.nn.functional as F

NUM_SAMPLES = 1

np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        # pe is of shape max_len(5000) x 2048(last layer of FC)
        pe = torch.zeros(max_len, d_model)
        # position is of shape 5000 x 1
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape 1 x 5000 x 2048
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class CosDistance(nn.Module):
   #余弦相似度的计算
    def __init__(self, args):
        super(CosDistance, self).__init__()
        self.args = args

    def forward(self, support_set, support_labels, queries):  
        # support_set : 25 x 8 x 4096, support_labels: 25, queries: 5 x 8 x 4096
        n_queries = queries.shape[0] #5
        queries = queries.mean(dim=1)  #前两个维度聚集  5×4096
        
        support_set = support_set.cuda(0)
        support_labels = support_labels.cuda(0)
        queries = queries.cuda(0)

        
        support_labels = support_labels
        unique_labels = torch.unique(support_labels) # 5  

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        dist_all = torch.zeros(n_queries, self.args.way) # 5 x 5

        for label_idx, c in enumerate(unique_labels):
            # Select keys corresponding to this class from the support set tuples
            class_k = torch.index_select(support_set, 0, self._extract_class_indices(support_labels, c)) # 25-8-4096    5 x 8 x 4096

            # Reshaping the selected keys
            # support_set_c = class_k.view(-1, 4096) # 40 x 4096

            support_set_c=class_k.mean(dim=1)

            # Calculate p-norm distance between the query embedding and the support set embedding
            distmat = torch.cdist(queries, support_set_c ) # 5 ×  5

            # Average across the 28 tuples
            query_dist = distmat.mean(dim=1)  # 5

            # Make it negative as this has to be reduced.
            distance = -1.0 * query_dist  #距离变负  5
            c_idx = c.long()
            dist_all[:,c_idx] = distance # Insert into the required location.

        return_dict =  dist_all
        
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


class DistanceLoss(nn.Module):
    "Compute the Query-class similarity on the patch-enriched features."

    def __init__(self, args, temporal_set_size=3):
        super(DistanceLoss, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.dropout = nn.Dropout(p=0.1)

        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb) for comb in frame_combinations]
        self.tuples_len = len(self.tuples)  # 28 for tempset_2

        # nn.Linear(4096, 1024)
        self.clsW = nn.Linear(self.args.trans_linear_in_dim *
                              self.temporal_set_size, self.args.trans_linear_in_dim//2)
        self.relu = torch.nn.ReLU()

    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0]  # 20
        n_support = support_set.shape[0]  # 25

        # Add a dropout before creating tuples
        support_set = self.dropout(support_set)  # 25 x 8 x 2048
        queries = self.dropout(queries)  # 20 x 8 x 2048

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked))
        self.tuples = [p.to(support_set.device) for p in self.tuples]
        s = [torch.index_select(
            support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1)
             for p in self.tuples]

        support_set = torch.stack(s, dim=-2)  # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2)  # 20 x 28 x 4096
        support_labels = support_labels.to(support_set.device)
        unique_labels = torch.unique(support_labels).to(support_set.device)

        query_embed = self.clsW(
            queries.view(-1, self.args.trans_linear_in_dim*self.temporal_set_size))  # 560[20x28] x 1024

        # Add relu after clsW
        query_embed = self.relu(query_embed)  # 560 x 1024

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        dist_all = torch.zeros(n_queries, self.args.way)  # 20 x 5

        for label_idx, c in enumerate(unique_labels):
            # Select keys corresponding to this class from the support set tuples
            class_k = torch.index_select(support_set, 0, self._extract_class_indices(
                support_labels, c.clone()))  # 5 x 28 x 4096

            # Reshaping the selected keys
            # 140 x 4096
            class_k = class_k.view(-1, self.args.trans_linear_in_dim *
                                   self.temporal_set_size)

            # Get the support set projection from the current class
            support_embed = self.clsW(class_k.to(
                queries.device))  # 140[5 x 28] x1024

            # Add relu after clsW
            support_embed = self.relu(support_embed)  # 140 x 1024

            # Calculate p-norm distance between the query embedding and the support set embedding
            # 560[20 x 28] x 140[28 x 5]
            distmat = torch.cdist(query_embed, support_embed)

            # Across the 140 tuples compared against, get the minimum distance for each of the 560 queries
            min_dist = distmat.min(dim=1)[0].reshape(
                n_queries, self.tuples_len)  # 20[5-way x 4-queries] x 28

            # Average across the 28 tuples
            query_dist = min_dist.mean(dim=1)  # 20

            # Make it negative as this has to be reduced.
            distance = -1.0 * query_dist
            c_idx = c.long()
            dist_all[:, c_idx] = distance  # Insert into the required location.

        return_dict = {'logits': dist_all}

        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        print
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(
            labels, which_class)  # binary mask of labels equal to which_class
        # indices of labels equal to which class
        class_mask_indices = torch.nonzero(class_mask)
        # reshape to be a 1D vector
        return torch.reshape(class_mask_indices, (-1,))


class TemporalCrossTransformer(nn.Module):
    def __init__(self, args, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()

        self.args = args
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(
            self.args.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.args.trans_linear_in_dim *
                                  temporal_set_size, self.args.trans_linear_out_dim)  # .cuda()
        self.v_linear = nn.Linear(self.args.trans_linear_in_dim *
                                  temporal_set_size, self.args.trans_linear_out_dim)  # .cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)

        self.class_softmax = torch.nn.Softmax(dim=1)

        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = [torch.tensor(comb) for comb in frame_combinations]
        self.tuples_len = len(self.tuples)  # 28

    def forward(self, support_set, support_labels, queries):
        support_labels = support_labels.to(support_set.device)
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0]  # 20
        n_support = support_set.shape[0]  # 25

        # static pe after adding the position embedding
        # Support set is of shape 25 x 8 x 2048 -> 25 x 8 x 2048
        support_set = self.pe(support_set)
        # Queries is of shape 20 x 8 x 2048 -> 20 x 8 x 2048
        queries = self.pe(queries)

        # construct new queries and support set made of tuples of images after pe
        # Support set s = number of tuples(28 for 2/56 for 3) stacked in a list form containing elements of form 25 x 4096(2 x 2048 - (2 frames stacked))
        self.tuples = [p.to(support_set.device) for p in self.tuples]
        s = [torch.index_select(
            support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1)
             for p in self.tuples]

        support_set = torch.stack(s, dim=-2)  # 25 x 28 x 4096
        queries = torch.stack(q, dim=-2)  # 20 x 28 x 4096

        # apply linear maps for performing self-normalization in the next step and the key map's output
        '''
            support_set_ks is of shape 25 x 28 x 1152, where 1152 is the dimension of the key = query head. converting the 5-way*5-shot x 28(tuples).
            query_set_ks is of shape 20 x 28 x 1152 covering 4 query/sample*5-way x 28(number of tuples)
        '''
        support_set_ks = self.k_linear(support_set)  # 25 x 28 x 1152
        queries_ks = self.k_linear(queries)  # 20 x 28 x 1152
        support_set_vs = self.v_linear(support_set)  # 25 x 28 x 1152
        queries_vs = self.v_linear(queries)  # 20 x 28 x 1152

        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)  # 25 x 28 x 1152
        mh_queries_ks = self.norm_k(queries_ks)  # 20 x 28 x 1152
        mh_support_set_vs = support_set_vs  # 25 x 28 x 1152
        mh_queries_vs = queries_vs  # 20 x 28 x 1152

        unique_labels = torch.unique(support_labels)  # 5
        unique_labels.to(support_set.device)
        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        all_distances_tensor = torch.zeros(n_queries, self.args.way)  # 20 x 5

        for label_idx, c in enumerate(unique_labels):

            # select keys and values for just this class
            # mh_support_set_ks = mh_support_set_ks.to(support_labels.device)
            # mh_support_set_vs = mh_support_set_vs.to(support_labels.device)
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(
                support_labels, c))  # 5 x 28 x 1152
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(
                support_labels, c))  # 5 x 28 x 1152
            k_bs = class_k.shape[0]  # 5

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(
                -2, -1)) / math.sqrt(self.args.trans_linear_out_dim)  # 20 x 5 x 28 x 28

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0, 2, 1, 3)  # 20 x 28 x 5 x 28

            # [For the 20 queries' 28 tuple pairs, find the best match against the 5 selected support samples from the same class
            class_scores = class_scores.reshape(
                n_queries, self.tuples_len, -1)  # 20 x 28 x 140
            class_scores = [self.class_softmax(class_scores[i]) for i in range(
                n_queries)]  # list(20) x 28 x 140
            # 560 x 140 - concatenate all the scores for the tuples
            class_scores = torch.cat(class_scores)
            class_scores = class_scores.reshape(
                n_queries, self.tuples_len, -1, self.tuples_len)  # 20 x 28 x 5 x 28
            class_scores = class_scores.permute(0, 2, 1, 3)  # 20 x 5 x 28 x 28

            # get query specific class prototype
            query_prototype = torch.matmul(
                class_scores, class_v)  # 20 x 5 x 28 x 1152
            # 20 x 28 x 1152 -> Sum across all the support set values of the corres. class
            query_prototype = torch.sum(query_prototype, dim=1)

            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype  # 20 x 28 x 1152
            norm_sq = torch.norm(diff, dim=[-2, -1])**2  # 20
            distance = torch.div(norm_sq, self.tuples_len)  # 20

            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:, c_idx] = distance  # 20

        return_dict = {'logits': all_distances_tensor}

        return return_dict

    @staticmethod
    def _extract_class_indices(labels, which_class):
        """
        Helper method to extract the indices of elements which have the specified label.
        :param labels: (torch.tensor) Labels of the context set.
        :param which_class: Label for which indices are extracted.
        :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
        """
        class_mask = torch.eq(
            labels, which_class)  # binary mask of labels equal to which_class
        # indices of labels equal to which class
        class_mask_indices = torch.nonzero(class_mask)
        # reshape to be a 1D vector
        return torch.reshape(class_mask_indices, (-1,))


class Token_Perceptron(torch.nn.Module):
    '''
        2-layer Token MLP
    '''

    def __init__(self, in_dim):
        super(Token_Perceptron, self).__init__()
        # in_dim 8
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        # Applying the linear layer on the input
        output = self.inp_fc(x)  # B x 2048 x 8

        # Apply the relu non-linearity
        output = self.relu(output)  # B x 2048 x 8

        # Apply the 2nd linear layer
        output = self.out_fc(output)

        return output


class Bottleneck_Perceptron_2_layer(torch.nn.Module):
    '''
        2-layer Bottleneck MLP
    '''

    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_2_layer, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.out_fc(output)

        return output


class Bottleneck_Perceptron_3_layer_res(torch.nn.Module):
    '''
        3-layer Bottleneck MLP followed by a residual layer
    '''

    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_3_layer_res, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim//2)
        self.hid_fc = nn.Linear(in_dim//2, in_dim//2)
        self.out_fc = nn.Linear(in_dim//2, in_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.relu(self.hid_fc(output))
        output = self.out_fc(output)

        return output + x  # Residual output


class Self_Attn_Bot(nn.Module):
    """ Self attention Layer
        Attention-based frame enrichment
    """

    def __init__(self, in_dim, seq_len):
        super(Self_Attn_Bot, self).__init__()
        self.chanel_in = in_dim  # 2048

        # Using Linear projections for Key, Query and Value vectors
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.Bot_MLP = Bottleneck_Perceptron_3_layer_res(in_dim)
        max_len = int(seq_len * 1.5)
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W )[B x 16 x 2048]
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width)
        """

        # Add a position embedding to the 16 patches
        x = self.pe(x)  # B x 16 x 2048

        m_batchsize, C, width = x.size()  # m = 200/160, C = 2048, width = 16

        # Save residual for later use
        residual = x  # B x 16 x 2048

        # Perform query projection
        proj_query = self.query_proj(x)  # B x 16 x 2048

        # Perform Key projection
        proj_key = self.key_proj(x).permute(0, 2, 1)  # B x 2048  x 16

        energy = torch.bmm(proj_query, proj_key)  # transpose check B x 16 x 16
        attention = self.softmax(energy)  # B x 16 x 16

        # Get the entire value in 2048 dimension
        proj_value = self.value_conv(x).permute(0, 2, 1)  # B x 2048 x 16

        # Element-wise multiplication of projected value and attention: shape is x B x C x N: 1 x 2048 x 8
        out = torch.bmm(proj_value, attention.permute(
            0, 2, 1))  # B x 2048 x 16

        # Reshaping before passing through MLP
        out = out.permute(0, 2, 1)  # B x 16 x 2048

        # Passing via gamma attention
        out = self.gamma*out + residual  # B x 16 x 2048

        # Pass it via a 3-layer Bottleneck MLP with Residual Layer defined within MLP
        out = self.Bot_MLP(out)  # B x 16 x 2048

        return out


class MLP_Mix_Enrich(nn.Module):
    """ 
        Pure Token-Bottleneck MLP-based enriching features mechanism
    """

    def __init__(self, in_dim, seq_len):
        super(MLP_Mix_Enrich, self).__init__()
        # in_dim = 2048
        self.Tok_MLP = Token_Perceptron(seq_len)  # seq_len = 8 frames
        self.Bot_MLP = Bottleneck_Perceptron_2_layer(in_dim)

        max_len = int(seq_len * 1.5)  # seq_len = 8
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W ) # B(25/20) x 8 x 2048
            returns :
                out : self MLP-enriched value + input feature 
        """

        # Add a position embedding to the 8 frames
        x = self.pe(x)  # B x 8 x 2048

        # Store the residual for use later
        residual1 = x  # B x 8 x 2048

        # Pass it via a 2-layer Token MLP followed by Residual Layer
        # Permuted before passing into the MLP: B x 2048 x 8
        out = self.Tok_MLP(x.permute(0, 2, 1)).permute(
            0, 2, 1) + residual1  # B x 8 x 2048

        # Storing a residual
        residual2 = out  # B x 8 x 2048

        # Pass it via 2-layer Bottleneck MLP defined on Channel(2048) features
        out = self.Bot_MLP(out) + residual2  # B x 8 x 2048

        return out


class TRX(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, args):
        super(TRX, self).__init__()

        self.train()
        self.args = args

        # Using ResNet Backbone
        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.num_patches = 16

        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))

        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList(
            [TemporalCrossTransformer(args, s) for s in args.temp_set])

        # New-distance metric for post patch-level enriched features
        self.new_dist_loss_post_pat = [
            DistanceLoss(args, s) for s in args.temp_set]

        # Linear-based patch-level attention over the 16 patches
        self.attn_pat = Self_Attn_Bot(
            self.args.trans_linear_in_dim, self.num_patches)

        # MLP-mixing frame-level enrichment over the 8 frames.
        self.fr_enrich = MLP_Mix_Enrich(
            self.args.trans_linear_in_dim, self.args.seq_len)

    @autocast()
    def forward(self, context_images, context_labels, target_images):
        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        # '''
        context_features = self.resnet(context_images)  # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_images)  # 160 x 2048 x 7 x 7

        # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(
            context_features)  # 200 x 2048 x 4 x 4
        target_features = self.adap_max(target_features)  # 160 x 2048 x 4 x 4

        # Reshape before averaging across all the patches
        context_features = context_features.reshape(
            -1, self.args.trans_linear_in_dim, self.num_patches)  # 200 x 2048 x 16
        target_features = target_features.reshape(
            -1, self.args.trans_linear_in_dim, self.num_patches)  # 160 x 2048 x 16

        # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1)  # 200 x 16 x 2048
        target_features = target_features.permute(0, 2, 1)  # 160 x 16 x 2048

        # Performing self-attention across the 16 patches
        # context_features = self.attn_pat(context_features)  # 200 x 16 x 2048
        # target_features = self.attn_pat(target_features)  # 160 x 16 x 2048

        # Average across the patches
        context_features = torch.mean(context_features, dim=1)  # 200 x 2048
        target_features = torch.mean(target_features, dim=1)  # 160 x 2048

        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 25 x 8 x 2048
        target_features = target_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 20 x 8 x 2048

        # Compute logits using the new loss before applying frame-level attention
        # all_logits_post_pat = [n(context_features, context_labels, target_features)[
        #     'logits'] for n in self.new_dist_loss_post_pat]
        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        # all_logits_post_pat = torch.stack(all_logits_post_pat, dim=-1)

        # Combing the patch and frame-level logits
        # sample_logits_post_pat = all_logits_post_pat
        # sample_logits_post_pat = torch.mean(
        #     sample_logits_post_pat, dim=[-1])  # 20 x 5

        # Perform self-attention across the 8 frames
        # context_features_fr = self.fr_enrich(context_features)  # 25 x 8 x 2048
        # target_features_fr = self.fr_enrich(target_features)  # 20 x 8 x 2048

        '''
            For different temporal lengths(2, 3, ...) get the final logits and perform mean.
        '''

        # Frame-level logits
        # all_logits_fr = [t(context_features_fr, context_labels, target_features_fr)[
        #     'logits'] for t in self.transformers]
        all_logits_fr = [t(context_features, context_labels, target_features)[
            'logits'] for t in self.transformers]

        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1)

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1])  # 20 x 5

        # return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
        #                'logits_post_pat': split_first_dim_linear(sample_logits_post_pat, [NUM_SAMPLES, target_features.shape[0]])}
        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
                       'logits_post_pat': torch.zeros((1))}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            # self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(
                self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)
            self.new_dist_loss_post_pat = [
                n.cuda(0) for n in self.new_dist_loss_post_pat]
            # self.new_dist_loss_post_pat = torch.nn.DataParallel([n.cuda(0) for n in self.new_dist_loss_post_pat])

            self.attn_pat.cuda(0)
            self.attn_pat = torch.nn.DataParallel(
                self.attn_pat, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.fr_enrich.cuda(0)
            self.fr_enrich = torch.nn.DataParallel(self.fr_enrich, device_ids=[
                                                   i for i in range(0, self.args.num_gpus)])

    def extract_feature(self, images):
        """
        获取一个视频的图片帧经过resnet后的特征.
        :param images: A batch of images
        :return: A batch of features
        """

        # images: batch_size x 3 x 224 x 224, 一般的，batch_size = 8
        context_features = self.resnet(images)  # batch_size x 2048 x 7 x 7

        # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(
            context_features)  # batch_size x 2048 x 4 x 4

        # Reshape before averaging across all the patches
        context_features = context_features.reshape(
            -1, self.args.trans_linear_in_dim, self.num_patches)  # batch_size x 2048 x 16

        # Permute
        context_features = context_features.permute(
            0, 2, 1)  # batch_size x 16 x 2048

        # Average across the patches
        context_features = torch.mean(
            context_features, dim=1)  # batch_size x 2048
        return context_features

    def compute_logis(self, context_features, target_features, context_labels):
        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim).cuda(0)  # 25 x 8 x 2048
        target_features = target_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim).cuda(0)  # 20 x 8 x 2048
        context_labels = context_labels.cuda(0)
        # Compute logits using the new loss before applying frame-level attention
        all_logits_post_pat = [n(context_features, context_labels, target_features)[
            'logits'] for n in self.new_dist_loss_post_pat]
        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_post_pat = torch.stack(all_logits_post_pat, dim=-1)

        # Combing the patch and frame-level logits
        sample_logits_post_pat = all_logits_post_pat
        sample_logits_post_pat = torch.mean(
            sample_logits_post_pat, dim=[-1])  # 20 x 5

        # Perform self-attention across the 8 frames
        context_features_fr = self.fr_enrich(context_features)  # 25 x 8 x 2048
        target_features_fr = self.fr_enrich(target_features)  # 20 x 8 x 2048

        '''
            For different temporal lengths(2, 3, ...) get the final logits and perform mean.
        '''

        # Frame-level logits
        all_logits_fr = [t(context_features_fr, context_labels, target_features_fr)[
            'logits'] for t in self.transformers]
        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1)

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1])  # 20 x 5

        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
                       'logits_post_pat': split_first_dim_linear(sample_logits_post_pat, [NUM_SAMPLES, target_features.shape[0]])}

        return return_dict


class CorrelationTRX(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, args):
        super(CorrelationTRX, self).__init__()

        self.train()
        self.args = args

        # Using ResNet Backbone
        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.num_patches = 16

        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))

        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList(
            [TemporalCrossTransformer(args, s) for s in args.temp_set])
        
        self.loss = My_Loss()

    def forward(self, context_images, context_labels, target_images):
        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        # '''
        context_features = self.resnet(context_images)  # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_images)  # 160 x 2048 x 7 x 7

        # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(
            context_features)  # 200 x 2048 x 4 x 4
        target_features = self.adap_max(target_features)  # 160 x 2048 x 4 x 4

        # Reshape before averaging across all the patches
        context_features = context_features.reshape(
            -1, self.args.trans_linear_in_dim, self.num_patches)  # 200 x 2048 x 16
        target_features = target_features.reshape(
            -1, self.args.trans_linear_in_dim, self.num_patches)  # 160 x 2048 x 16

        # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1)  # 200 x 16 x 2048
        target_features = target_features.permute(0, 2, 1)  # 160 x 16 x 2048

        # Performing self-attention across the 16 patches
        # context_features = self.attn_pat(context_features)  # 200 x 16 x 2048
        # target_features = self.attn_pat(target_features)  # 160 x 16 x 2048

        # Average across the patches
        context_features = torch.mean(context_features, dim=1)  # 200 x 2048
        target_features = torch.mean(target_features, dim=1)  # 160 x 2048

        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 25 x 8 x 2048
        target_features = target_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 20 x 8 x 2048

        # 计算支持集相似度损失
        support_videos = [context_features[i * self.args.shot: (i+1) * self.args.shot] for i in range(self.args.way)]
        my_loss = self.loss(support_videos)


        '''
            For different temporal lengths(2, 3, ...) get the final logits and perform mean.
        '''

        # Frame-level logits
        # all_logits_fr = [t(context_features_fr, context_labels, target_features_fr)[
        #     'logits'] for t in self.transformers]
        all_logits_fr = [t(context_features, context_labels, target_features)[
            'logits'] for t in self.transformers]

        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1)

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1])  # 20 x 5

        # return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
        #                'logits_post_pat': split_first_dim_linear(sample_logits_post_pat, [NUM_SAMPLES, target_features.shape[0]])}
        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
                       "my_loss": my_loss}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            # self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=list(range(self.args.num_gpus)))


            self.transformers.cuda(0)
            

    def extract_feature(self, images):
        """
        获取一个视频的图片帧经过resnet后的特征.
        :param images: A batch of images
        :return: A batch of features
        """

        # images: batch_size x 3 x 224 x 224, 一般的，batch_size = 8
        context_features = self.resnet(images)  # batch_size x 2048 x 7 x 7

        # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(
            context_features)  # batch_size x 2048 x 4 x 4

        # Reshape before averaging across all the patches
        context_features = context_features.reshape(
            -1, self.args.trans_linear_in_dim, self.num_patches)  # batch_size x 2048 x 16

        # Permute
        context_features = context_features.permute(
            0, 2, 1)  # batch_size x 16 x 2048

        # Average across the patches
        context_features = torch.mean(
            context_features, dim=1)  # batch_size x 2048
        return context_features


class TRM(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, args):
        super(TRM, self).__init__()

        self.train()
        self.args = args

        resnet = models.resnet50(pretrained=True)
        last_layer_idx = -1
        self.backbone = nn.Sequential(*list(resnet.children())[:last_layer_idx])

        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList(
            [TemporalCrossTransformer(args, s) for s in args.temp_set])

        # New-distance metric for post patch-level enriched features
        self.new_dist_loss_post_pat = [
            DistanceLoss(args, s) for s in args.temp_set]


    def forward(self, context_images, context_labels, target_images):
        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        # '''
        context_features = self.backbone(context_images)  # 200 x 2048 
        target_features = self.backbone(target_images)  # 160 x 2048 

        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 25 x 8 x 2048
        target_features = target_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 20 x 8 x 2048

       
        '''
            For different temporal lengths(2, 3, ...) get the final logits and perform mean.
        '''

        all_logits_fr = [t(context_features, context_labels, target_features)[
            'logits'] for t in self.transformers]

        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1)

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1])  # 20 x 5

        # return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
        #                'logits_post_pat': split_first_dim_linear(sample_logits_post_pat, [NUM_SAMPLES, target_features.shape[0]])}
        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
                       'logits_post_pat': torch.zeros((1))}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            # self.resnet.cuda(0)
            self.backbone = torch.nn.DataParallel(
                self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)
            self.new_dist_loss_post_pat = [
                n.cuda(0) for n in self.new_dist_loss_post_pat]
            # self.new_dist_loss_post_pat = torch.nn.DataParallel([n.cuda(0) for n in self.new_dist_loss_post_pat])
            
    def extract_feature(self, images):
        """
        获取一个视频的图片帧经过resnet后的特征.
        :param images: A batch of images
        :return: A batch of features
        """

        # images: batch_size x 3 x 224 x 224
        context_features = self.backbone(images)  
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 25 x 8 x 2048
        return context_features

class Branch(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, args, ids):
        super(Branch, self).__init__()

        self.train()
        self.args = args
        self.num_patches = 16
        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList(
            [TemporalCrossTransformer(args, s) for s in args.temp_set])

        # New-distance metric for post patch-level enriched features
        self.new_dist_loss_post_pat = [
            DistanceLoss(args, s) for s in args.temp_set]

        # MLP-mixing frame-level enrichment over the 8 frames.
        self.fr_enrich = MLP_Mix_Enrich(
            self.args.trans_linear_in_dim, self.args.seq_len)

    def forward(self, context_features, target_features, context_labels):
        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 25 x 8 x 2048
        target_features = target_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 20 x 8 x 2048
        # Compute logits using the new loss before applying frame-level attention
        self.new_dist_loss_post_pat = [
            n.to(context_features.device) for n in self.new_dist_loss_post_pat]
        all_logits_post_pat = [n(context_features, context_labels, target_features)[
            'logits'] for n in self.new_dist_loss_post_pat]
        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_post_pat = torch.stack(all_logits_post_pat, dim=-1)

        # Combing the patch and frame-level logits
        sample_logits_post_pat = all_logits_post_pat
        sample_logits_post_pat = torch.mean(
            sample_logits_post_pat, dim=[-1])  # 20 x 5

        # Perform self-attention across the 8 frames
        context_features_fr = self.fr_enrich(context_features)  # 25 x 8 x 2048
        target_features_fr = self.fr_enrich(target_features)  # 20 x 8 x 2048

        '''
            For different temporal lengths(2, 3, ...) get the final logits and perform mean.
        '''

        # Frame-level logits
        all_logits_fr = [t(context_features_fr, context_labels, target_features_fr)[
            'logits'] for t in self.transformers]
        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1)

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1])  # 20 x 5

        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
                       'logits_post_pat': split_first_dim_linear(sample_logits_post_pat, [NUM_SAMPLES, target_features.shape[0]])}

        return return_dict


class SingleBranch(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, args, ids):
        super(SingleBranch, self).__init__()

        self.train()
        self.args = args
        self.num_patches = 16
        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList(
            [TemporalCrossTransformer(args, s) for s in args.temp_set])

        # MLP-mixing frame-level enrichment over the 8 frames.
        self.fr_enrich = MLP_Mix_Enrich(
            self.args.trans_linear_in_dim, self.args.seq_len)

    def forward(self, context_features, target_features, context_labels):
        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 25 x 8 x 2048
        target_features = target_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 20 x 8 x 2048

        # Perform self-attention across the 8 frames
        context_features_fr = self.fr_enrich(context_features)  # 25 x 8 x 2048
        target_features_fr = self.fr_enrich(target_features)  # 20 x 8 x 2048

        # Frame-level logits
        all_logits_fr = [t(context_features_fr, context_labels, target_features_fr)[
            'logits'] for t in self.transformers]
        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1)

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1])  # 20 x 5

        return_dict = {'logits': split_first_dim_linear(
            sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]])}

        return return_dict

    def distribute_model(self):
        self.transformers = self.transformers.cuda(0)

        self.fr_enrich.cuda(0)
        self.fr_enrich = torch.nn.DataParallel(self.fr_enrich, device_ids=[
                                                i for i in range(0, self.args.num_gpus)])


class TrxBranch(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """
    def __init__(self, args, ids=0):
        super(TrxBranch, self).__init__()

        self.train()
        self.args = args
        self.num_patches = 16
        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList(
            [TemporalCrossTransformer(args, s) for s in args.temp_set])

    def forward(self, context_features, target_features, context_labels):
        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 25 x 8 x 2048
        target_features = target_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 20 x 8 x 2048

        # Frame-level logits
        all_logits_fr = [t(context_features, context_labels, target_features)[
            'logits'] for t in self.transformers]
        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1)

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1])  # 20 x 5

        return_dict = {'logits': split_first_dim_linear(
            sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]])}

        return return_dict

    def distribute_model(self, args):
        self.transformers = torch.nn.DataParallel(self.transformers, device_ids=[
                                                i for i in range(0, args.num_gpus)])


class TrainablePositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, frames_length = input_feat.shape[:2]
        position_ids = torch.arange(
            frames_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# three modality trx score fusion
class TSF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.m1_branch = TrxBranch(args, 0 % args.num_gpus)
        self.skeleton_branch = TrxBranch(args, 1 % args.num_gpus)
        self.flow_branch = TrxBranch(args, 2 % args.num_gpus)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        m1 = self.args.m1
        m2 = self.args.m2
        m3 = self.args.m3
        m1_context, m1_target, context_labels = context_features[m1], target_features[m1], context_labels
        m2_context, m2_target, context_labels = context_features[m2], target_features[m2], context_labels
        m3_context, m3_target, context_labels = context_features[m3], target_features[m3], context_labels
        m1_logits = self.m1_branch(m1_context, m1_target, context_labels)
        m2_logits = self.skeleton_branch(m2_context, m2_target, context_labels)
        m3_logits = self.flow_branch(m3_context, m3_target, context_labels)
        
        m1_logits = m1_logits['logits']
        m2_logits = m2_logits['logits']
        m3_logits = m3_logits['logits']
        
        return {"logits" : m1_logits * self.args.a + m2_logits * self.args.b + m3_logits * self.args.c}

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.m1_branch.cuda(self.ids[0 % self.args.num_gpus])
            self.skeleton_branch.cuda(self.ids[1 % self.args.num_gpus])
            self.flow_branch.cuda(self.ids[2 % self.args.num_gpus])
        else:
            self.cuda()

class FourTransforFusion(nn.Module):
    def __init__(self, args, out_channels=None, dropout=0.1):
        super().__init__()
        in_channels = 2048
        self.positionEncoding1 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding2 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding3 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding4 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels * 4, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2)
        self.f1 = nn.Linear(in_channels * 4, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, y1, y2, z1, z2, g1, g2):
        x1 = self.positionEncoding1(x1)
        x2 = self.positionEncoding1(x2)
        y1 = self.positionEncoding2(y1)
        y2 = self.positionEncoding2(y2)
        z1 = self.positionEncoding3(z1)
        z2 = self.positionEncoding3(z2)
        g1 = self.positionEncoding3(g1)
        g2 = self.positionEncoding3(g2)
        xyzg1 = torch.cat((torch.cat((torch.cat((x1, y1), dim=-1), z1), dim=-1), g1), dim=-1)
        xyzg2 = torch.cat((torch.cat((torch.cat((x2, y2), dim=-1), z2), dim=-1), g2), dim=-1)
        fusion1 = self.transformer_encoder(xyzg1)
        fusion2 = self.transformer_encoder(xyzg2)
        return self.dropout(self.f1(fusion1)), self.dropout(self.f1(fusion2))

    def extract_feature(self, x1, y1, z1):
        x1 = self.positionEncoding1(x1)
        y1 = self.positionEncoding2(y1)
        z1 = self.positionEncoding3(z1)
        xyz1 = torch.cat((torch.cat((x1, y1), dim=-1), z1), dim=-1)
        fusion1 = self.transformer_encoder(xyz1)
        return self.dropout(self.f1(fusion1))

class ThreeTransforFusion(nn.Module):
    def __init__(self, args, out_channels=None, dropout=0.1):
        super().__init__()
        in_channels = 2048
        self.positionEncoding1 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding2 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding3 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels * 3, nhead=3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1)
        self.f1 = nn.Linear(in_channels * 3, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, y1, z1):
        x1 = self.positionEncoding1(x1)
        y1 = self.positionEncoding2(y1)
        z1 = self.positionEncoding3(z1)
        xyz1 = torch.cat((torch.cat((x1, y1), dim=-1), z1), dim=-1)
        fusion1 = self.transformer_encoder(xyz1)
        return self.dropout(self.f1(fusion1))

    def extract_feature(self, x1, y1, z1):
        x1 = self.positionEncoding1(x1)
        y1 = self.positionEncoding2(y1)
        z1 = self.positionEncoding3(z1)
        xyz1 = torch.cat((torch.cat((x1, y1), dim=-1), z1), dim=-1)
        fusion1 = self.transformer_encoder(xyz1)
        return self.dropout(self.f1(fusion1))

class ThreeTransforTask(nn.Module):
    def __init__(self, args, out_channels=None, dropout=0.1):
        super().__init__()
        in_channels = 2048
        self.positionEncoding1 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding2 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding3 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels * 3, nhead=3)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2)
        self.f1 = nn.Linear(in_channels * 3, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, y1, z1):
        x1 = self.positionEncoding1(x1)
        y1 = self.positionEncoding2(y1)
        z1 = self.positionEncoding3(z1)
        xyz1 = torch.cat((torch.cat((x1, y1), dim=-1), z1), dim=-1)
        fusion1 = self.transformer_encoder(xyz1)
        return self.dropout(self.f1(fusion1))

    def extract_feature(self, x1, y1, z1):
        x1 = self.positionEncoding1(x1)
        y1 = self.positionEncoding2(y1)
        z1 = self.positionEncoding3(z1)
        xyz1 = torch.cat((torch.cat((x1, y1), dim=-1), z1), dim=-1)
        fusion1 = self.transformer_encoder(xyz1)
        return self.dropout(self.f1(fusion1))

class ThreeTransforTemproal(nn.Module):
    def __init__(self, args, out_channels=None, dropout=0.1):
        super().__init__()
        in_channels = 2048
        self.positionEncoding1 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding2 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding3 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels * 3, nhead=3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=args.trans_num)
        self.f1 = nn.Linear(in_channels * 3, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, y1, z1):
        x1 = self.positionEncoding1(x1)
        y1 = self.positionEncoding2(y1)
        z1 = self.positionEncoding3(z1)
        xyz1 = torch.cat((torch.cat((x1, y1), dim=-1), z1), dim=-1)
        fusion1 = self.transformer_encoder(xyz1)
        return self.dropout(self.f1(fusion1))

    def extract_feature(self, x1, y1, z1):
        x1 = self.positionEncoding1(x1)
        y1 = self.positionEncoding2(y1)
        z1 = self.positionEncoding3(z1)
        xyz1 = torch.cat((torch.cat((x1, y1), dim=-1), z1), dim=-1)
        fusion1 = self.transformer_encoder(xyz1)
        return self.dropout(self.f1(fusion1))

class FourTransforTemproal(nn.Module):
    def __init__(self, args, out_channels=None, dropout=0.1):
        super().__init__()
        in_channels = 2048
        self.positionEncoding1 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding2 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding3 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding4 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels * 4, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=args.trans_num)
        self.f1 = nn.Linear(in_channels * 4, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, y1, z1, h1):
        x1 = self.positionEncoding1(x1)
        y1 = self.positionEncoding2(y1)
        z1 = self.positionEncoding3(z1)
        h1 = self.positionEncoding4(h1)
        xyzh1 = torch.cat((torch.cat((torch.cat((x1, y1), dim=-1), z1), dim=-1), h1),dim=-1)
        fusion1 = self.transformer_encoder(xyzh1)
        return self.dropout(self.f1(fusion1))

class TwoTransforFusion(nn.Module):
    def __init__(self, args, out_channels=None, dropout=0.1):
        super().__init__()
        in_channels = 2048
        self.positionEncoding1 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding2 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels * 2, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=args.trans_num)
        self.f1 = nn.Linear(in_channels * 2, in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, y1, y2):
        x1 = self.positionEncoding1(x1)
        x2 = self.positionEncoding1(x2)
        y1 = self.positionEncoding2(y1)
        y2 = self.positionEncoding2(y2)
        xyz1 = torch.cat((x1, y1), dim=-1)
        xyz2 = torch.cat((x2, y2), dim=-1)
        fusion1 = self.transformer_encoder(xyz1)
        fusion2 = self.transformer_encoder(xyz2)
        return self.dropout(self.f1(fusion1)), self.dropout(self.f1(fusion2))

    def extract_feature(self, x1, y1):
        x1 = self.positionEncoding1(x1)
        y1 = self.positionEncoding2(y1)
        xy1 = torch.cat((x1, y1), dim=-1)
        fusion1 = self.transformer_encoder(xy1)
        return self.dropout(self.f1(fusion1))
    
class TwoTRX(nn.Module):
    def __init__(self, args):
        super(TwoTRX, self).__init__()
        print("当前使用的是双模态融合代码")
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        
        fusion_context, fusion_target = self.fusion(
            first_context, first_target, second_context, second_target,)

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model()

            self.fusion.cuda(0)
            self.fusion = torch.nn.DataParallel(self.fusion, device_ids=[
                                                   i for i in range(0, self.args.num_gpus)])

class TwoCross(nn.Module):
    def __init__(self, args):
        super(TwoCross, self).__init__()
        # print("当前使用的是三模态的融合")
        self.args = args
        self.args.num_attention_heads = 2
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = BertAttention(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        
        fusion_context = self.fusion(first_context, second_context)
        
        fusion_target = self.fusion(first_target, second_target)

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits
    
    def extract(self, first_context, first_target, second_context, second_target):
        fusion_context = self.fusion(first_context, second_context)
        fusion_target = self.fusion(first_target, second_target)
        return fusion_context, fusion_target

class ThreeCross(nn.Module):
    def __init__(self, args):
        super(ThreeCross, self).__init__()
        self.args = args
        self.args.num_attention_heads = 2
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion1 = BertAttention(args)
        self.fusion2 = BertAttention(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        fusion_context1 = self.fusion1(first_context, second_context)
        fusion_target1 = self.fusion1(first_target, second_target)
        
        fusion_context2 = self.fusion1(first_context, third_context)
        fusion_target2 = self.fusion1(first_target, third_target)
        
        fusion_context = self.fusion2(fusion_context1, fusion_context2)
        fusion_target = self.fusion2(fusion_target1, fusion_target2)

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits
    
    def extract(self, first_context, first_target, second_context, second_target):
        fusion_context = self.fusion1(first_context, second_context)
        fusion_target = self.fusion1(first_target, second_target)
        return fusion_context, fusion_target

class TwoTRXShuffleTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        
        fusion_context1, fusion_target1 = self.fusion(
            first_context, first_target, second_context, second_target,)

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)
        
        second_target_shuffle = second_target[:, :self.args.shirt_num, :]
        second_target_prefix = second_target[:, self.args.shirt_num:, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)


        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle,
        )

        fusion_context = fusion_context1 + fusion_context2
        fusion_target = fusion_target1 + fusion_target2
        del fusion_context1, fusion_context2, fusion_target1, fusion_target2

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

class ThreeTRXShuffleTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        second_context_shuffle = second_context[:, 1:, :]
        second_context_shuffle = nn.functional.pad(second_context_shuffle, (0, 0, 0, 1), 'constant', 0)
        
        second_target_shuffle = second_target[:, 1:, :]
        second_target_shuffle = nn.functional.pad(second_target_shuffle, (0, 0, 0, 1), 'constant', 0)

        third_context_shuffle = third_context[:, :7, :]
        third_context_shuffle = nn.functional.pad(third_context_shuffle, (0, 0, 1, 0), 'constant', 0)

        third_target_shuffle = third_target[:, :7, :]
        third_target_shuffle = nn.functional.pad(third_target_shuffle, (0, 0, 1, 0), 'constant', 0)

        fusion_context1, fusion_target1 = self.fusion(
            first_context, first_target, second_context, second_target,)

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle,
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context_shuffle, third_target_shuffle
        )

        fusion_context = fusion_context1 + fusion_context2 + fusion_context3
        fusion_target = fusion_target1 + fusion_target2 + fusion_target3
        del fusion_context1, fusion_context2, fusion_target1, fusion_target2, fusion_context3, fusion_target3

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

class ThreeTRXShiftLoopTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.three_fusion = ThreeTransforTemproal(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)

        second_target_shuffle = second_target[:, self.args.shirt_num:, :]
        second_target_prefix = second_target[:, :self.args.shirt_num, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)

        third_context_shuffle = third_context[:, self.args.shirt_num:, :]
        third_context_suffix = third_context[:, :self.args.shirt_num, :]
        third_context_shuffle = torch.cat((third_context_suffix, third_context_shuffle), dim=1)

        third_target_shuffle = third_target[:, self.args.shirt_num:, :]
        third_target_suffix = third_target[:, :self.args.shirt_num, :]
        third_target_shuffle = torch.cat((third_target_suffix, third_target_shuffle), dim=1)

        fusion_context1 = self.three_fusion(first_context, second_context, third_context)
        fusion_target1 = self.three_fusion(first_target, second_target, third_target)

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle,
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context_shuffle, third_target_shuffle
        )

        fusion_context = fusion_context1 + fusion_context2 + fusion_context3
        fusion_target = fusion_target1 + fusion_target2 + fusion_target3
        del fusion_context1, fusion_context2, fusion_target1, fusion_target2, fusion_context3, fusion_target3

        return self.feature_test(fusion_context, context_labels, fusion_target)

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model(self.args)
            self.fusion.distribute_model(self.args)
            self.three_fusion.distribute_model(self.args)

    def extract_feature(self, feature):
        rgb = feature['rgb'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim).cuda()
        depth = feature['depth'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim).cuda()
        flow = feature['flow'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim).cuda()
        feature1 = self.three_fusion.extract_feature(rgb, depth, flow)
        
        feature2 = depth[:, self.args.shirt_num:, :]
        feature2_prefix = depth[:, :self.args.shirt_num, :]
        feature2 = torch.cat((feature2, feature2_prefix), dim=1)
        feature2 = self.fusion.extract_feature(rgb, feature2)
        
        feature3 = flow[:, self.args.shirt_num:, :]
        feature3_prefix = flow[:, :self.args.shirt_num, :]
        feature3 = torch.cat((feature3, feature3_prefix), dim=1)
        feature3 = self.fusion.extract_feature(rgb, feature3)
        
        return feature1 + feature2 + feature3

    def extract_task_feature(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)
        
        second_target_shuffle = second_target[:, self.args.shirt_num:, :]
        second_target_prefix = second_target[:, :self.args.shirt_num, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)

        third_context_shuffle = third_context[:, self.args.shirt_num:, :]
        third_context_suffix = third_context[:, :self.args.shirt_num, :]
        third_context_shuffle = torch.cat((third_context_suffix, third_context_shuffle), dim=1)

        third_target_shuffle = third_target[:, self.args.shirt_num:, :]
        third_target_suffix = third_target[:, :self.args.shirt_num, :]
        third_target_shuffle = torch.cat((third_target_suffix, third_target_shuffle), dim=1)

        fusion_context1, fusion_target1 = self.three_fusion(
            first_context, first_target, second_context, second_target, third_context, third_target)

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle,
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context_shuffle, third_target_shuffle
        )

        fusion_context = fusion_context1 + fusion_context2 + fusion_context3
        fusion_target = fusion_target1 + fusion_target2 + fusion_target3
        del fusion_context1, fusion_context2, fusion_target1, fusion_target2, fusion_context3, fusion_target3
        return fusion_context, fusion_target
 
    def feature_test(self, fusion_context, context_labels, fusion_target):
        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        return self.bracnch(fusion_context, fusion_target, context_labels)


class FourShiftFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.four_fusion = FourTransforTemproal(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        fourth_context, fourth_target, context_labels = context_features[self.args.m4].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m4].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)

        second_target_shuffle = second_target[:, self.args.shirt_num:, :]
        second_target_prefix = second_target[:, :self.args.shirt_num, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)

        third_context_shuffle = third_context[:, self.args.shirt_num:, :]
        third_context_suffix = third_context[:, :self.args.shirt_num, :]
        third_context_shuffle = torch.cat((third_context_suffix, third_context_shuffle), dim=1)

        third_target_shuffle = third_target[:, self.args.shirt_num:, :]
        third_target_suffix = third_target[:, :self.args.shirt_num, :]
        third_target_shuffle = torch.cat((third_target_suffix, third_target_shuffle), dim=1)

        fourth_context_shuffle = fourth_context[:, self.args.shirt_num:, :]
        fourth_context_suffix = fourth_context[:, :self.args.shirt_num, :]
        fourth_context_shuffle = torch.cat((fourth_context_suffix, fourth_context_shuffle), dim=1)

        fourth_target_shuffle = fourth_target[:, self.args.shirt_num:, :]
        fourth_target_suffix = fourth_target[:, :self.args.shirt_num, :]
        fourth_target_shuffle = torch.cat((fourth_target_suffix, fourth_target_shuffle), dim=1)

        fusion_context1 = self.four_fusion(first_context, second_context, third_context, fourth_context)
        fusion_target1 = self.four_fusion(first_target, second_target, third_target, fourth_target)

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle,
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context_shuffle, third_target_shuffle
        )

        fusion_context4, fusion_target4 = self.fusion(
            first_context, first_target, fourth_context_shuffle, fourth_target_shuffle
        )

        fusion_context = fusion_context1 + fusion_context2 + fusion_context3 + fusion_context4
        fusion_target = fusion_target1 + fusion_target2 + fusion_target3 + fusion_target4
        del fusion_context1, fusion_context2, fusion_target1, fusion_target2, fusion_context3, fusion_target3

        return self.feature_test(fusion_context, context_labels, fusion_target)


    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model(self.args)
            self.fusion.distribute_model(self.args)
            self.four_fusion.distribute_model(self.args)

    def extract_feature(self, feature):
        pass
    

    def feature_test(self, fusion_context, context_labels, fusion_target):
        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        return self.bracnch(fusion_context, fusion_target, context_labels)


class FiveShiftFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.three_fusion = ThreeTransforTemproal(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        fourth_context, fourth_target, context_labels = context_features[self.args.m4].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m4].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        fifth_context, fifth_target, context_labels = context_features[self.args.m5].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m5].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)

        second_target_shuffle = second_target[:, self.args.shirt_num:, :]
        second_target_prefix = second_target[:, :self.args.shirt_num, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)

        third_context_shuffle = third_context[:, self.args.shirt_num:, :]
        third_context_suffix = third_context[:, :self.args.shirt_num, :]
        third_context_shuffle = torch.cat((third_context_suffix, third_context_shuffle), dim=1)

        third_target_shuffle = third_target[:, self.args.shirt_num:, :]
        third_target_suffix = third_target[:, :self.args.shirt_num, :]
        third_target_shuffle = torch.cat((third_target_suffix, third_target_shuffle), dim=1)

        fourth_context_shuffle = fourth_context[:, self.args.shirt_num:, :]
        fourth_context_suffix = fourth_context[:, :self.args.shirt_num, :]
        fourth_context_shuffle = torch.cat((fourth_context_suffix, fourth_context_shuffle), dim=1)

        fourth_target_shuffle = fourth_target[:, self.args.shirt_num:, :]
        fourth_target_suffix = fourth_target[:, :self.args.shirt_num, :]
        fourth_target_shuffle = torch.cat((fourth_target_suffix, fourth_target_shuffle), dim=1)

        fifth_context_shuffle = fifth_context[:, self.args.shirt_num:, :]
        fifth_context_prefix = fifth_context[:, :self.args.shirt_num, :]
        fifth_context_shuffle = torch.cat((fifth_context_shuffle, fifth_context_prefix), dim=1)


        fifth_target_shuffle = fifth_target[:, self.args.shirt_num:, :]
        fifth_context_prefix = fifth_target[:, :self.args.shirt_num, :]
        fifth_target_shuffle = torch.cat((fifth_target_shuffle, fifth_context_prefix), dim=1)

        fusion_context1 = self.three_fusion(first_context, second_context, third_context)
        fusion_target1 = self.three_fusion(first_target, second_target, third_target)

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle,
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context_shuffle, third_target_shuffle
        )

        fusion_context4, fusion_target4 = self.fusion(
            first_context, first_target, fourth_context_shuffle, fourth_target_shuffle
        )

        fusion_context5, fusion_target5 = self.fusion(
            first_context, first_target, fifth_context_shuffle, fifth_target_shuffle
        )

        fusion_context = fusion_context1 + fusion_context2 + fusion_context3 + fusion_context4 + fusion_context5
        fusion_target = fusion_target1 + fusion_target2 + fusion_target3 + fusion_target4 + fusion_target5
        del fusion_context1, fusion_context2, fusion_target1, fusion_target2, fusion_context3, fusion_target3, fusion_context4, fusion_target4, fusion_context5, fusion_target5

        return self.feature_test(fusion_context, context_labels, fusion_target)


    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model(self.args)
            self.fusion.distribute_model(self.args)
            self.three_fusion.distribute_model(self.args)

    def extract_feature(self, feature):
        pass
    

    def feature_test(self, fusion_context, context_labels, fusion_target):
        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        return self.bracnch(fusion_context, fusion_target, context_labels)

class OTAMThreeTRXShiftLoopTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = CNN_OTAM()
        self.fusion = TwoTransforFusion(args)
        self.three_fusion = ThreeTransforTemproal(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)

        second_target_shuffle = second_target[:, self.args.shirt_num:, :]
        second_target_prefix = second_target[:, :self.args.shirt_num, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)

        third_context_shuffle = third_context[:, self.args.shirt_num:, :]
        third_context_suffix = third_context[:, :self.args.shirt_num, :]
        third_context_shuffle = torch.cat((third_context_suffix, third_context_shuffle), dim=1)

        third_target_shuffle = third_target[:, self.args.shirt_num:, :]
        third_target_suffix = third_target[:, :self.args.shirt_num, :]
        third_target_shuffle = torch.cat((third_target_suffix, third_target_shuffle), dim=1)

        fusion_context1 = self.three_fusion(first_context, second_context, third_context)
        fusion_target1 = self.three_fusion(first_target, second_target, third_target)

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle,
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context_shuffle, third_target_shuffle
        )
        
        fusion_context = fusion_context1 + fusion_context2 + fusion_context3
        fusion_target = fusion_target1 + fusion_target2 + fusion_target3

        del fusion_context1, fusion_context2, fusion_target1, fusion_target2, fusion_context3, fusion_target3

        return self.feature_test(fusion_context, context_labels, fusion_target)

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model(self.args)
            self.fusion.distribute_model(self.args)
            self.three_fusion.distribute_model(self.args)
 
    def feature_test(self, fusion_context, context_labels, fusion_target):
        return self.bracnch(fusion_context, context_labels, fusion_target)
  
class ScoreFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.three_fusion = ThreeTransforTemproal(args)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        fusion_context = self.three_fusion(first_context, second_context, third_context)
        fusion_target = self.three_fusion(first_target, second_target, third_target)
        return self.feature_test(fusion_context, context_labels, fusion_target)

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model(self.args)
 
    def feature_test(self, fusion_context, context_labels, fusion_target):
        return self.bracnch(fusion_context, fusion_target, context_labels)
  
class TwoCombinationTRX(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context, second_target
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context, third_target
        )

        fusion_context = fusion_context2 + fusion_context3
        fusion_target = fusion_target2 + fusion_target3
        del fusion_context2, fusion_target2, fusion_context3, fusion_target3

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        return self.bracnch(fusion_context, fusion_target, context_labels)

class TwoCombinationCTX(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion1 = TwoCross(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        fusion_context2, fusion_target2 = self.fusion1.extract(
            first_context, first_target, second_context, second_target
        )

        fusion_context3, fusion_target3 = self.fusion1.extract(
            first_context, first_target, third_context, third_target
        )

        fusion_context = fusion_context2 + fusion_context3
        fusion_target = fusion_target2 + fusion_target3
        del fusion_context2, fusion_target2, fusion_context3, fusion_target3

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        return self.bracnch(fusion_context, fusion_target, context_labels)

class ThreeCombinationTRX(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        four_context, four_target, context_labels = context_features[self.args.m4].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m4].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels


        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context, second_target
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context, third_target
        )

        fusion_context4, fusion_target4 = self.fusion(
            first_context, first_target, four_context, four_target
        )

        fusion_context = fusion_context2 + fusion_context3 + fusion_context4 
        fusion_target = fusion_target2 + fusion_target3 + fusion_target4
        del fusion_context2, fusion_target2, fusion_context3, fusion_target3

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        return self.bracnch(fusion_context, fusion_target, context_labels)
    
class TwoCombinationShiftTRX(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)
        
        second_target_shuffle = second_target[:, self.args.shirt_num:, :]
        second_target_prefix = second_target[:, :self.args.shirt_num, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)

        third_context_shuffle = third_context[:, self.args.shirt_num:, :]
        third_context_suffix = third_context[:, :self.args.shirt_num, :]
        third_context_shuffle = torch.cat((third_context_suffix, third_context_shuffle), dim=1)

        third_target_shuffle = third_target[:, self.args.shirt_num:, :]
        third_target_suffix = third_target[:, :self.args.shirt_num, :]
        third_target_shuffle = torch.cat((third_target_suffix, third_target_shuffle), dim=1)


        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context_shuffle, third_target_shuffle
        )

        fusion_context =  fusion_context2 + fusion_context3
        fusion_target = fusion_target2 + fusion_target3
        del fusion_context2, fusion_target2, fusion_context3, fusion_target3

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        return self.bracnch(fusion_context, fusion_target, context_labels)

class model_distillation(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        distillation = timm.create_model("deit_small_distilled_patch16_224", pretrained=True)
        distillation.reset_classifier(0)
        self.convnet = distillation
        self.fc = nn.Linear(384, num_classes)
    
    def forward(self,x):
        feature = self.convnet(x)
        if(type(feature) == tuple):
            feature = feature[0]
        feature = feature.reshape(x.shape[0], -1)
        feature = self.fc(feature)
        return feature

class TwoCombinationTemTroShiftTRX(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.three_fusion = ThreeTransforTask(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)
        
        second_target_shuffle = second_target[:, self.args.shirt_num:, :]
        second_target_prefix = second_target[:, :self.args.shirt_num, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)

        third_context_shuffle = third_context[:, self.args.shirt_num:, :]
        third_context_suffix = third_context[:, :self.args.shirt_num, :]
        third_context_shuffle = torch.cat((third_context_suffix, third_context_shuffle), dim=1)

        third_target_shuffle = third_target[:, self.args.shirt_num:, :]
        third_target_suffix = third_target[:, :self.args.shirt_num, :]
        third_target_shuffle = torch.cat((third_target_suffix, third_target_shuffle), dim=1)

        fusion_context1 =  self.three_fusion(first_context, second_context, third_context)

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context_shuffle, third_target_shuffle
        )

        fusion_context =  fusion_context1 + fusion_context2 + fusion_context3
        fusion_target = fusion_target2 + fusion_target3
        del fusion_context1, fusion_context2, fusion_target2, fusion_context3, fusion_target3

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        return self.bracnch(fusion_context, fusion_target, context_labels)
    
class ThreeTRXLRShiftLoopTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)
        
        second_target_shuffle = second_target[:, :self.args.shirt_num, :]
        second_target_prefix = second_target[:, self.args.shirt_num:, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)

        third_context_shuffle = third_context[:, self.args.shirt_num:, :]
        third_context_suffix = third_context[:, :self.args.shirt_num, :]
        third_context_shuffle = torch.cat((third_context_suffix, third_context_shuffle), dim=1)

        third_target_shuffle = third_target[:, :self.args.shirt_num, :]
        third_target_suffix = third_target[:, self.args.shirt_num:, :]
        third_target_shuffle = torch.cat((third_target_suffix, third_target_shuffle), dim=1)

        fusion_context1, fusion_target1 = self.fusion(
            first_context, first_target, second_context, second_target,)

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle,
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context_shuffle, third_target_shuffle
        )

        fusion_context = fusion_context1 + fusion_context2 + fusion_context3
        fusion_target = fusion_target1 + fusion_target2 + fusion_target3
        del fusion_context1, fusion_context2, fusion_target1, fusion_target2, fusion_context3, fusion_target3

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

class ThreeStrm(nn.Module):
    def __init__(self, args):
        super(ThreeStrm, self).__init__()
        self.args = args
        in_channels = self.args.trans_linear_in_dim
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion_temproal = ThreeTransforTemproal(args)
        self.f1 = nn.Linear(in_channels * 3, in_channels)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        m1_context, m1_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        m2_context, m2_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        m3_context, m3_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        
        fusion_context = self.fusion_temproal(m1_context, m2_context, m3_context)
        fusion_target = self.fusion_temproal(m1_target, m2_target, m3_target)
        
        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model()
            self.fusion.cuda(0)
            self.fusion = torch.nn.DataParallel(self.fusion, device_ids=[
                                                   i for i in range(0, self.args.num_gpus)])

    def test_only(self, fusion_context, context_labels, fusion_target):
        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

    def extract_feature(self, feature):
        rgb = feature['rgb'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim).cuda()
        skeleton = feature['skeleton'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim).cuda()
        flow = feature['flow'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim).cuda()
        feature = self.fusion.extract_feature(rgb, skeleton, flow)
        return feature


    def extract_task_feature(self, context_features, context_labels, target_features):
        m1_context, m1_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        m2_context, m2_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        m3_context, m3_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        
        fusion_context, fusion_target = self.fusion(
            m1_context, m1_target, m2_context, m2_target, m3_context, m3_target)
        return fusion_context, fusion_target

    def feature_test(self, fusion_context, context_labels, fusion_target):
        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

class FourStrm(nn.Module):
    def __init__(self, args):
        super(FourStrm, self).__init__()
        # print("当前使用的是双模态融合代码")
        print("当前使用的是r+s+f+d的融合")
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = FourTransforFusion(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        rgb_context, rgb_target, context_labels = context_features['rgb'].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features['rgb'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        skeleton_context, skeleton_target, context_labels = context_features['skeleton'].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features['skeleton'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        flow_context, flow_target, context_labels = context_features['flow'].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features['flow'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        depth_context, depth_target, depth_labels = context_features['depth'].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features['depth'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        
        fusion_context, fusion_target = self.fusion(
            rgb_context, rgb_target, skeleton_context, skeleton_target, flow_context, flow_target, depth_context, depth_target)
        # fusion_context, fusion_target = self.fusion(
        #     rgb_context, rgb_target, flow_context, flow_target)
        # fusion_context, fusion_target = self.fusion(
        #     rgb_context, rgb_target, skeleton_context, skeleton_target)

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model()

            self.fusion.cuda(0)
            self.fusion = torch.nn.DataParallel(self.fusion, device_ids=[
                                                   i for i in range(0, self.args.num_gpus)])

class ResnetBranch(nn.Module):
    def __init__(self, args):
        super(ResnetBranch, self).__init__()

        self.train()
        self.args = args

        # Using ResNet Backbone
        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.num_patches = 16

        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))
        self.distance = CosDistance(args)

    def forward(self, context_images, context_labels, target_images):
        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        '''
        context_features = self.resnet(context_images)  # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_images)  # 160 x 2048 x 7 x 7

        # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(
            context_features)  # 200 x 2048 x 4 x 4
        target_features = self.adap_max(target_features)  # 160 x 2048 x 4 x 4

        # Reshape before averaging across all the patches
        context_features = context_features.reshape(
            -1, self.args.trans_linear_in_dim, self.num_patches)  # 200 x 2048 x 16
        target_features = target_features.reshape(
            -1, self.args.trans_linear_in_dim, self.num_patches)  # 160 x 2048 x 16

        # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1)  # 200 x 16 x 2048
        target_features = target_features.permute(0, 2, 1)  # 160 x 16 x 2048


        # Average across the patches
        context_features = torch.mean(context_features, dim=1)  # 200 x 2048
        target_features = torch.mean(target_features, dim=1)  # 160 x 2048

        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 25 x 8 x 2048
        target_features = target_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 20 x 8 x 2048

        sample_logits_fr = self.distance(context_features, context_labels, target_features)

        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]])}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            # self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(
                self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

class DGAdaIN(nn.Module):
    def __init__(self, in_channels=2048, out_channels=2048):
        super(DGAdaIN, self).__init__()

        self.affine_scale = nn.Linear(in_channels, out_channels, bias=True)
        self.affine_bias = nn.Linear(in_channels, out_channels, bias=True)
        self.norm = nn.InstanceNorm1d(in_channels, affine = False, momentum=0.9,  track_running_stats=False)

    def forward(self, x, w):
        y_scale = 1 + self.affine_scale(w)
        y_bias = 0 + self.affine_bias(w)
        x = self.norm(x)
        x_scale = (x * y_scale) + y_bias
        return x_scale

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model()
            self.fusion1.cuda(0)
            self.fusion1 = torch.nn.DataParallel(self.fusion, device_ids=[
                                                   i for i in range(0, self.args.num_gpus)])
            self.fusion2.cuda(0)
            self.fusion2 = torch.nn.DataParallel(self.fusion, device_ids=[
                                                   i for i in range(0, self.args.num_gpus)])

class ThreeFusionDGA(nn.Module):
    def __init__(self, args):
        super(ThreeFusionDGA, self).__init__()
        # print("当前使用的是双模态融合代码")
        print("当前使用的是三模态的ThreeFusionDGA融合")
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion1 = TwoTransforFusion(args)
        self.fusion2 = DGAdaIN(self.args.trans_linear_in_dim, self.args.trans_linear_in_dim)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        rgb_context, rgb_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        skeleton_context, skeleton_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        flow_context, flow_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        
        fusion_context, fusion_target = self.fusion1(skeleton_context, skeleton_target, flow_context, flow_target)

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        rgb_context = rgb_context.reshape(1, -1, self.args.trans_linear_in_dim)
        rgb_target = rgb_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_context = self.fusion2(rgb_context, fusion_context)
        fusion_target = self.fusion2(rgb_target, fusion_target)

        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

class ThreeFusionDGA2(nn.Module):
    def __init__(self, args):
        super(ThreeFusionDGA2, self).__init__()
        # print("当前使用的是双模态融合代码")
        print("当前使用的是三模态的ThreeFusionDGA2融合")
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion1 = TwoTransforFusion(args)
        self.fusion2 = DGAdaIN(self.args.trans_linear_in_dim, self.args.trans_linear_in_dim)
        self.mlp1= MLP_Mix_Enrich(self.args.trans_linear_in_dim, self.args.seq_len)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        rgb_context, rgb_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        skeleton_context, skeleton_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        flow_context, flow_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        
        fusion_context, fusion_target = self.fusion1(skeleton_context, skeleton_target, flow_context, flow_target)

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        rgb_context = rgb_context.reshape(1, -1, self.args.trans_linear_in_dim)
        rgb_target = rgb_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_context = self.fusion2(rgb_context, fusion_context)
        fusion_target = self.fusion2(rgb_target, fusion_target)
        fusion_context = fusion_context.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim)
        fusion_context = self.mlp1(fusion_context)
        fusion_target = self.mlp1(fusion_target)
        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)        
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

class ThreeFusion3(nn.Module):
    def __init__(self, args):
        super(ThreeFusion3, self).__init__()
        # print("当前使用的是双模态融合代码")
        print("当前使用的是ThreeFusion3的融合")
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.ids = range(args.num_gpus)
        self.positionEncoding1 = TrainablePositionalEncoding(args.seq_len, self.args.trans_linear_in_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.args.trans_linear_in_dim, nhead=1)
        self.tran = nn.TransformerEncoder(
            encoder_layer, num_layers=3)
        self.MLP = Bottleneck_Perceptron_2_layer(self.args.trans_linear_in_dim)

    def forward(self, context_features, context_labels, target_features):
        rgb_context, rgb_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        skeleton_context, skeleton_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        flow_context, flow_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        rgb_context = self.tran(rgb_context)
        rgb_target = self.tran(rgb_target)
        fusion_context, fusion_target = self.fusion(skeleton_context, skeleton_target, flow_context, flow_target)
        fusion_context += rgb_context
        fusion_target += rgb_target
        fusion_context = self.MLP(fusion_context)
        fusion_target = self.MLP(fusion_target)
        # fusion_context, fusion_target = self.fusion(
        #     rgb_context, rgb_target, flow_context, flow_target)
        # fusion_context, fusion_target = self.fusion(
        #     rgb_context, rgb_target, skeleton_context, skeleton_target)

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model()
            self.fusion.cuda(0)
            self.fusion = torch.nn.DataParallel(self.fusion, device_ids=[
                                                   i for i in range(0, self.args.num_gpus)])

class BatchTwoFusion(nn.Module):
    def __init__(self, in_channels=2048, out_channels=2048):
        super().__init__()
        self.in_channels = in_channels
        self.f1 = nn.Linear(in_channels, out_channels, bias=True)
        self.norm = torch.nn.BatchNorm1d(in_channels, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.eps = 1e-05
    
    def forward(self, x, w):
        x = x.reshape((-1, self.in_channels))
        w = w.reshape((-1, self.in_channels))
        return self.f1(x + (x - torch.mean(w)) / (torch.std(w)+self.eps))    

class ThreeTranToTwo(nn.Module):
    def __init__(self, args, out_channels=None, dropout=0.1):
        super().__init__()
        in_channels = 2048
        self.positionEncoding1 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding2 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        self.positionEncoding3 = TrainablePositionalEncoding(
            args.seq_len, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels * 3, nhead=3)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=4)
        self.f1 = nn.Linear(in_channels * 3, in_channels * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, y1, y2, z1, z2):
        x1 = self.positionEncoding1(x1)
        x2 = self.positionEncoding1(x2)
        y1 = self.positionEncoding2(y1)
        y2 = self.positionEncoding2(y2)
        z1 = self.positionEncoding3(z1)
        z2 = self.positionEncoding3(z2)
        xyz1 = torch.cat((torch.cat((x1, y1), dim=-1), z1), dim=-1)
        xyz2 = torch.cat((torch.cat((x2, y2), dim=-1), z2), dim=-1)
        fusion1 = self.transformer_encoder(xyz1)
        fusion2 = self.transformer_encoder(xyz2)
        return self.dropout(self.f1(fusion1)), self.dropout(self.f1(fusion2))

class ThreeFusionTwoRoad(nn.Module):
    def __init__(self, args):
        super(ThreeFusionTwoRoad, self).__init__()
        # print("当前使用的是双模态融合代码")
        print("当前使用的是三模态的ThreeFusionTwoRoad融合")
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = ThreeTranToTwo(args)
        self.ids = range(args.num_gpus)
        self.f1 = nn.Linear(self.args.trans_linear_in_dim//2, self.args.trans_linear_in_dim//2, bias=True)
        self.f2 = nn.Linear(self.args.trans_linear_in_dim//2, self.args.trans_linear_in_dim//2, bias=True)
        self.MLP1 = Bottleneck_Perceptron_2_layer(self.args.trans_linear_in_dim//2)
        self.MLP2 = Bottleneck_Perceptron_2_layer(self.args.trans_linear_in_dim//2)

    def forward(self, context_features, context_labels, target_features):
        rgb_context, rgb_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        skeleton_context, skeleton_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        flow_context, flow_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
       
        fusion_context, fusion_target = self.fusion(
            rgb_context, rgb_target, skeleton_context, skeleton_target, flow_context, flow_target)

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)

        pre = fusion_context[:,:, :self.args.trans_linear_in_dim//2]
        sec = fusion_context[:,:, self.args.trans_linear_in_dim//2:]
        x = self.f1(pre)
        y = self.f2(sec)
        x = self.MLP1(x)
        y = self.MLP2(y)
        c1 = x + y

        pre = fusion_target[:,:, :self.args.trans_linear_in_dim//2]
        sec = fusion_target[:,:, self.args.trans_linear_in_dim//2:]

        x = self.f1(pre)
        y = self.f2(sec)
        x = self.MLP1(x)
        y = self.MLP2(y)
        c2 = x + y

        fusion_logits = self.bracnch(
            c1, c2, context_labels)
        return fusion_logits

class TwoFusionBatchFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("当前使用的是两模态的BatchFusion")
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion2 = BatchTwoFusion(self.args.trans_linear_in_dim, self.args.trans_linear_in_dim)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        rgb_context, rgb_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        fusion_context = self.fusion2(rgb_context, second_context)
        fusion_target = self.fusion2(rgb_target, second_target)

        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

class S3D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.train()
        self.args = args
        self.encoder = S3DEncoder(args)
        self.num_patches = 16

        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList(
            [TemporalCrossTransformer(args, s) for s in args.temp_set])

    def forward(self, context_images, context_labels, target_images):
        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 25 x 64 x 17 x 3, target_images = 20 x 64 x 17 x 3 
        # '''
        context_images = context_images.reshape(context_images.shape[0], context_images.shape[1], -1).float()
        target_images = target_images.reshape(target_images.shape[0], target_images.shape[1], -1).float()
        context_features = self.encoder(context_images)  
        target_features = self.encoder(target_images)  

        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  
        target_features = target_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  

        all_logits_fr = [t(context_features, context_labels, target_features)[
            'logits'] for t in self.transformers]

        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1)

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1])  

        # return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
        #                'logits_post_pat': split_first_dim_linear(sample_logits_post_pat, [NUM_SAMPLES, target_features.shape[0]])}
        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]),
                       'logits_post_pat': torch.zeros((1))}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.encoder = torch.nn.DataParallel(
                self.encoder, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)

    def extract_feature(self, images):
        """
        获取一个视频的图片帧经过resnet后的特征.
        :param images: A batch of images
        :return: A batch of features
        """

        # images: batch_size x 3 x 224 x 224, 一般的，batch_size = 8
        context_features = self.encoder(images)  # batch_size x 2048 x 7 x 7

        return context_features

class S3DEncoder(nn.Module):
    """graph spatial-temporal transformer"""

    def __init__(self, args, t_input_size=17*3, 
                 num_head=1, num_layer=3, dropout=0.1
                ) -> None:
        super().__init__()
        self.args = args
        hidden_size = self.args.trans_linear_in_dim
        self.d_model  = hidden_size 
        # temproal transformer encoder
        self.t_embedding = nn.Sequential(
                nn.Linear(t_input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, hidden_size),
        ) 

        self.pe = PositionalEncoding(self.d_model, dropout=dropout)

        t_layer = nn.TransformerEncoderLayer(self.d_model , num_head, self.d_model) 
        self.t_tr = nn.TransformerEncoder(t_layer, num_layer)

        # self.t_tr = nn.LSTM(input_size=self.d_model,hidden_size=self.d_model//2,num_layers=num_layer,batch_first=True,bidirectional=True)

    def forward(self, src):
        src = self.t_embedding(src)

        t_out = self.t_tr(self.pe(src))
        return t_out

class TimeTransformer(nn.Module):
    def __init__(self, args, out_channels=None, dropout=0.1 ):
        super(TimeTransformer, self).__init__()
        self.train()
        self.args = args
        in_channels = 2048
        self.positionEncoding = TrainablePositionalEncoding(
        args.seq_len, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels , nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=2)
        self.f1 = nn.Linear(in_channels , in_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.positionEncoding(x)
        fusion = self.transformer_encoder(x)
        
        return fusion

class CrossTransformer(nn.Module):
    def __init__(self, args):
        super(CrossTransformer, self).__init__()
        self.args = args

        max_len = int(self.args.seq_len * 1.5)
        self.pe = PositionalEncoding(2048, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(2048,  self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(2048 , self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all ordered tuples corresponding to the temporal set size 2 or 3.
        frame_idxs = [i for i in range(self.args.seq_len)]
    
    def forward(self, support_set, support_labels, queries):
        # support_set : 25 x 8 x 2048, support_labels: 25, queries: 20 x 8 x 2048
        n_queries = queries.shape[0] #20
        n_support = support_set.shape[0] #25
        
        # static pe after adding the position embedding
        support_set = self.pe(support_set)# Support set is of shape 25 x 8 x 2048 -> 25 x 8 x 2048
        queries = self.pe(queries) # Queries is of shape 20 x 8 x 2048 -> 20 x 8 x 2048

        '''
            support_set_ks is of shape 25 x 28 x 1152, where 1152 is the dimension of the key = query head. converting the 5-way*5-shot x 28(tuples).
            query_set_ks is of shape 20 x 28 x 1152 covering 4 query/sample*5-way x 28(number of tuples)
        '''
        support_set_ks = self.k_linear(support_set) # 25 x 8 x 1152
        queries_ks = self.k_linear(queries) # 20 x 8 x 1152
        support_set_vs = self.v_linear(support_set) # 25 x 8 x 1152
        queries_vs = self.v_linear(queries) # 20 x 8 x 1152
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks) # 25 x 8 x 1152
        mh_queries_ks = self.norm_k(queries_ks) # 20 x 8 x 1152
        support_labels = support_labels
        mh_support_set_vs = support_set_vs # 25 x 8 x 1152
        mh_queries_vs = queries_vs # 20 x 8 x 1152
        
        unique_labels = torch.unique(support_labels) # 5

        # init tensor to hold distances between every support tuple and every target tuple. It is of shape 20  x 5
        '''
            4-queries * 5 classes x 5(5 classes) and store this in a logit vector
        '''
        all_distances_tensor = torch.zeros(n_queries, self.args.way) # 20 x 5

        for label_idx, c in enumerate(unique_labels):
            # select keys and values for just this class 
            class_k = torch.index_select(mh_support_set_ks, 0, self._extract_class_indices(support_labels, c)) # 5 x 8 x 1152
            class_v = torch.index_select(mh_support_set_vs, 0, self._extract_class_indices(support_labels, c)) # 5 x 8 x 1152
            k_bs = class_k.shape[0] # 5
            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim) # 20 x 5 x 8 x 8
            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3) # 20 x 8 x 5 x 8 
            
            # [For the 20 queries' 28 tuple pairs, find the best match against the 5 selected support samples from the same class
            class_scores = class_scores.reshape(n_queries, self.args.seq_len, -1) # 20 x 28 x 140
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)] # list(20) x 8 x 140
            class_scores = torch.cat(class_scores) # 560 x 140 - concatenate all the scores for the tuples
            class_scores = class_scores.reshape(n_queries, self.args.seq_len, -1, self.args.seq_len) # 20 x 8 x 5 x 8
            class_scores = class_scores.permute(0,2,1,3) # 20 x 5 x 8 x 8
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v) # 20 x 5 x 8 x 1152 
            query_prototype = torch.sum(query_prototype, dim=1) # 20 x 8 x 1152 -> Sum across all the support set values of the corres. class
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype # 20 x 8 x 1152
            norm_sq = torch.norm(diff, dim=[-2,-1])**2 # 20 
            distance = torch.div(norm_sq,self.args.seq_len) # 20
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance # 20
        
        return_dict = {'logits': all_distances_tensor}
        
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

class CTX(nn.Module):

    def __init__(self, args, ):
        super(CTX, self).__init__()
        self.train()
        self.args = args
        
        if self.args.mode=="KD_res18":
            self.args.trans_linear_in_dim=2048
        if self.args.mode=="no_grad_res18_train_trx":
            self.args.trans_linear_in_dim=2048
        if self.args.mode=="KD_KL_meta":
            self.args.trans_linear_in_dim=2048
            
        
        self.cos_distance=CosDistance(args)
        self.transformers = CrossTransformer(args)
        self.time_trans = TimeTransformer(args)
        self.num_patches = 16
        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))

        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)
        
        last_layer_idx = -2  #最后两层
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])#去掉最后两层

                   
    def forward(self, context_feature, context_labels, target_feature,mode):
            
        context_features = self.resnet(context_feature) # 200 x 512 x 7 x 7
        target_features = self.resnet(target_feature) # 160 x 512 x 7 x 7

            # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(context_features) # 200 x 512 x 4 x 4
        target_features = self.adap_max(target_features) # 160 x 512 x 4 x 4
            
            # Reshape before averaging across all the patches
        context_features = context_features.reshape(-1, 512, self.num_patches) # 200 x 512 x 16
        target_features = target_features.reshape(-1, 512, self.num_patches) # 160 x 512 x 16     

            # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1) # 200 x 16 x 512
        target_features = target_features.permute(0, 2, 1) # 160 x 16 x 512

            # Average across the patches 
        context_features = torch.mean(context_features, dim = 1) # 200 x 512
        target_features = torch.mean(target_features, dim = 1) # 160 x 512
            #聚合特征
            # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 512
        target_features = target_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 20 x 8 x 512
        # print(context_features.shape)
        
        context_features =self.time_trans(context_features) # 25 x 8 x 2048
        target_features = self.time_trans(target_features)
        # print(context_features.shape)
        
        all_logits = [self.transformers(context_features,context_labels,target_features)['logits']]
        all_logits = torch.stack(all_logits, dim=-1)
        # print(all_logits.shape)

        sample_logits = all_logits

        sample_logits = torch.mean(sample_logits, dim=[-1]) # 20 x 5
        # print(sample_logits.shape)
        # print([NUM_SAMPLES, target_feature.shape[0]])
        return_dict = {'logits': split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_features.shape[0]])}
        
        return return_dict

    
    def train_test_trx(self,context_feature, context_labels, target_feature):
        
        all_logits = [t(context_feature, context_labels, target_feature)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)

        sample_logits = all_logits

        sample_logits = torch.mean(sample_logits, dim=[-1]) # 20 x 5


        return_dict = {'logits': split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_feature.shape[0]])}
        return return_dict
    
    def test_cos(self,context_feature, context_labels, target_feature):
        
        all_logits =self.cos_distance(context_feature,context_labels,target_feature)
        sample_logits = all_logits

        return_dict = {'logits': split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_feature.shape[0]])}
        return return_dict
    
    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:

            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])
            self.transformers.cuda(0)

class CTXBranch(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """
    def __init__(self, args, ids=0):
        super().__init__()

        self.train()
        self.args = args
        self.num_patches = 16
        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = CrossTransformer(args)

    def forward(self, context_features, target_features, context_labels):
        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 25 x 8 x 2048
        target_features = target_features.reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim)  # 20 x 8 x 2048

        # Frame-level logits
        all_logits_fr = [self.transformers(context_features, context_labels, target_features)[
            'logits']]
        # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1)

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1])  # 20 x 5

        return_dict = {'logits': split_first_dim_linear(
            sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]])}

        return return_dict

    def distribute_model(self):
        self.transformers = self.transformers.cuda(0)

class TwoCTXShuffleTime(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = CTXBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        
        fusion_context1, fusion_target1 = self.fusion(
            first_context, first_target, second_context, second_target,)

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)
        
        second_target_shuffle = second_target[:, :self.args.shirt_num, :]
        second_target_prefix = second_target[:, self.args.shirt_num:, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)


        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle,
        )

        fusion_context = fusion_context1 + fusion_context2
        fusion_target = fusion_target1 + fusion_target2
        del fusion_context1, fusion_context2, fusion_target1, fusion_target2

        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_logits = self.bracnch(
            fusion_context, fusion_target, context_labels)
        return fusion_logits

class CNN_STRM(nn.Module):
    """
        Standard Video Backbone connected to a Temporal Cross Transformer, Query Distance 
        Similarity Loss and Patch-level and Frame-level Attention Blocks.
    """

    def __init__(self, args):
        super(CNN_STRM, self).__init__()

        self.train()
        self.args = args

        # Using ResNet Backbone
        if self.args.method == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.args.method == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.args.method == "resnet50":
            resnet = models.resnet50(pretrained=True)

        last_layer_idx = -2
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.num_patches = 16

        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))

        # Temporal Cross Transformer for modelling temporal relations
        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 

        # New-distance metric for post patch-level enriched features
        self.new_dist_loss_post_pat = [DistanceLoss(args, s) for s in args.temp_set]

        # Linear-based patch-level attention over the 16 patches
        self.attn_pat = Self_Attn_Bot(self.args.trans_linear_in_dim, self.num_patches)

        # MLP-mixing frame-level enrichment over the 8 frames.
        self.fr_enrich = MLP_Mix_Enrich(self.args.trans_linear_in_dim, self.args.seq_len)

    def forward(self, context_images, context_labels, target_images):

        '''
            context_features/target_features is of shape (num_images x 2048) [final Resnet FC layer] after squeezing
        '''
        '''
            context_images: 200 x 3 x 224 x 224, target_images = 160 x 3 x 224 x 224
        '''
        context_features = self.resnet(context_images) # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_images) # 160 x 2048 x 7 x 7

        # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(context_features) # 200 x 2048 x 4 x 4
        target_features = self.adap_max(target_features) # 160 x 2048 x 4 x 4

        # Reshape before averaging across all the patches
        context_features = context_features.reshape(-1, self.args.trans_linear_in_dim, self.num_patches) # 200 x 2048 x 16
        target_features = target_features.reshape(-1, self.args.trans_linear_in_dim, self.num_patches) # 160 x 2048 x 16       

        # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1) # 200 x 16 x 2048
        target_features = target_features.permute(0, 2, 1) # 160 x 16 x 2048

        # Performing self-attention across the 16 patches
        context_features = self.attn_pat(context_features) # 200 x 16 x 2048 
        target_features = self.attn_pat(target_features) # 160 x 16 x 2048

        # Average across the patches 
        context_features = torch.mean(context_features, dim = 1) # 200 x 2048
        target_features = torch.mean(target_features, dim = 1) # 160 x 2048

        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        target_features = target_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 20 x 8 x 2048

        # Compute logits using the new loss before applying frame-level attention
        all_logits_post_pat = [n(context_features, context_labels, target_features)['logits'] for n in self.new_dist_loss_post_pat]
        all_logits_post_pat = torch.stack(all_logits_post_pat, dim=-1) # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]

        # Combing the patch and frame-level logits
        sample_logits_post_pat = all_logits_post_pat
        sample_logits_post_pat = torch.mean(sample_logits_post_pat, dim=[-1]) # 20 x 5

        # Perform self-attention across the 8 frames
        context_features_fr = self.fr_enrich(context_features) # 25 x 8 x 2048
        target_features_fr = self.fr_enrich(target_features) # 20 x 8 x 2048

        '''
            For different temporal lengths(2, 3, ...) get the final logits and perform mean.
        '''

        # Frame-level logits
        all_logits_fr = [t(context_features_fr, context_labels, target_features_fr)['logits'] for t in self.transformers]
        all_logits_fr = torch.stack(all_logits_fr, dim=-1) # 20 x 5 x 1[number of timesteps] 20 - 5 x 4[5-way x 4 queries/class]

        sample_logits_fr = all_logits_fr
        sample_logits_fr = torch.mean(sample_logits_fr, dim=[-1]) # 20 x 5

        return_dict = {'logits': split_first_dim_linear(sample_logits_fr, [NUM_SAMPLES, target_features.shape[0]]), 
                    'logits_post_pat': split_first_dim_linear(sample_logits_post_pat, [NUM_SAMPLES, target_features.shape[0]])}

        return return_dict

    def extract_feature(self, image):

        context_features = self.resnet(image) 
        # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(context_features)

        context_features = context_features.reshape(-1, self.args.trans_linear_in_dim, self.num_patches)  

        # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1) 

        # Performing self-attention across the 16 patches
        context_features = self.attn_pat(context_features) 

        # Average across the patches 
        context_features = torch.mean(context_features, dim = 1) 
        return context_features

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)
            self.new_dist_loss_post_pat = [n.cuda(0) for n in self.new_dist_loss_post_pat]

            self.attn_pat.cuda(0)
            self.attn_pat = torch.nn.DataParallel(self.attn_pat, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.fr_enrich.cuda(0)
            self.fr_enrich = torch.nn.DataParallel(self.fr_enrich, device_ids=[i for i in range(0, self.args.num_gpus)])

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists

def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)

    cum_dists = torch.zeros(dists.shape, device=dists.device)
    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 

    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    # print(cum_dists)
    return cum_dists[:,:,-1,-1]

def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

class CNN_OTAM(nn.Module):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self):
        super(CNN_OTAM, self).__init__()

    def forward(self, support_features, support_labels, target_features):
        unique_labels = torch.unique(support_labels)
        # print(support_features)
        if torch.isnan(support_features).any():
            # print("出现nan的feature")
            return {'logits':  torch.zeros((target_features.shape[0], 5), device=target_features.device).requires_grad_(True)}

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        support_features = rearrange(support_features, 'b s d -> (b s) d')
        target_features = rearrange(target_features, 'b s d -> (b s) d')

        frame_sim = cos_sim(target_features, support_features)
        frame_dists = 1 - frame_sim

        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)

        # calculate query -> support and support -> query
        cum_dists = OTAM_cum_dist(dists) + OTAM_cum_dist(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return {'logits':  F.softmax(-class_dists)}

class Action_Recognition_Resnet50(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]     # delete the last fc layer.
        self.convnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, self.args.num_classes)

    def forward(self, x):
        b, t = x.shape[:2]
        x = x.reshape((b*t,) + x.shape[2:])
        feature = self.convnet(x).squeeze()
        feature = feature.view(b, t, -1)
        feature = feature.mean(dim=1)
        return self.fc(feature)
    
    def extract_feature(self, x):
        t = x.shape[0]
        feature = self.convnet(x).squeeze()
        feature = feature.view(t, -1)
        return feature
    
    def distribute_model(self):
        if self.args.num_gpus > 1:
            self.convnet = self.convnet.cuda(0)
            self.fc = self.fc.cuda(0)

class Baseline(nn.Module):
    '''
        The Baseline model for few-shot action recognition, using ResNet50 as the feature extractor.
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.trans_linear_in_dim=2048
        self.resnet = models.resnet50(pretrained=True)  
        self.resnet.fc=nn.Identity()
        self.resnet.final_feat_dim=2048

        
    def forward(self, context_images, context_labels, target_images):
        '''
            Input:
                Context_images: A support set including 25 videos (200 images). The shape: 200, 3, 224, 224.
                Context_labels: the label of 20 query videos. The shape: 20, 1.
                target_images: A query set including 20 videos (160 images). The shape: 160, 3, 224, 224.
            
            Return:
                dic: A dictionary containing the logits of the query set.
        '''

        context_features = self.resnet(context_images) # 200 x 2048
        target_features = self.resnet(target_images) # 160 x 2048 
        supports = context_features.reshape(-1, 8, 2048)    # 25*8*2048
        queries = target_features.reshape(-1, 8, 2048)      # 20*8*2048
        
        n_queries = queries.shape[0] #4
        queries = queries.mean(dim=1)  #前两个维度聚集  4*2048
        
        support_labels = context_labels.to(supports.device)
        unique_labels = torch.unique(support_labels) # 5  

        dist_all = torch.zeros(n_queries, self.args.way) # 5 x 5

        for label_idx, c in enumerate(unique_labels):
            # Select keys corresponding to this class from the support set tuples
            class_k = torch.index_select(supports, 0, self._extract_class_indices(support_labels, c)) # 5-8-2048    

            support_set_c = class_k.mean(dim=1) #5-2048

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

    def extract_feature(self, x):
        t = x.shape[0]
        feature = self.resnet(x).squeeze()
        feature = feature.view(t, -1)
        return feature

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            # self.resnet.cuda(0)
            self.resnet = torch.nn.DataParallel(
                self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])


class ThreeTRXCombination(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bracnch = TrxBranch(args, 0 % args.num_gpus)
        self.fusion = TwoTransforFusion(args)
        self.three_fusion = ThreeTransforTemproal(args)
        self.ids = range(args.num_gpus)

    def forward(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        fusion_context1 = self.three_fusion(first_context, second_context, third_context)
        fusion_target1 = self.three_fusion(first_target, second_target, third_target)

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context, second_target,
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context, third_target
        )

        fusion_context = fusion_context1 + fusion_context2 + fusion_context3
        fusion_target = fusion_target1 + fusion_target2 + fusion_target3
        del fusion_context1, fusion_context2, fusion_target1, fusion_target2, fusion_context3, fusion_target3

        return self.feature_test(fusion_context, context_labels, fusion_target)

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.bracnch.distribute_model(self.args)
            self.fusion.distribute_model(self.args)
            self.three_fusion.distribute_model(self.args)

    def extract_feature(self, feature):
        rgb = feature['rgb'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim).cuda()
        depth = feature['depth'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim).cuda()
        flow = feature['flow'].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim).cuda()
        feature1 = self.three_fusion.extract_feature(rgb, depth, flow)
        
        feature2 = depth[:, self.args.shirt_num:, :]
        feature2_prefix = depth[:, :self.args.shirt_num, :]
        feature2 = torch.cat((feature2, feature2_prefix), dim=1)
        feature2 = self.fusion.extract_feature(rgb, feature2)
        
        feature3 = flow[:, self.args.shirt_num:, :]
        feature3_prefix = flow[:, :self.args.shirt_num, :]
        feature3 = torch.cat((feature3, feature3_prefix), dim=1)
        feature3 = self.fusion.extract_feature(rgb, feature3)
        
        return feature1 + feature2 + feature3

    def extract_task_feature(self, context_features, context_labels, target_features):
        first_context, first_target, context_labels = context_features[self.args.m1].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m1].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        second_context, second_target, context_labels = context_features[self.args.m2].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m2].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels
        third_context, third_target, context_labels = context_features[self.args.m3].reshape(
            -1, self.args.seq_len, self.args.trans_linear_in_dim), target_features[self.args.m3].reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim), context_labels

        second_context_shuffle = second_context[:, self.args.shirt_num:, :]
        second_context_prefix = second_context[:, :self.args.shirt_num, :]
        second_context_shuffle = torch.cat((second_context_shuffle, second_context_prefix), dim=1)
        
        second_target_shuffle = second_target[:, self.args.shirt_num:, :]
        second_target_prefix = second_target[:, :self.args.shirt_num, :]
        second_target_shuffle = torch.cat((second_target_shuffle, second_target_prefix), dim=1)

        third_context_shuffle = third_context[:, self.args.shirt_num:, :]
        third_context_suffix = third_context[:, :self.args.shirt_num, :]
        third_context_shuffle = torch.cat((third_context_suffix, third_context_shuffle), dim=1)

        third_target_shuffle = third_target[:, self.args.shirt_num:, :]
        third_target_suffix = third_target[:, :self.args.shirt_num, :]
        third_target_shuffle = torch.cat((third_target_suffix, third_target_shuffle), dim=1)

        fusion_context1, fusion_target1 = self.three_fusion(
            first_context, first_target, second_context, second_target, third_context, third_target)

        fusion_context2, fusion_target2 = self.fusion(
            first_context, first_target, second_context_shuffle, second_target_shuffle,
        )

        fusion_context3, fusion_target3 = self.fusion(
            first_context, first_target, third_context_shuffle, third_target_shuffle
        )

        fusion_context = fusion_context1 + fusion_context2 + fusion_context3
        fusion_target = fusion_target1 + fusion_target2 + fusion_target3
        del fusion_context1, fusion_context2, fusion_target1, fusion_target2, fusion_context3, fusion_target3
        return fusion_context, fusion_target
 
    def feature_test(self, fusion_context, context_labels, fusion_target):
        fusion_context = fusion_context.reshape(1, -1, self.args.trans_linear_in_dim)
        fusion_target = fusion_target.reshape(1, -1, self.args.trans_linear_in_dim)
        return self.bracnch(fusion_context, fusion_target, context_labels)

if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 1
            self.query_per_class = 5
            self.trans_dropout = 0.1
            self.seq_len = 8
            self.img_size = 84
            self.method = "resnet18"
            self.num_gpus = 1
            self.temp_set = [2, 3]
    args = ArgsObject()
    torch.manual_seed(STRM(args))

    support_imgs = torch.rand(
        args.way * args.shot * args.seq_len, 3, args.img_size, args.img_size)
    target_imgs = torch.rand(
        args.way * args.query_per_class * args.seq_len, 3, args.img_size, args.img_size)
    support_labels = torch.tensor([0, 1, 2, 3, 4])

    out = model(support_imgs, support_labels, target_imgs)

    print("STRM returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(
        out['logits'].shape))

