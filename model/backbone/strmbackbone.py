
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import math

from itertools import combinations 

from torch.autograd import Variable

NUM_SAMPLES=1
np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)

#----------------------------------------------------------------------------------------------------

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
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape 1 x 5000 x 2048
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)

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
        output = self.inp_fc(x) # B x 2048 x 8

        # Apply the relu non-linearity
        output = self.relu(output) # B x 2048 x 8

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
        
        return output + x # Residual output

class Self_Attn_Bot(nn.Module):
    """ Self attention Layer
        Attention-based frame enrichment
    """
    def __init__(self,in_dim, seq_len):
        super(Self_Attn_Bot,self).__init__()
        self.chanel_in = in_dim # 2048
        
        # Using Linear projections for Key, Query and Value vectors
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.value_conv = nn.Linear(in_dim, in_dim)

        self.softmax  = nn.Softmax(dim=-1) #
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
        x = self.pe(x) # B x 16 x 2048

        m_batchsize,C,width = x.size() # m = 200/160, C = 2048, width = 16

        # Save residual for later use
        residual = x # B x 16 x 2048

        # Perform query projection
        proj_query  = self.query_proj(x) # B x 16 x 2048

        # Perform Key projection
        proj_key = self.key_proj(x).permute(0, 2, 1) # B x 2048  x 16

        energy = torch.bmm(proj_query,proj_key) # transpose check B x 16 x 16
        attention = self.softmax(energy) #  B x 16 x 16

        # Get the entire value in 2048 dimension 
        proj_value = self.value_conv(x).permute(0, 2, 1) # B x 2048 x 16

        # Element-wise multiplication of projected value and attention: shape is x B x C x N: 1 x 2048 x 8
        out = torch.bmm(proj_value,attention.permute(0,2,1)) # B x 2048 x 16

        # Reshaping before passing through MLP
        out = out.permute(0, 2, 1) # B x 16 x 2048

        # Passing via gamma attention
        out = self.gamma*out + residual # B x 16 x 2048

        # Pass it via a 3-layer Bottleneck MLP with Residual Layer defined within MLP
        out = self.Bot_MLP(out)  # B x 16 x 2048

        return out

class MLP_Mix_Enrich(nn.Module):
    """ 
        Pure Token-Bottleneck MLP-based enriching features mechanism
    """
    def __init__(self,in_dim, seq_len):
        super(MLP_Mix_Enrich,self).__init__()
        # in_dim = 2048
        self.Tok_MLP = Token_Perceptron(seq_len) # seq_len = 8 frames
        self.Bot_MLP = Bottleneck_Perceptron_2_layer(in_dim)

        max_len = int(seq_len * 1.5) # seq_len = 8
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W ) # B(25/20) x 8 x 2048
            returns :
                out : self MLP-enriched value + input feature 
        """

        # Add a position embedding to the 8 frames
        x = self.pe(x) # B x 8 x 2048

        # Store the residual for use later
        residual1 = x # B x 8 x 2048

        # Pass it via a 2-layer Token MLP followed by Residual Layer
        # Permuted before passing into the MLP: B x 2048 x 8 
        out = self.Tok_MLP(x.permute(0, 2, 1)).permute(0, 2, 1) + residual1 # B x 8 x 2048

        # Storing a residual 
        residual2 = out # B x 8 x 2048
        
        # Pass it via 2-layer Bottleneck MLP defined on Channel(2048) features
        out = self.Bot_MLP(out) + residual2 # B x 8 x 2048

        return out

#-----------------------------------------------------------------------------------------------------


class strmbackbone(nn.Module):
    
    '''
    resnet18的backbone
    输入是dataload出来的一个task的视频（包括support_set，和query）
    输出是对应的特征  维度为 S：25*8*2048 Q：20*8*2048
    '''
    def __init__(self, args, ):
        super(strmbackbone, self).__init__()
        # self.train()
        self.args = args
        self.args.trans_linear_in_dim=2048
        self.num_patches = 16
        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))
        resnet = models.resnet18(pretrained=True)  
        
        # Linear-based patch-level attention over the 16 patches
        self.attn_pat = Self_Attn_Bot(512, self.num_patches)
        # MLP-mixing frame-level enrichment over the 8 frames.
        self.fr_enrich = MLP_Mix_Enrich(self.args.trans_linear_in_dim, self.args.seq_len)

        last_layer_idx = -2  #最后两层
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])#去掉最后两层
        self.res18_2048 = nn.Linear(512, 2048)
        
        self.fc1 = nn.Linear(512, 2048) #res18最后输出变为2048
        self.fc2 = nn.Linear(512, 2048) #res18最后输出变为2048
                   
    def forward(self, context_feature,context_labels, target_feature):
            
        context_features = self.resnet(context_feature) # 200 x 512 x 7 x 7
        target_features = self.resnet(target_feature) # 160 x 2048 x 7 x 7
            # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(context_features) # 200 x 512 x 4 x4
        target_features = self.adap_max(target_features)    
            # Reshape before averaging across all the patches
        context_features = context_features.reshape(-1, 512, self.num_patches) # 200 x 512 x 16  
        target_features = target_features.reshape(-1, 512,self.num_patches)   
            # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1) # 200 x 16 x 512
        target_features = target_features.permute(0, 2, 1)
            # Average across the patches 
            
        context_features = self.attn_pat(context_features) # 200 x 16 x 2048 
        target_features = self.attn_pat(target_features) # 160 x 16 x 2048
        
        context_features = torch.mean(context_features, dim = 1) # 200 x 512 
        target_features = torch.mean(target_features, dim = 1)
        
        context_features=self.res18_2048(context_features)
        target_features=self.res18_2048(target_features)
        
        # context_features_pat = context_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        # target_features_pat = target_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        
        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        target_features = target_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        
        context_features_pat = context_features
        target_features_pat = target_features
        
        context_features_fr = self.fr_enrich(context_features) # 25 x 8 x 2048
        target_features_fr = self.fr_enrich(target_features) # 20 x 8 x 2048
        
        # context_features_fr = context_features.reshape(-1, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        # target_features_fr = target_features.reshape(-1, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        
        # context_features_1 = self.fc1(context_features_fr) # 25 x 8 x 2048
        # target_features_1 = self.fc1(target_features_fr) # 20 x 8 x 2048
        
        # context_features_1 = context_features_1.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        # target_features_1 = target_features_1.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        
        
        # context_features_2 = self.fc2(context_features_fr) # 25 x 8 x 2048
        # target_features_2 = self.fc2(target_features_fr) # 20 x 8 x 2048
        
        # context_features_2 = context_features_2.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        # target_features_2 = target_features_2.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        
        context_features_dict={
            'distance':context_features_pat,
            'trx':context_features_fr
        }
        
        target_features_dict={
            'distance':target_features_pat,
            'trx':target_features_fr
        }
        
        return context_features_dict,target_features_dict
    
    
    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=list(range(self.args.num_gpus)))


 