import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models


class resnet50_stduent(nn.Module):

    def __init__(self, args, ):
        super(resnet50_stduent, self).__init__()
        print('resnet50')
        self.train()
        self.args = args
        
        self.args.trans_linear_in_dim=2048
        self.num_patches = 16
        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))
        resnet = models.resnet50(pretrained=True)
        
        last_layer_idx = -2  #最后两层
        self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])#去掉最后两层
        
        # self.res_2048 = nn.Linear(2048, 2048) #res18最后输出变为2048
                   
    def forward(self, context_feature, context_labels, target_feature):
        context_features = self.resnet(context_feature) # 200 x 2048 x 7 x 7
        target_features = self.resnet(target_feature) # 160 x 2048 x 7 x 7
        # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(context_features) # 200 x 2048 x 4 x 4
        target_features = self.adap_max(target_features) # 160 x 2048 x 4 x 4
            
        # Reshape before averaging across all the patches
        context_features = context_features.reshape(-1, 2048, self.num_patches) # 200 x 2048 x 16
        target_features = target_features.reshape(-1, 2048, self.num_patches) # 160 x 2048 x 16     

        # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1) # 200 x 16 x 2048
        target_features = target_features.permute(0, 2, 1) # 160 x 16 x 2048

        # Average across the patches 
        context_features = torch.mean(context_features, dim = 1) # 200 x 2048
        target_features = torch.mean(target_features, dim = 1) # 160 x 2048
        
        #可加线性层，因为50最后一层是relu，
        
        
            #升维
        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        target_features = target_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 20 x 8 x 2048
        
        return context_features,target_features
    
    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])
            print("多卡")