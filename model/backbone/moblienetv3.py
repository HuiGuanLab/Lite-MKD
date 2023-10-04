
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models


NUM_SAMPLES=1

np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)



class mobile_large_2fc(nn.Module):
    '''
    resnet18的backbone
    输入是dataload出来的一个task的视频（包括support_set，和query）
    输出是对应的特征  维度为 S：25*8*2048 Q：20*8*2048
    '''
    def __init__(self, args, ):
        super(mobile_large_2fc, self).__init__()
        # self.train()
        self.args = args
        self.args.trans_linear_in_dim=2048
        self.num_patches = 16
        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))
        mobile = models.mobilenet_v3_large(pretrained=True)

        last_layer_idx = -2  #最后两层
        self.mobile = nn.Sequential(*list(mobile.children())[:last_layer_idx])#去掉最后两层
        # print(self.mobile)
        self.fc1 = nn.Linear(960, 2048) #res18最后输出变为2048
        self.fc2 = nn.Linear(960, 2048) #res18最后输出变为2048
                   
    def forward(self, context_feature,context_labels, target_feature):

        context_features = self.mobile(context_feature) # 200 x 512 x 7 x 7
        target_features = self.mobile(target_feature) # 160 x 2048 x 7 x 7
            # Decrease to 4 x 4 = 16 patches
        context_features = self.adap_max(context_features) # 200 x 512 x 4 x
        target_features = self.adap_max(target_features)    
            # Reshape before averaging across all the patches
        context_features = context_features.reshape(-1, 960, self.num_patches) # 200 x 512 x 16  
        target_features = target_features.reshape(-1, 960,self.num_patches)   
            # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1) # 200 x 16 x 512
        target_features = target_features.permute(0, 2, 1)
            # Average across the patches 
        context_features = torch.mean(context_features, dim = 1) # 200 x 512 
        target_features = torch.mean(target_features, dim = 1)
            #聚合特征
        context_features_1=self.fc1(context_features)
        target_features_1=self.fc1(target_features)
            #升维
        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features_1 = context_features_1.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        target_features_1 = target_features_1.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        
        context_features_2=self.fc2(context_features)
        target_features_2=self.fc2(target_features)
        
        context_features_2 = context_features_2.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        target_features_2 = target_features_2.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        
        context_feature_dict={
            'context_features_1':context_features_1,
            'context_features_2':context_features_2
        }
        target_features_dict={
            'target_features_1':target_features_1,
            'target_features_2':target_features_2
        }
        return context_feature_dict,target_features_dict
    
    
    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=list(range(self.args.num_gpus)))


class mobile_large(nn.Module):
    '''
    resnet18的backbone
    输入是dataload出来的一个task的视频（包括support_set，和query）
    输出是对应的特征  维度为 S：25*8*2048 Q：20*8*2048
    '''
    def __init__(self, args, ):
        super(mobile_large, self).__init__()
        # self.train()
        self.args = args
        self.args.trans_linear_in_dim=2048
        self.num_patches = 16
        self.adap_max = nn.AdaptiveMaxPool2d((4, 4))
        mobile = models.mobilenet_v3_large(pretrained=True)

        last_layer_idx = -2  #最后两层
        self.mobile = nn.Sequential(*list(mobile.children())[:last_layer_idx])#去掉最后两层
        # print(self.mobile)
        self.fc = nn.Linear(960, 2048) #res18最后输出变为2048
                   
    def forward(self, context_feature,context_labels, target_feature):

        context_features = self.mobile(context_feature) # 200 x 512 x 7 x 7
        target_features = self.mobile(target_feature) # 160 x 2048 x 7 x 7

        context_features = self.adap_max(context_features) # 200 x 512 x 4 x
        target_features = self.adap_max(target_features)    
            # Reshape before averaging across all the patches
        # print(context_features.shape)
        context_features = context_features.reshape(-1, 960, self.num_patches) # 200 x 512 x 16  
        target_features = target_features.reshape(-1, 960,self.num_patches)   
            # Permute before passing to the self-attention layer
        context_features = context_features.permute(0, 2, 1) # 200 x 16 x 512
        target_features = target_features.permute(0, 2, 1)
            # Average across the patches 
        context_features = torch.mean(context_features, dim = 1) # 200 x 512 
        target_features = torch.mean(target_features, dim = 1)
            #聚合特征
        context_features=self.fc(context_features)
        target_features=self.fc(target_features)
            #升维
        # Reshaping before passing to the Cross-Transformer and computing the distance after patch-enrichment as well
        context_features = context_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        target_features = target_features.reshape(-1, self.args.seq_len, self.args.trans_linear_in_dim) # 25 x 8 x 2048
        # print(55)
        # print(target_features.shape)

        return context_features,target_features
    
    
    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=list(range(self.args.num_gpus)))


 