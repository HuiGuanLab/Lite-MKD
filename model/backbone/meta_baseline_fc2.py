
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models


NUM_SAMPLES=1

np.random.seed(3483)
torch.manual_seed(3483)
torch.cuda.manual_seed(3483)
torch.cuda.manual_seed_all(3483)


    # new_model.fc = nn.Identity()  #去掉最后一层
    # new_model.final_feat_dim = 2048
    # return new_model


class meta_baseline_fc2(nn.Module):
    '''
    这里是直接使用resnet50的
    trx有改进
    '''
    def __init__(self, args, ):
        super(meta_baseline_fc2, self).__init__()
        # self.train()
        self.args = args
        self.args.trans_linear_in_dim=2048
        self.resnet = models.resnet50(pretrained=True)  
        self.resnet.fc=nn.Identity()
        self.resnet.final_feat_dim=2048
        self.fc1 = nn.Linear(2048, 2048) #res18最后输出变为2048
        self.fc2 = nn.Linear(2048, 2048) #res18最后输出变为2048
        
                   
    def forward(self, context_feature,context_labels, target_feature):
            
        context_features = self.resnet(context_feature) # 200 x 2048
        target_features = self.resnet(target_feature) # 160 x 2048 
        
        context_features_1 = self.fc1(context_features)
        target_features_1 = self.fc1(target_features)
        
        context_features_2 = self.fc2(context_features)
        target_features_2 = self.fc2(target_features)
        
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


 