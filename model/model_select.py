from model.backbone.resnet18_student import resnet18_student
from model.backbone.resnet18_2fc import resnet18_2fc
from model.backbone.resnet50_2fc import resnet50_2fc
from model.backbone.strm18_student import strm18_student
from model.backbone.strmbackbone import strmbackbone
from model.backbone.meta_baseline import meta_baseline
from model.backbone.meta_baseline_fc2 import meta_baseline_fc2
from model.backbone.resnet50_student import resnet50_stduent
from model.backbone.moblienetv3 import mobile_large_2fc,mobile_large

from torch.nn.parameter import Parameter
import model.classifiers as classifiers
import torch
import torch.nn as nn


class Student(nn.Module):

    def __init__(self, args, ):
        super(Student, self).__init__()
        self.train()
        self.args=args
        
        self.backbone,self.classifier=select_model_student(args)
            
    def forward(self, context_feature, context_labels, target_feature):
        context_features,target_features=self.backbone(context_feature, context_labels, target_feature)
        # print(context_features.shape)
        logits=self.classifier(context_features,context_labels,target_features)['logits']
        model_dict={
            'logits':logits,
            'context_features':context_features,
            'target_features':target_features
        }
        
        return model_dict
      
class Teacher(nn.Module):
    def __init__(self, args, ):
        super(Teacher, self).__init__()
        self.train()
        self.args=args
        self.classifier=select_model_teacher(args)
        
    def forward(self, context_feature, context_labels, target_feature):
        
        model_dict=self.classifier(context_feature,context_labels,target_feature)
        
        return model_dict   
    
    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.classifier = torch.nn.DataParallel(self.classifier, device_ids=[i for i in range(0, self.args.num_gpus)]) 
        
class Extracter(nn.Module):
    def __init__(self, args ):
        super(Extracter, self).__init__()
        print(222)
        self.train()
        self.backbone=resnet18_extract(args)
        
        
    def forward(self, context_feature):

        return self.backbone(context_feature)
    
    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        
        if self.args.num_gpus > 1:
            self.Extracter = torch.nn.DataParallel(self.Extracter, device_ids=[i for i in range(0, self.args.num_gpus)])
     
    
def load_teacher(teacher,args):
    # checkpoint_teacher_trx=torch.load(args.teacher_checkpoint)
    
    # print(torch.load(args.teacher_checkpoint)['model_state_dict'].keys())
    
    
    # teacher.transformers.pe.pe=torch.load(args.teacher_checkpoint)['model_state_dict']['transformers.pe.pe']
    
    # teacher.transformers.k_linear.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['transformers.k_linear.weight'])
    # teacher.transformers.k_linear.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['transformers.k_linear.bias'])
    
    # teacher.transformers.v_linear.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['transformers.v_linear.weight'])
    # teacher.transformers.v_linear.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['transformers.v_linear.bias'])
    
    # teacher.transformers.norm_k.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['transformers.norm_k.weight'])
    # teacher.transformers.norm_k.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['transformers.norm_k.bias'])
    
    # teacher.transformers.norm_v.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['transformers.norm_v.weight'])
    # teacher.transformers.norm_v.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['transformers.norm_v.bias'])
    
    
    
   #----------------------------------------------------师兄那的老师————————————————————————————————————————————-___________
    
    teacher.transformers.pe.pe=torch.load(args.teacher_checkpoint)['model_state_dict']['bracnch.transformers.0.pe.pe']
    
    teacher.transformers.k_linear.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['bracnch.transformers.0.k_linear.weight'])
    teacher.transformers.k_linear.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['bracnch.transformers.0.k_linear.bias'])
    
    teacher.transformers.v_linear.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['bracnch.transformers.0.v_linear.weight'])
    teacher.transformers.v_linear.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['bracnch.transformers.0.v_linear.bias'])
    
    teacher.transformers.norm_k.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['bracnch.transformers.0.norm_k.weight'])
    teacher.transformers.norm_k.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['bracnch.transformers.0.norm_k.bias'])
    
    teacher.transformers.norm_v.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['bracnch.transformers.0.norm_v.weight'])
    teacher.transformers.norm_v.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['bracnch.transformers.0.norm_v.bias'])
    
    
    
    # teacher.transformers.pe.pe=torch.load(args.teacher_checkpoint)['model_state_dict']['classifier.transformers.pe.pe']
    
    # teacher.transformers.k_linear.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['classifier.transformers.k_linear.weight'])
    # teacher.transformers.k_linear.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['classifier.transformers.k_linear.bias'])
    
    # teacher.transformers.v_linear.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['classifier.transformers.v_linear.weight'])
    # teacher.transformers.v_linear.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['classifier.transformers.v_linear.bias'])
    
    # teacher.transformers.norm_k.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['classifier.transformers.norm_k.weight'])
    # teacher.transformers.norm_k.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['classifier.transformers.norm_k.bias'])
    
    # teacher.transformers.norm_v.weight=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['classifier.transformers.norm_v.weight'])
    # teacher.transformers.norm_v.bias=Parameter(torch.load(args.teacher_checkpoint)['model_state_dict']['classifier.transformers.norm_v.bias'])
    
    # teacher.load_state_dict(checkpoint_teacher_trx['model_state_dict'])
    return teacher
     
def load_student(args):
    
    student=Student(args)
    checkpoint_student=torch.load(args.test_model_path)

    for key in list(checkpoint_student['model_state_dict'].keys()):
        print(key)
        newkey=key.split('.')
        if newkey[2]=='module':
            newkey = key[:15] + key[22:]
            print(newkey)
            checkpoint_student['model_state_dict'][newkey]=checkpoint_student['model_state_dict'][key]
            del checkpoint_student['model_state_dict'][key]
    print(checkpoint_student['model_state_dict'].keys())
    student.load_state_dict(checkpoint_student['model_state_dict'])
    return student

def load_extracter(args):
    extracter=Extracter(args)
    checkpoint_extracter=torch.load(args.test_model_path)
    extracter.load_state_dict(checkpoint_extracter['model_state_dict'],strict=False)
    return extracter

def select_model_student(args):
    
    '''
    根据参数选择你要使用的backbone和分类器
    '''

    name2backbone={
        'resnet18_student':resnet18_student,
        'resnet50_student':resnet50_stduent,
        'strm18_student':strm18_student,
        'resnet18_2fc':resnet18_2fc,
        'resnet50_2fc':resnet50_2fc,
        'strmbackbone':strmbackbone,
        'meta_baseline':meta_baseline,
        'meta_baseline_fc2':meta_baseline_fc2,
        'moblienetv3_fc2':mobile_large_2fc,
        'moblienetv3':mobile_large,
        
        
    }
    
    name2classifier={
        'cos':'CosDistance',
        'TRX':'TRX',
        'TRX_sup':'TRX_sup',
        'CTX':'CTX',
        'TRX_2fc':'TRX_2fc',
        'TRX_1fc_sup':'TRX_1fc_sup',
        'TRX_2fcsup':'TRX_2fcsup',
        'TRX_2fcsup_2':'TRX_2fcsup_2',
        'strmclassifiers':'strmclassifiers',
        'e_dist':'e_dist',
        'e_dist_fc2':'e_dist_fc2',
        'e_dist_fc2_sup':'e_dist_fc2_sup',
        'strm_res18':'strmclassifiers_resnet18',
        'strm_res18_sup':'strmclassifiers_resnet18_sup',
        'strm_1fc_sup':'strm_1fc_sup',
        'e_dist_1fc_sup':'e_dist_1fc_sup'
    }
    
    backbone=name2backbone[args.model_backbone](args)
    classifiername=name2classifier[args.model_classifier]   
    classifier=getattr(classifiers, classifiername)(args) 
    
    if args.num_gpus > 1:
        #只有backbone会多卡运行
        backbone.resnet = torch.nn.DataParallel(backbone.resnet, device_ids=list(range(args.num_gpus)))
    
    return backbone,classifier
    
def select_model_teacher(args):
    '''
    教师的backbone直接加载师兄融合模态的特征
    蒸馏这里的教师模型只需要一个分类器
    
    一般选择test_teacher,因为老师是固定的
    但也可以用train_teacher,自己来测试教师
    '''
    
    name2classifier={    #名字的一个映射
        'cos':'CosDistance',
        'e_dist':'e_dist',
        'e_dist_fc2_sup':'e_dist_fc2_sup_fixed',
        
        'train_teacher':'TRX',
        'test_teacher':'TRX_fixed',
        
        'train_teacher_TRX_sup':'TRX_sup',
        'test_teacher_TRX_sup_fixed':'TRX_sup_fixed',
        
        'train_teacher_TRX_2fcsup':'TRX_2fcsup',
        'test_teacher_TRX_2fcsup_fixed':'TRX_2fcsup_fixed'
    }
    classifiername=name2classifier[args.model_teacher]
    
    classifier=getattr(classifiers, classifiername)(args)  #创建分类器
    
    if args.model_teacher in ['test_teacher','test_teacher_TRX_sup_fixed']:  #分类器有参数的需要加载参数
        classifier=load_teacher(classifier,args)
    
    return classifier


def select_test(args):
    
    '''
    根据输入参数选择需要测试的模型
    并且加载好参数
    '''  
    teacher=classifiers.TRX_fixed(args)
    
    name2test={
        'teacher':load_teacher(teacher,args),
        # 'student':load_student(args),
        # 'extract_feature':load_extracter(args)
    }
    
    model=name2test[args.test_model]
    
    return model