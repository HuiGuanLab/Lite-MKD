from abc import abstractmethod
from collections import namedtuple
from .Register import Register
import torch
import torch.nn.functional as F
import torch.nn as nn

loss_register = Register()
class BaseLoss():
    def __init__(self, **kwargs):
        if "args" in kwargs:
            self.args = kwargs['args']
    
    @abstractmethod
    def loss(self, logits, labels, **args):
        pass
    
class LossBuilder:
    @staticmethod
    def build_loss(type):
        return loss_register[type]
    
@loss_register    
class TRXLoss(BaseLoss):
    def loss(self, logits, labels, device):
        """
        Compute the classification loss.
        """
        tasks_per_batch = self.args.tasks_per_batch
        size = logits.size()
        sample_count = size[0]  # scalar for the loop counter
        num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

        log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
        for sample in range(sample_count):
            log_py[sample] = -F.cross_entropy(logits[sample], labels, reduction='none')
        score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
        return -torch.sum(score, dim=0) / tasks_per_batch

@loss_register
class MyLoss(BaseLoss):
    def loss(self, logits, labels, **args):
        # this function need a parameter named support_videos
        # support_videos: [tensor:shape=(b,t,c) ... tensor:shape=(b,t,c) ]
        # len(support_videos) = 5
        
        trx_loss = LossBuilder.build_loss("TRXLoss")(args=self.args).loss(logits, labels)
        support_videos = args["support_videos"]
        
        cij = self.compute_class_distance(support_videos)
        vij = self.compute_video_distance(support_videos)
        return trx_loss , cij / vij

    def compute_class_distance(self, support_videos):
        c = [i.mean(dim=0).squeeze() for i in support_videos]
        cij = torch.zeros(1).cuda()
        t = 0
        for i in range(len(c)):
            for j in range(i, len(c)):
                cij += torch.cosine_similarity(c[i].view(1, -1), c[j].view(1, -1))
                t += 1
        return cij / t
    
    def compute_video_distance(self, support_videos):
        vij = torch.zeros(1).cuda()
        t = 0
        for class_videos in support_videos:
            for i in range(class_videos.shape[0]):
                for j in range(i, class_videos.shape[0]):
                    vij += torch.cosine_similarity(class_videos[i].view(1, -1), class_videos[j].view(1, -1))
                    t += 1
        return vij /t 

@loss_register
class CELoss(BaseLoss):
    def loss(self, logits, labels, device, **args):
        logits = logits.reshape(-1, self.args.way).to(device)
        labels = labels.to(device)
        return F.cross_entropy(logits, labels)

if __name__ == '__main__':
    logits = torch.randn(5, 10).cuda(0)
    labels = torch.randint(3, (5,)).cuda(0)
    labels = labels.long()
    supprt_videos = [torch.randn((1, 8, 2048)).cuda() for _ in range(5)]
    Args = namedtuple('args', ['tasks_per_batch'])
    args = Args(tasks_per_batch=8)
    l = LossBuilder.build_loss("CELoss").loss(logits, labels)
    print(l)