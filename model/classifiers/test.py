import torch

s=torch.rand(2,2,3)
print(s.shape)
x=s.unsqueeze(1).repeat(1,2,1,1)
y=s.unsqueeze(0).repeat(2,1,1,1)
print(x)
print(y)
diff=x-y
norm_sq = torch.norm(diff, dim=[-2,-1])**2
print(norm_sq)

