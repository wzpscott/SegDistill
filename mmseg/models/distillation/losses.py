import torch.nn as nn
import os
import sys
import torch
import numpy as np
from torch.nn import functional as F


class Baseloss(nn.Module):
    def __init__(self,weight,tau,softmax_dim=-1):
        super().__init__()
        self.weight = weight
        self.tau = tau
        self.softmax_dim = softmax_dim
        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')
    def transform(self,x):
        return x
    def forward(self,x_student,x_teacher):
        x_student,x_teacher = self.transform(x_student),self.transform(x_teacher)
        x_student = F.log_softmax(x_student/self.tau,dim=self.softmax_dim)
        x_teacher = F.softmax(x_teacher/self.tau,dim=self.softmax_dim)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        return loss/(x_student.numel()/x_student.shape[-1])

# losses for logits
# input shape: [B,C,H,W]
class ChannelWiseLoss(Baseloss):
    def __init__(self,weight,tau):
        super().__init__(weight,tau)
    def transform(self,x):
        B,C,W,H = x.shape
        x = x.reshape(B,C,W*H)
        return x

class SpatialWiseLoss(Baseloss):
    def __init__(self,weight,tau):
        super().__init__(weight,tau)
    def transform(self,x):
        B,C,W,H = x.shape
        x = x.permute(0,2,3,1)
        return x

class ChannelGroupLoss(Baseloss):
    def __init__(self,weight,tau,group_num):
        super().__init__(weight,tau)
        self.group_num = group_num
    def transform(self,x):
        B,C,W,H = x.shape
        x = x.reshape(B,C,W*H)
        x = x.reshape(B,C//self.group_num,self.group_num*W*H)
        return x

class SpatialGroupLoss(Baseloss):
    def __init__(self,weight,tau,kernel_size, dilation=1, padding=0, stride=1):
        super().__init__(weight,tau)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
    def transform(self,x):
        B,C,W,H = x.shape
        x = F.unfold(x,self.kernel_size, self.dilation, self.padding, self.stride)
        x = x.permute(0,2,1)
        return x

# losses for attention map
# input shape: [B,num_head,WH,256]

class ChannelWiseLossAttention(Baseloss):
    def __init__(self,weight,tau,softmax_dim=-1):
        super().__init__(weight,tau,softmax_dim = -1)
    def transform(self,x):
        B,num_head,WH,_ = x.shape
        x = x.mean(dim=1)
        x = x.permute(0,2,1)
        return x

class SpatialLossAttention(Baseloss):
    def __init__(self,weight,tau,softmax_dim=-1):
        super().__init__(weight,tau,softmax_dim = -1)
    def transform(self,x):
        x = x.mean(dim=1)
        return x

class ChannelGroupLossAttention(Baseloss):
    def __init__(self,weight,tau,group_num,softmax_dim = -1):
        super().__init__(weight,tau,softmax_dim = -1)
        self.group_num = group_num
    def transform(self,x):
        B,num_head,WH,_ = x.shape
        x = x.mean(dim=1)
        x = x.transpose(B,-1,WH)
        x = x.reshape(B,-1,self.group_num*WH)
        return x

# losses for feature
# input shape: [B,WH,C]
class ChannelWiseLossFeature(Baseloss):
    def __init__(self,weight,tau,softmax_dim=-1):
        super().__init__(weight,tau,softmax_dim = -1)
    def transform(self,x):
        x = x.permute(0,2,1)
        return x