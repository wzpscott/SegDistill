import torch.nn as nn
import os
import sys
import torch
import numpy as np
from torch.nn import functional as F
from mmseg.ops import resize

class MSELoss(nn.Module):
    def __init__(self,weight):
        super().__init__()
        self.weight = weight
        self.MSE = nn.MSELoss()
    def forward(self,x_student,x_teacher,gt_semantic_seg):
        loss = self.weight * self.MSE(x_student,x_teacher)
        return loss
        
class Baseloss(nn.Module):
    '''
    Other losses can inherit Baseloss class and rewrite the tranform function to get desirable output shape.
    When rewriting, permute the output to put the dimension you want to apply softmax to the last 
        because by default all loss function apply softmax to the last dimention followed by a KLDivLoss
    '''
    def __init__(self,weight,tau,softmax_dim=-1):
        super().__init__()
        self.weight = weight
        self.tau = tau
        self.softmax_dim = softmax_dim
        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')
    def resize(self,x,gt_semantic_seg,mode='bilinear'):
            x= resize(
                input=x,
                size=gt_semantic_seg.shape[2:],
                mode=mode,
                align_corners=False)
            return x
    def transform(self,x,gt_semantic_seg):
        return x
    def forward(self,x_student,x_teacher,gt_semantic_seg):
        x_student,x_teacher = self.transform(x_student,gt_semantic_seg),self.transform(x_teacher,gt_semantic_seg)
        x_student = F.log_softmax(x_student/self.tau,dim=self.softmax_dim)
        x_teacher = F.softmax(x_teacher/self.tau,dim=self.softmax_dim)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        return loss/(x_student.numel()/x_student.shape[-1])

# losses for logits
# input shape: [B,C,H,W]
class ChannelWiseLoss(Baseloss):
    # 'Apply softmax channel wise'
    def __init__(self,weight,tau):
        super().__init__(weight,tau)
    def transform(self,x,gt_semantic_seg):
        x = self.resize(x,gt_semantic_seg)
        B,C,W,H = x.shape
        x = x.reshape(B,C,W*H)
        return x

class ChannelWiseLossWithMask(Baseloss):
    def __init__(self,weight,tau):
        super().__init__(weight,tau)
    def transform(self,x,gt_semantic_seg):
        x = self.resize(x,gt_semantic_seg)
        B,C,W,H = x.shape
        x = x.reshape(B,C,W*H)
        return x
    def forward(self,x_student,x_teacher,gt_semantic_seg):
        x_student,x_teacher = self.transform(x_student,gt_semantic_seg),self.transform(x_teacher,gt_semantic_seg)
        teacher_pd = torch.argmax(x_teacher,dim=1,keepdim=True)

        gt_semantic_seg = gt_semantic_seg.reshape(gt_semantic_seg.shape[0],1,-1)

        mask_tm = (teacher_pd != gt_semantic_seg).bool() # teacher mistakes
        mask_bg = (gt_semantic_seg == 255).bool() # background (255)
        mask = (mask_bg & mask_tm).expand(-1,150,-1)

        x_student[mask] = -1e9
        x_teacher[mask] = -1e9

        x_student = F.log_softmax(x_student/self.tau,dim=self.softmax_dim)
        x_teacher = F.softmax(x_teacher/self.tau,dim=self.softmax_dim)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        return loss/(x_student.numel()/x_student.shape[-1])


class SpatialWiseLoss(Baseloss):
    # 'Apply softmax spatial wise'
    def __init__(self,weight,tau):
        super().__init__(weight,tau)
    def transform(self,x,gt_semantic_seg):
        x = self.resize(x,gt_semantic_seg)
        B,C,W,H = x.shape
        x = x.permute(0,2,3,1)
        return x

class ChannelGroupLoss(Baseloss):
    # group the channels and apply softmax
    def __init__(self,weight,tau,group_num):
        super().__init__(weight,tau)
        self.group_num = group_num
    def transform(self,x,gt_semantic_seg):
        x = self.resize(x,gt_semantic_seg)
        B,C,W,H = x.shape
        x = x.reshape(B,C,W*H)
        x = x.reshape(B,C//self.group_num,self.group_num*W*H)
        return x

class SpatialGroupLoss(Baseloss):
    # use a shift window to get spatial groups
    def __init__(self,weight,tau,kernel_size, dilation=1, padding=0, stride=1):
        super().__init__(weight,tau)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
    def transform(self,x,gt_semantic_seg):
        x = self.resize(x,gt_semantic_seg)
        B,C,W,H = x.shape
        x = F.unfold(x,self.kernel_size, self.dilation, self.padding, self.stride)
        x = x.permute(0,2,1)
        return x

# losses for attention map
# input shape: [B,num_head,WH,256]

class ChannelWiseLossAttention(Baseloss):
    def __init__(self,weight,tau,softmax_dim=-1):
        super().__init__(weight,tau,softmax_dim = -1)
    def transform(self,x,gt_semantic_seg):
        B,num_head,WH,_ = x.shape
        x = x.mean(dim=1)
        x = x.permute(0,2,1)
        return x

class SpatialLossAttention(Baseloss):
    def __init__(self,weight,tau,softmax_dim=-1):
        super().__init__(weight,tau,softmax_dim = -1)
    def transform(self,x,gt_semantic_seg):
        x = x.mean(dim=1)
        return x

class ChannelGroupLossAttention(Baseloss):
    def __init__(self,weight,tau,group_num,softmax_dim = -1):
        super().__init__(weight,tau,softmax_dim = -1)
        self.group_num = group_num
    def transform(self,x,gt_semantic_seg):
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
    def transform(self,x,gt_semantic_seg):
        x = x.permute(0,2,1)
        return x

class SpatialLossFeature(Baseloss):
    def __init__(self,weight,tau,softmax_dim=-1):
        super().__init__(weight,tau,softmax_dim = -1)
    def transform(self,x,gt_semantic_seg):
        # x = x.permute(0,2,1)
        return x