import torch.nn as nn
import os
import sys
import torch
import numpy as np
from torch.nn import functional as F
from mmseg.ops import resize

class FeedForward(nn.Module):
    # feed forward apply to optional dim
    def __init__(self,c_in,c_out,dim):
        super().__init__()
        self.ff = nn.Linear(c_in,c_out)
        self.dim = dim
    def forward(self,x):
        x = x.transpose(1,self.dim)
        x = self.ff(x)
        x = x.transpose(1,self.dim)
        return x

class KLDLoss(nn.Module):
    def __init__(self,weight,tau,softmax_dim,ff=None):
        super().__init__()
        self.weight = weight
        self.tau = tau
        self.softmax_dim = softmax_dim
        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')

        if ff is None:
            self.ff = None
        else:
            self.ff = FeedForward(**ff)

    def _resize(self,x,gt_semantic_seg):
        return x
    def _mask(self,x_student,x_teacher,gt_semantic_seg):
        return x_student,x_teacher
    def _transform(self,x):
        return x
    def forward(self,x_student,x_teacher,gt_semantic_seg):
        if self.ff is not None:
            x_student = self.ff(x_student)
        x_student,x_teacher = self._resize(x_student,gt_semantic_seg),self._resize(x_teacher,gt_semantic_seg)
        x_student,x_teacher = self._mask(x_student,x_teacher,gt_semantic_seg)
        x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)
        
        x_student = F.log_softmax(x_student/self.tau,dim=self.softmax_dim)
        x_teacher = F.softmax(x_teacher/self.tau,dim=self.softmax_dim)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        loss = loss/(x_student.numel()/x_student.shape[-1])
        return loss

# logits: [b,C,255,255]
class ChannelGroupLossForLogits(KLDLoss):
    def __init__(self,weight,tau,group_size,apply_mask=False,apply_resize=True,ff=None):
        super().__init__(weight,tau,softmax_dim=-1,ff=None)
        self.group_size = group_size
        self.apply_mask = apply_mask
        self.apply_resize = apply_resize
    def _resize(self,x,gt_semantic_seg):
        if self.apply_resize:
            x= resize(
                input=x,
                size=gt_semantic_seg.shape[2:],
                mode='bilinear',
                align_corners=False)
        return x
    def _mask(self,x_student,x_teacher,gt_semantic_seg):
        if not self.apply_mask:
            return x_student,x_teacher

        teacher_pd = torch.argmax(x_teacher,dim=1,keepdim=True)

        mask_tm = (teacher_pd != gt_semantic_seg).bool() # teacher mistakes
        mask_bg = (gt_semantic_seg == 255).bool() # background (255)
        mask = (mask_bg & mask_tm).expand(-1,150,-1,-1)

        x_student[mask] = -1e9
        x_teacher[mask] = -1e9
        return x_student,x_teacher
    def _transform(self,x):
        B,C,W,H = x.shape
        x = x.reshape(B,C,W*H)
        x = x.reshape(B,C//self.group_size,self.group_size*W*H)
        return x

class SpatialGroupLossForLogits(KLDLoss):
    def __init__(self,weight,tau,kernel_size, dilation, padding, stride, apply_mask=False,ff=None):
        super().__init__(weight,tau,softmax_dim=-1,ff=None)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.apply_mask = apply_mask
    def _resize(self,x,gt_semantic_seg):
        x= resize(
            input=x,
            size=gt_semantic_seg.shape[2:],
            mode='bilinear',
            align_corners=False)
        return x
    def _mask(self,x_student,x_teacher,gt_semantic_seg):
        if not self.apply_mask:
            return x_student,x_teacher

        teacher_pd = torch.argmax(x_teacher,dim=1,keepdim=True)

        mask_tm = (teacher_pd != gt_semantic_seg).bool() # teacher mistakes
        mask_bg = (gt_semantic_seg == 255).bool() # background (255)
        mask = (mask_bg & mask_tm).expand(-1,150,-1,-1)

        x_student[mask] = -1e9
        x_teacher[mask] = -1e9
        return x_student,x_teacher
    def _transform(self,x):
        x = F.unfold(x,self.kernel_size, self.dilation, self.padding, self.stride)
        x = x.transpose(1,2)
        return x

# attention:[b,num_head,WH,WH']
class ChannelGroupLossForAttention(KLDLoss):
    def __init__(self,weight,tau,group_size,ff=None):
        super().__init__(weight,tau,softmax_dim=-1,ff=None)
        self.group_size = group_size
    def _transform(self,x):
        B,num_head,WH,WH_ = x.shape
        x = x.reshape(B*num_head,WH,WH_)
        x = x.reshape(B*num_head,WH*self.group_size,WH_//self.group_size)
        x = x.transpose(1,2)
        return x

class SpatialGroupLossForAttention(KLDLoss):
    def __init__(self,weight,tau,kernel_size, dilation, padding, stride,ff=None):
        super().__init__(weight,tau,softmax_dim=-1,ff=None)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
    def _transform(self,x):
        B,num_head,WH,WH_ = x.shape
        x = x.reshape(B*num_head,WH,WH_)
        x = x.transpose(1,2)
        W = int(np.sqrt(WH))
        x = x.reshape(B*num_head,WH_,W,W)
        x = F.unfold(x,self.kernel_size, self.dilation, self.padding, self.stride)
        return x

# feature:[B,WH,C]

class ChannelGroupLossForFeature(KLDLoss):
    def __init__(self,weight,tau,group_size,ff=None):
        super().__init__(weight,tau,softmax_dim=-1,ff=None)
        self.group_size = group_size
    def _transform(self,x):
        B,WH,C = x.shape
        x = x.reshape(B,WH*self.group_size,C//self.group_size)
        x = x.transpose(1,2)
        return x

class SpatialGroupLossForFeature(KLDLoss):
    def __init__(self,weight,tau,kernel_size, dilation, padding, stride,ff=None):
        super().__init__(weight,tau,softmax_dim=-1,ff=None)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
    def _transform(self,x):
        B,WH,C = x.shape
        x = x.transpose(1,2)
        W = int(np.sqrt(WH))
        x = x.reshape(B,C,W,W)
        x = F.unfold(x,self.kernel_size, self.dilation, self.padding, self.stride)
        return x

class MSELoss(nn.Module):
    def __init__(self,weight):
        super().__init__()
        self.weight = weight
        self.MSE = nn.MSELoss()
    def forward(self,x_student,x_teacher,gt_semantic_seg):
        loss = self.weight * self.MSE(x_student,x_teacher)
        return loss

