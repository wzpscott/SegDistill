import torch.nn as nn
import os
import sys
import torch
import numpy as np
from torch.nn import functional as F
from mmseg.ops import resize


class KLDLoss(nn.Module):
    def __init__(self,weight,tau,reshape_config,resize_config,mask_config,transform_config,ff_config):
        super().__init__()
        self.weight = weight
        self.tau = tau
        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')

        self.reshape_config = reshape_config
        self.resize_config = resize_config
        self.mask_config = mask_config
        self.transform_config = transform_config

        self.ff = nn.Conv2d(**ff_config).cuda() if ff_config else False


    def _resize(self,x,gt_semantic_seg):
        x = F.interpolate(
            input=x,
            size=gt_semantic_seg.shape[2:],
            **self.resize_config)
        return x

    def _mask(self,x_student,x_teacher,gt_semantic_seg):
        teacher_pd = torch.argmax(x_teacher,dim=1,keepdim=True)
        student_pd = torch.argmax(x_student,dim=1,keepdim=True)
        if not self.resize_config: # 对gt进行下采样，使size一致
            gt_semantic_seg = F.interpolate(
                input=gt_semantic_seg,
                size=teacher_pd.shape[2:],
                mode='nearest'
            )
        
        mask_tm = (teacher_pd != gt_semantic_seg).bool() # teacher预测错误的pixel
        mask_st = (student_pd != gt_semantic_seg).bool() # student预测正确的pixel
        mask_bg = (gt_semantic_seg == 255).bool() # 标签为background的pixel
        masks = {'mask_tm':mask_tm,'mask_st':mask_st,'mask_bg':mask_bg}

        masks_to_apply = [masks[mask_name] for mask_name in self.mask_config['masks']] # 使用的mask
        mask = masks_to_apply[0]
        for m in masks_to_apply:
            mask = mask & m
        mask = mask.expand(-1,150,-1,-1)

        x_student[mask] = -1e7
        x_teacher[mask] = -1e7 # 被mask的元素不计算蒸馏损失
        return x_student,x_teacher

    def _reshape(self,x):
        # 对任意输入，化为[B,C,W,H]的形式
        if self.reshape_config == 'logits':
            # [B,C,W,H]
            x = x # 不用进行操作
        elif self.reshape_config == 'attention':
            # [B,num_head,WH,C]
            B,num_head,WH,C = x.shape
            W = int(np.sqrt(WH))
            x = x.reshape(B*num_head,W,-1,C)
            x = x.permute(0,3,1,2)
        elif self.reshape_config == 'feature':
            # [B,WH,C]
            B,WH,C = x.shape
            W = int(np.sqrt(WH))
            x = x.reshape(B,W,-1,C)
            x = x.permute(0,3,1,2)
            
        return x

    def _ff(self,x):
        return self.ff(x)

    def _transform(self,x):
        loss_type = self.transform_config['loss_type']
        if loss_type == 'channel':
            group_size = self.transform_config['group_size']
            B,C,W,H = x.shape
            x = x.reshape(B,C//group_size,-1)
        elif loss_type == 'spatial':
            kernel_size = self.transform_config['kernel_size']
            stride = self.transform_config['stride']
            padding = self.transform_config['padding'] if 'padding' in self.transform_config else 0
            x = F.unfold(x,kernel_size, dilation=1, padding=padding, stride=stride)
            x = x.transpose(2,1)
        return x

    def forward(self,x_student,x_teacher,gt_semantic_seg,step):
        x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
        if self.ff :
            x_student = self._ff(x_student)
        if self.resize_config:
            x_student,x_teacher = self._resize(x_student,gt_semantic_seg),self._resize(x_teacher,gt_semantic_seg)
        if self.mask_config:
            x_student,x_teacher = self._mask(x_student,x_teacher,gt_semantic_seg)
        if self.transform_config:
            x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)
        
        x_student = F.log_softmax(x_student/self.tau,dim=-1)
        x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        loss = loss/(x_student.numel()/x_student.shape[-1])
        return loss

class TwoStageChannelGroup(KLDLoss):
    def __init__(self):
        weight = 1
        tau = 1

        reshape_config = 'logits'
        resize_config = {'mode':'bilinear','align_corners':False}
        mask_config = False
        transform_config = {'loss_type':'channel'}

        ff_config = False

        super().__init__(weight,tau,reshape_config,resize_config,mask_config,transform_config,ff_config)
    def forward(self,x_student,x_teacher,gt_semantic_seg,step):
        if step < 96000:
            self.transform_config['group_size'] = 3
        else:
            self.transform_config['group_size'] = 1
        
        x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
        if self.ff :
            x_student = self._ff(x_student)
        if self.resize_config:
            x_student,x_teacher = self._resize(x_student,gt_semantic_seg),self._resize(x_teacher,gt_semantic_seg)
        if self.mask_config:
            x_student,x_teacher = self._mask(x_student,x_teacher,gt_semantic_seg)
        if self.transform_config:
            x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)
        
        x_student = F.log_softmax(x_student/self.tau,dim=-1)
        x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        loss = loss/(x_student.numel()/x_student.shape[-1])
        return loss

class ThreeStageChannelGroup(KLDLoss):
    def __init__(self):
        weight = 1
        tau = 1

        reshape_config = 'logits'
        resize_config = {'mode':'bilinear','align_corners':False}
        mask_config = False
        transform_config = {'loss_type':'channel'}

        ff_config = False

        super().__init__(weight,tau,reshape_config,resize_config,mask_config,transform_config,ff_config)
    def forward(self,x_student,x_teacher,gt_semantic_seg,step):
        if step < 96000:
            self.transform_config['group_size'] = 3
        elif step < 144000:
            self.transform_config['group_size'] = 1
        else:
            self.weight = 0
        
        x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
        if self.ff :
            x_student = self._ff(x_student)
        if self.resize_config:
            x_student,x_teacher = self._resize(x_student,gt_semantic_seg),self._resize(x_teacher,gt_semantic_seg)
        if self.mask_config:
            x_student,x_teacher = self._mask(x_student,x_teacher,gt_semantic_seg)
        if self.transform_config:
            x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)
        
        x_student = F.log_softmax(x_student/self.tau,dim=-1)
        x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        loss = loss/(x_student.numel()/x_student.shape[-1])
        return loss