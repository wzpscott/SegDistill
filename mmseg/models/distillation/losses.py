import torch.nn as nn
import os
import sys
import torch
import numpy as np
from torch.nn import functional as F
from mmseg.ops import resize


class KLDLoss(nn.Module):
    def __init__(self,weight,tau,\
        reshape_config=None,resize_config=None,mask_config=None,transform_config=None,ff_config=None,\
        earlystop_config=None,shift_config=None,warmup_config=0):
        super().__init__()
        self.weight = weight
        self.tau = tau
        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')

        self.reshape_config = reshape_config if reshape_config else False
        self.resize_config = resize_config if resize_config else False
        self.mask_config = mask_config if mask_config else False
        self.transform_config = transform_config if transform_config else False

        self.earlystop_config = earlystop_config if earlystop_config else False
        self.shift_config = shift_config if shift_config else False

        self.ff = nn.Conv2d(**ff_config,kernel_size=1).cuda() if ff_config else False
        
        self.warmup_config = warmup_config if warmup_config > 0 else False

        self.weight_ = weight



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
        mask_st = (student_pd == gt_semantic_seg).bool() # student预测正确的pixel
        mask_bg = (gt_semantic_seg == 255).bool() # 标签为background的pixel
        masks = {'mask_tm':mask_tm,'mask_st':mask_st,'mask_bg':mask_bg}

        masks_to_apply = [masks[mask_name] for mask_name in self.mask_config] # 使用的mask
        mask = masks_to_apply[0]
        for m in masks_to_apply:
            mask = mask & m
        mask = mask.expand(-1,150,-1,-1)

        return mask

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

    def _shift(self,x,step):
        B,C,W,H = x.shape
        stride = step % C
        x1 = x[:,:stride,:,:]
        x2 = x[:,stride:,:,:]
        x = torch.cat([x2,x1],dim=1).contiguous()
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
        if self.warmup_config:
            if step < self.warmup_config:
                self.weight = step/self.warmup_config
            else:
                self.weight = self.weight_
        if self.earlystop_config:
            if step > self.earlystop_config:
                self.weight = 0

        x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
        if self.ff :
            x_student = self._ff(x_student)
        if self.resize_config:
            x_student,x_teacher = self._resize(x_student,gt_semantic_seg),self._resize(x_teacher,gt_semantic_seg)
        if self.mask_config:
            mask = self._mask(x_student,x_teacher,gt_semantic_seg)
        if self.shift_config:
            x_student,x_teacher = self._shift(x_student,step),self._shift(x_teacher,step)
            
        if self.transform_config:
            x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)
            if self.mask_config:
                mask = self._transform(mask)
            else:
                mask = torch.zeros_like(x_student).cuda()

        x_student = F.log_softmax(x_student/self.tau,dim=-1)
        x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x_student,x_teacher)*(1-mask.float())
        loss = loss/(x_student.numel()/x_student.shape[-1])
        return loss

class ShiftChannelLoss(KLDLoss):
    def __init__(self,weight,tau,\
        reshape_config,resize_config,mask_config,transform_config,ff_config,\
        earlystop_config=None):
        super().__init__(weight,tau,reshape_config,resize_config,mask_config,transform_config,ff_config,earlystop_config)
    def _transform(self,x):
        B,C,W,H = x.shape
        x = x.reshape(B,C,W*H,1).permute(0,2,1,3)
        x = F.unfold(x,kernel_size=(self.transform_config['group_size'],1)).transpose(1,2)
        return x
    def forward(self,x_student,x_teacher,gt_semantic_seg,step):
        if self.earlystop_config:
            if step > self.earlystop_config:
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

class ShuffleChannelLoss(KLDLoss):
    def __init__(self,weight,tau,\
        reshape_config,resize_config,mask_config,transform_config,ff_config,\
        earlystop_config=None,shuffle_config=None):
        self.weight_ = weight
        self.shuffle_config = shuffle_config
        super().__init__(weight,tau,reshape_config,resize_config,mask_config,transform_config,ff_config,earlystop_config)
    def _shuffle(self,x,step):
        B,N,G = x.shape
        if self.shuffle_config != 0:
            if step % self.shuffle_config == 0:
                x = x.transpose(1,2)
                x = x.reshape(B,-1)
                x = x.reshape(B,N,G)
        else:
            x = x.transpose(1,2)
            x = x.reshape(B,-1)
            x = x.reshape(B,N,G)
        return x

        
    def forward(self,x_student,x_teacher,gt_semantic_seg,step):
        if step < self.warmup_config:
            self.weight = step/self.warmup_config
        else:
            self.weight = self.weight_

        if self.earlystop_config:
            if step > self.earlystop_config:
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
        x_student,x_teacher = self._shuffle(x_student,step),self._shuffle(x_teacher,step)

        x_student = F.log_softmax(x_student/self.tau,dim=-1)
        x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        loss = loss/(x_student.numel()/x_student.shape[-1])
        return loss
    
class ShuffleShiftLoss(KLDLoss):
    def __init__(self,weight,tau,\
        reshape_config,resize_config,mask_config,transform_config,ff_config,\
        earlystop_config=None):
        super().__init__(weight,tau,reshape_config,resize_config,mask_config,transform_config,ff_config,earlystop_config)
    def _transform(self,x):
        B,C,W,H = x.shape
        x = x.reshape(B,C,W*H,1).permute(0,2,1,3)
        x = F.unfold(x,kernel_size=(self.transform_config['group_size'],1)).transpose(1,2)
        return x
    def _shuffle(self,x_student,x_teacher):
        B,C,W,H = x_student.shape
        idx = torch.randperm(C)
        x_student = x_student[:,idx,:,:].contiguous()
        x_teacher = x_teacher[:,idx,:,:].contiguous()
        return x_student,x_teacher
    def forward(self,x_student,x_teacher,gt_semantic_seg,step):
        if self.earlystop_config:
            if step > self.earlystop_config:
                self.weight = 0
        x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
        if self.ff :
            x_student = self._ff(x_student)
        if self.resize_config:
            x_student,x_teacher = self._resize(x_student,gt_semantic_seg),self._resize(x_teacher,gt_semantic_seg)
        if self.mask_config:
            x_student,x_teacher = self._mask(x_student,x_teacher,gt_semantic_seg)
        x_student,x_teacher = self._shuffle(x_student,x_teacher)
        if self.transform_config:
            x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)
        
        x_student = F.log_softmax(x_student/self.tau,dim=-1)
        x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        loss = loss/(x_student.numel()/x_student.shape[-1])
        return loss 


class AttentionLoss(nn.Module):
    def __init__(self,weight,tau,transform_config,earlystop_config,warmup_config):
        super().__init__()
        self.weight = weight
        self.weight_ = weight
        self.tau = tau
        self.transform_config = transform_config
        self.earlystop_config = earlystop_config if earlystop_config else False
        self.warmup_config = warmup_config

        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')

    def _transform(self,x):
        loss_type = self.transform_config['loss_type']
        B,N,C = x.shape
        if loss_type == 'channel':
            group_size = self.transform_config['group_size']
            x = x.permute(0,2,1)
            x = x.reshape(B,C//group_size,-1)
        elif loss_type == 'spatial':
            pass
        return x
    def forward(self,attn_student,v_student,attn_teacher,v_teacher,gt,step):
        if step < self.warmup_config:
            self.weight = step/self.warmup_config
        else:
            self.weight = self.weight_

        if self.earlystop_config:
            if step > self.earlystop_config:
                self.weight = 0

        B,num_head,_,C = v_student.shape
        _,_,N,_ = attn_student.shape
        attn_student  = attn_student.softmax(dim=-1)
        attn_teacher  = attn_teacher.softmax(-1)
        C = C * num_head


        print((attn_student @ v_student).transpose(1, 2).shape)
        print((teacher @ v_student).transpose(1, 2).shape)
        x = (attn_student @ v_student).transpose(1, 2).reshape(B, N, C)
        x_mimic = (attn_teacher @ v_student).transpose(1, 2).reshape(B, N, C)

        x,x_mimic = self._transform(x),self._transform(x_mimic)
        x = F.log_softmax(x/self.tau,dim=-1)
        x_mimic = F.softmax(x_mimic/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x,x_mimic)
        loss = loss/(x.numel()/x.shape[-1])
        return loss


class StudentRE(nn.Module):
    def __init__(self,weight,tau,transform_config,earlystop_config,**kwargs):
        super().__init__()
        self.weight = weight
        self.weight_ = weight
        self.tau = tau
        self.transform_config = transform_config
        self.earlystop_config = earlystop_config if earlystop_config else False

        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')

    def _transform(self,x):
        loss_type = self.transform_config['loss_type']
        B,N,C = x.shape
        if loss_type == 'channel':
            group_size = self.transform_config['group_size']
            x = x.permute(0,2,1)
            x = x.reshape(B,C//group_size,-1)
        elif loss_type == 'spatial':
            pass
        return x
    def forward(self,attn_student,v_student,attn_teacher,v_teacher,gt,step):

        if self.earlystop_config:
            if step > self.earlystop_config:
                self.weight = 0
        v_student = v_student.detach()
        B,num_head,_,C = v_student.shape
        _,_,N,_ = attn_student.shape
        attn_student  = attn_student.softmax(dim=-1)
        attn_teacher  = attn_teacher.softmax(-1)
        C = C * num_head
        x = (attn_student @ v_student).transpose(1, 2).reshape(B, N, C)
        x_mimic = (attn_teacher @ v_student).transpose(1, 2).reshape(B, N, C)

        x,x_mimic = self._transform(x),self._transform(x_mimic)
        x = F.log_softmax(x/self.tau,dim=-1)
        x_mimic = F.softmax(x_mimic/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x,x_mimic)
        loss = loss/(x.numel()/x.shape[-1])
        return loss

class TeacherRE(nn.Module):
    def __init__(self,weight,tau,transform_config,earlystop_config,**kwargs):
        super().__init__()
        self.weight = weight
        self.weight_ = weight
        self.tau = tau
        self.transform_config = transform_config
        self.earlystop_config = earlystop_config if earlystop_config else False

        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')

    def _transform(self,x):
        loss_type = self.transform_config['loss_type']
        B,N,C = x.shape
        if loss_type == 'channel':
            group_size = self.transform_config['group_size']
            x = x.permute(0,2,1)
            x = x.reshape(B,C//group_size,-1)
        elif loss_type == 'spatial':
            pass
        return x
    def forward(self,attn_student,v_student,attn_teacher,v_teacher,gt,step):

        if self.earlystop_config:
            if step > self.earlystop_config:
                self.weight = 0

        B,num_head,_,C = v_teacher.shape
        _,_,N,_ = attn_teacher.shape
        attn_student  = attn_student.softmax(dim=-1)
        attn_teacher  = attn_teacher.softmax(-1)
        C = C * num_head

        # print((attn_student @ v_student).transpose(1, 2).shape)
        # print((teacher @ v_student).transpose(1, 2).shape)
        x = (attn_teacher @ v_teacher).transpose(1, 2).reshape(B, N, C)
        x_mimic = (attn_student @ v_teacher).transpose(1, 2).reshape(B, N, C)

        x,x_mimic = self._transform(x),self._transform(x_mimic)
        x = F.log_softmax(x/self.tau,dim=-1)
        x_mimic = F.softmax(x_mimic/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x,x_mimic)
        loss = loss/(x.numel()/x.shape[-1])
        return loss

class FlattenLoss(nn.Module):
    def __init__(self,weight,tau,transform_config,earlystop_config,**kwargs):
        super().__init__()
        self.weight = weight
        self.weight_ = weight
        self.tau = tau
        self.transform_config = transform_config
        self.earlystop_config = earlystop_config if earlystop_config else False

        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')

    def forward(self,x_student,x_teacher,gt,step):
        if self.earlystop_config:
            if step > self.earlystop_config:
                self.weight = 0

        B,WH,C_s = x_student.shape
        B,WH,C_t = x_teacher.shape

        x_student = F.log_softmax(x_student/self.tau,dim=1)
        x_teacher = F.softmax(x_teacher/self.tau,dim=1)

        x_student = x_student.mean(dim=2)
        x_teacher = x_teacher.mean(dim=2)

        loss = self.weight*self.KLDiv(x_student,x_teacher)
        loss = loss/(x_student.numel()/x_student.shape[-1])
        return loss


class MTLoss(KLDLoss):
    def __init__(self,weight,tau,reshape_config,resize_config,transform_config,latestart_config,earlystop_config,rot_config=None,**kwargs):
        super().__init__(weight,tau,reshape_config=reshape_config,resize_config=resize_config,\
            transform_config=transform_config)
        self.weight_ = weight

        self.latestart_config = latestart_config if latestart_config else False
        self.earlystop_config = earlystop_config if earlystop_config else False
        self.rot = rot_config if rot_config else False

        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')
    def forward(self,x_student,x_teacher,gt,step):
        if self.latestart_config:
            if step < self.latestart_config:
                self.weight = 0
            else:
                self.weight = self.weight_

        if self.rot:
            i,interval = self.rot
            if ((step-1) // interval) % 4 == i:
                self.weight = self.weight_
            else:
                self.weight = 0

        if self.earlystop_config:
            if step > self.earlystop_config:
                self.weight = 0

        x_student,x_teacher = self._reshape(x_student),self._reshape(x_teacher)
        if self.resize_config:
            x_student,x_teacher = self._resize(x_student,gt),self._resize(x_teacher,gt)
        if self.transform_config:
            x_student,x_teacher = self._transform(x_student),self._transform(x_teacher)
        
        x_student = F.log_softmax(x_student/self.tau,dim=-1)
        x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        loss = loss/(x_student.numel()/x_student.shape[-1])
        return loss



class MTRandomLoss(KLDLoss):
    def __init__(self,weight,tau,reshape_config,resize_config,transform_config,latestart_config,earlystop_config,rot_config=None,**kwargs):
        super().__init__(weight,tau,reshape_config=reshape_config,resize_config=resize_config,\
            transform_config=transform_config)
        self.weight_ = weight

        self.latestart_config = latestart_config if latestart_config else False
        self.earlystop_config = earlystop_config if earlystop_config else False
        self.rot = rot_config if rot_config else False

        self.KLDiv = torch.nn.KLDivLoss(reduction='sum')
    def forward(self,x_student,x_teacher,gt,step):
        if self.latestart_config:
            if step < self.latestart_config:
                self.weight = 0
            else:
                self.weight = self.weight_
        if self.earlystop_config:
            if step > self.earlystop_config:
                self.weight = 0

        if self.rot:
            i,interval = self.rot
            if ((step-1) // interval) % 4 == i:
                self.weight = self.weight_
            else:
                self.weight = 0

        # N_teacher,B,C,W,H = x_teacher.shape
        # teacher_mask = torch.randint(0,2,[N_teacher,B,C]).cuda()
        # cnt = teacher_mask.sum()
        # teacher_mask = .expand(N_teacher,B,C,W,H).cuda() # [N_teacher,B,C,W,H]
        # x_teacher = (x_teacher*teacher_mask).sum(dim=0)
        # x_teacher = x_teacher / (teacher_mask.sum(dim=0))

        x_student = self._reshape(x_student)
        x_student = self._resize(x_student,gt)
        x_student = self._transform(x_student)

        x_teachers = []
        for x in x_teacher:
            x = self._reshape(x)
            x = self._resize(x,gt)
            x = self._transform(x)
            x_teachers.append(x)
        x_teacher = torch.cat([i.unsqueeze(0) for i in x_teachers],dim=0) 
        num_teachers,B,C,N = x_teacher.shape

        teacher_mask = torch.randint(0,2,[num_teachers,B,C]).cuda()
        teacher_cnt = teacher_mask.sum(dim=0).reshape(B,C,1).expand(B,C,N)
        teacher_mask = teacher_mask.reshape(num_teachers,B,C,1)
        x_teacher = (x_teacher*teacher_mask).sum(dim=0)
        x_teacher = x_teacher / teacher_cnt
        x_teacher[teacher_cnt==0] = x_student[teacher_cnt==0].clone()

        x_student = F.log_softmax(x_student/self.tau,dim=-1)
        x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
        loss = self.weight*self.KLDiv(x_student,x_teacher)
        loss = loss/(x_student.numel()/x_student.shape[-1])
        return loss
