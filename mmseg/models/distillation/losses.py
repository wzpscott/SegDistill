import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmseg.ops import resize

class KLDLoss(nn.Module):
    def __init__(self,alpha=1,tau=1,resize_config=None,shuffle_config=None,transform_config=None,\
                warmup_config=None,earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

    def resize(self,x,gt):
        mode = self.resize_config['mode']
        align_corners = self.resize_config['align_corners']
        x = F.interpolate(
            input=x,
            size=gt.shape[2:],
            mode=mode,
            align_corners=align_corners)
        return x

    def shuffle(self,x_student,x_teacher,n_iter):
        interval = self.shuffle_config['interval']
        B,C,W,H = x_student.shape
        if n_iter % interval == 0:
            idx = torch.randperm(C)
            x_student = x_student[:,idx,:,:].contiguous()
            x_teacher = x_teacher[:,idx,:,:].contiguous()
        return x_student,x_teacher

    def transform(self,x):
        B,C,W,H = x.shape
        loss_type = self.transform_config['loss_type']
        if loss_type == 'pixel':
            x = x.permute(0,2,3,1)
            x = x.reshape(B,W*H,C)
        elif loss_type == 'channel':
            group_size = self.transform_config['group_size']
            if C % group_size == 0:
                x = x.reshape(B,C//group_size,-1)
            else:
                n = group_size - C % group_size
                x_pad =-1e9 * torch.ones(B,n,W,H).cuda()
                x = torch.cat([x,x_pad],dim=1)
                x = x.reshape(B,(C+n)//group_size,-1)
        return x

    def warmup(self,n_iter):
        mode = self.warmup_config['mode']
        warmup_iters = self.warmup_config['warmup_iters']
        if n_iter > warmup_iters:
            return
        elif n_iter == warmup_iters:
            self.alpha = self.alpha_0
            return 
        else:
            if mode == 'linear' :
                self.alpha = self.alpha_0 * (n_iter/warmup_iters)
            elif mode == 'exp':
                self.alpha = self.alpha_0 ** (n_iter/warmup_iters)
            elif mode == 'jump':
                self.alpha = 0

    def earlydecay(self,n_iter):
        mode = self.earlydecay_config['mode']
        earlydecay_start = self.earlydecay_config['earlydecay_start']
        earlydecay_end = self.earlydecay_config['earlydecay_end']

        if n_iter < earlydecay_start:
            return
        elif n_iter > earlydecay_start and n_iter < earlydecay_end:
            if mode == 'linear' :
                self.alpha = self.alpha_0 * ((earlydecay_end-n_iter)/(earlydecay_end-earlydecay_start))
            elif mode == 'exp':
                self.alpha = 0.001 * self.alpha_0 ** ((earlydecay_end-n_iter)/(earlydecay_end-earlydecay_start))
            elif mode == 'jump':
                self.alpha = 0
        elif n_iter >= earlydecay_end:
            self.alpha = 0

        
    def forward(self,x_student,x_teacher,gt,n_iter):
        if self.warmup_config:
            self.warmup(n_iter)
        if self.earlydecay_config:
            self.earlydecay(n_iter)

        if self.resize_config:
            x_student,x_teacher = self.resize(x_student,gt),self.resize(x_teacher,gt)
        if self.shuffle_config:
            x_student,x_teacher = self.shuffle(x_student,x_teacher,n_iter)
        if self.transform_config:
            x_student,x_teacher = self.transform(x_student),self.transform(x_teacher)

        x_student = F.log_softmax(x_student/self.tau,dim=-1)
        x_teacher = F.softmax(x_teacher/self.tau,dim=-1)
        
        loss = self.KLD(x_student,x_teacher)/(x_student.numel()/x_student.shape[-1])
        loss = self.alpha * loss
        return loss

class PDLoss(KLDLoss):
    def __init__(self):
        super().__init__()
        self.alpha_0 = 1
        self.alpha = 1
        self.tau = 1

        self.resize_config = {'mode':'bilinear','align_corners':False}
        self.shuffle_config = None
        self.transform_config = {'loss_type':'pixel'}
        self.warmup_config = None
        self.earlydecay_config = None

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

class CDLoss(KLDLoss):
    def __init__(self):
        super().__init__()
        self.alpha_0 = 1
        self.alpha = 1
        self.tau = 1

        self.resize_config = {'mode':'bilinear','align_corners':False}
        self.shuffle_config = None
        self.transform_config = {'loss_type':'channel','group_size':1}
        self.warmup_config = None
        self.earlydecay_config = None

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

class CGDLoss(KLDLoss):
    def __init__(self,group_size=10,alpha=3,tau=2):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = {'mode':'bilinear','align_corners':False}
        self.shuffle_config = {'interval':1000}
        self.transform_config = {'loss_type':'channel','group_size':group_size}
        self.warmup_config = None
        self.earlydecay_config = None

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

class CGDLossWS(KLDLoss):
    def __init__(self):
        super().__init__()
        self.alpha_0 = 3
        self.alpha = 3
        self.tau = 2

        self.resize_config = {'mode':'bilinear','align_corners':False}
        self.shuffle_config = {'interval':1000}
        self.transform_config = {'loss_type':'channel','group_size':10}
        self.warmup_config = {'mode':'linear','warmup_iters':2000}
        self.earlydecay_config = {'mode':'linear','earlydecay_start':110000,'earlydecay_end':120000}

        self.KLD = torch.nn.KLDivLoss(reduction='sum')

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = nn.KLDivLoss(reduction='sum')

    def _resize(self,x,x_t):
        x = F.interpolate(
            input=x,
            size=x_t.shape[2:],
            mode='bilinear',align_corners=False)
        return x

    def forward(self,x_student, x_teacher, gt,step):
        # x_student = self._resize(x_student,x_teacher)
        loss_AT = self.mse(x_student.mean(dim=1), x_teacher.mean(dim=1))

        x_student = F.log_softmax(x_student,dim=1)
        x_teacher = F.softmax(x_teacher,dim=1)

        loss_PD = self.KLD(x_student, x_teacher)/(x_student.numel()/x_student.shape[1])
        loss = loss_AT + loss_PD
        return loss

class IFVDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduce='mean')
        self.KLD = nn.KLDivLoss(reduction='sum')
    def resize(self,x,x_t):
        x = F.interpolate(
            input=x,
            size=x_t.shape[2:],
            mode='bilinear',align_corners=False)
        return x
    def forward(self, preds_S, preds_T, target,step):
        feat_S = preds_S
        feat_T = preds_T
        feat_T = self.resize(feat_T,feat_S)
        feat_T.detach()

        C = feat_T.shape[1]
        x_student = F.log_softmax(feat_S,dim=1)
        x_teacher = F.softmax(feat_T,dim=1)
        loss_PD = self.KLD(x_student, x_teacher)/(x_student.numel()/x_student.shape[1])

        size_f = (feat_S.shape[2], feat_S.shape[3])
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.float()).expand(feat_T.size())
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()
        for i in range(C):
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * ((mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * ((mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(-1).unsqueeze(-1)
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S)
        pcsim_feat_T = cos(feat_T, center_feat_T)

        loss_IFVD = 10*self.mse(pcsim_feat_S, pcsim_feat_T)

        loss = loss_IFVD + loss_PD
        return loss

    
        