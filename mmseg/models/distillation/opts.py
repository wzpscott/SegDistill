from functools import partial
from .losses import *
import re
from collections import OrderedDict

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from mmseg.ops import resize

class Extractor(nn.Module):
    def __init__(self, student, teacher, layers):
        super().__init__()

        self.teacher_features = []
        self.student_features = []
        self.channel_dims = []  # student 和 teacher 被提取层的输出通道数
        self.total_dims = []  # student 和 teacher 被提取层的输出维数

        for i,(student_layer,teacher_layer,channel_dim,total_dim) in enumerate(layers):
            self.channel_dims.append(channel_dim)
            self.total_dims.append(total_dim)

            if not isinstance(teacher_layer,list):
                teacher_layer = [teacher_layer]

            for name, module in teacher.named_modules():
                if name in teacher_layer:
                    module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='teacher',layer_num=i))
                    print(f'teacher_layer :{teacher_layer} hooked!!!!')

            for name, module in student.named_modules():
                if name == student_layer:
                    module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='student'))
                    print(f'student_layer :{student_layer} hooked!!!!')


    def hook_fn_forward(self, module, input, output, name, type,layer_num=None):
        if self.training == True:
            if 'fc' in name:
                output = output.permute(0,2,1)
            # if 'ATTN' in name: #[B,num_head,WH,WH]
            #     output = output.permute(0,1,3,2)
            #     B,num_head,WH,_ = output.shape
            #     output = output.reshape(B*num_head,WH,_)
            #     print(B,num_head,WH,_)

            if type == 'student':
                self.student_features.append(output)
            if type == 'teacher':
                # if len(self.teacher_features)>layer_num:
                #     self.teacher_features[layer_num].append(output)
                # else:
                #     self.teacher_features.append([output])
                self.teacher_features.append([output])


class ff(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()

        # self.ff = nn.Sequential(
        #     nn.Conv1d(input_size,input_size, kernel_size=1, stride=1, padding=0),
        #     nn.GELU(),
        #     nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0),
        #     nn.GELU(),
        #     nn.Conv1d(output_size,output_size, kernel_size=1, stride=1, padding=0),
        # )
        
        self.ff = nn.Sequential(
            nn.Linear(input_size,input_size),
            nn.GELU(),
            nn.Linear(input_size,output_size),
            nn.GELU(),
            nn.Linear(output_size,output_size),
        )
    def forward(self,x):
        x = self.ff(x)
        return x

class conv1d(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.conv = nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        return x

class AttnAdaptor(nn.Module):
    def __init__(self, student_size, teacher_size, teacher_num):
        super().__init__()
        Wt = nn.ModuleList()
        for _ in range(teacher_num):
            Wt.append(conv1d(teacher_size, teacher_size))
        self.Wt = Wt
        self.Ws = conv1d(student_size, teacher_size)

        self.KL = nn.KLDivLoss(reduction='none')
    def forward(self, x_student, x_teachers):
        b, WH, C_t = x_teachers[0].shape
        teacher_num = len(x_teachers)
        C_s = x_student.shape[2]
        T = 1

        for i in range(len(x_teachers)):
            x_teachers[i] = self.Wt[i](x_teachers[i])  # x_teachers[i]:[b,WH,C_t]

        
        x_teachers = torch.stack(x_teachers, dim=0)  # x_teachers:[teacher_num,b,WH,C_t]
        x_teachers = F.normalize(x_teachers,dim=2)
        # print('x_teachers.shape',x_teachers.shape)
        x_student = self.Ws(x_student)  # x_student:[b,WH,C_t]
        x_student = F.normalize(x_student,dim=1)
        x_student = x_student.unsqueeze(0)  # x_student:[1,b,WH,C_t]
        # print('x_student.shape',x_student.shape)
        # x_student = x_student.permute(1,0,2).reshape(WH,b*C_s) # x_student:[WH,b*C_s]
        # x_student = F.softmax(x_student,dim=0).mean(dim=1).unsqueeze(0) # x_student:[1,WH]
        # x_student = x_student.mean(dim=1).unsqueeze(0) # x_student:[1,WH]

        attn = (x_teachers * x_student).reshape(teacher_num,-1)
        # print(attn.shape)
        attn = attn.sum(dim=1)  # attn:[teacher_num]
        # print('attn_before',attn)
        attn = F.softmax(attn / T, dim=0).unsqueeze(1).unsqueeze(1).unsqueeze(1)  # attn:[teacher_num,1,1,1]
        # print('attn:',attn)

        x_teacher = (attn * x_teachers).sum(dim=0) # x_teacher:[1,b,WH,C_t]
        # print('x_teacher.shape', x_teacher.shape)
        # x_student = x_student.permute(1,0,2)
        # print('x_teacher',x_teacher)
        return x_student.squeeze(0), x_teacher, attn.reshape(teacher_num)

class Adaptor(nn.Module):
    def __init__(self,input_size,output_size,total_dim):
        super().__init__()
        self.total_dim = total_dim

        # ff = nn.Conv1d(input_size,input_size, kernel_size=1, stride=1, padding=0)

        if total_dim == 3:
            self.ff = nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0)
        elif total_dim == 4:
            # [b,num_head,WH,WH']
            self.ff1 = nn.Linear(256,256)
            self.ff2 = nn.Conv2d(input_size,output_size, kernel_size=1, stride=1, padding=0)

    def forward(self,x,x_teacher,gt_semantic_seg):
        if self.total_dim == 4:
            x = self.ff1(x)
            x = self.ff2(x)

        # if self.total_dim == 2:
        #     x = x
        # elif self.total_dim == 3:
        #     # x = x.permute(0,2,1)
        #     x = self.ff(x)
        #     # x = x.permute(0,2,1)
        # elif self.total_dim == 4:
        #     x = self.ff(x)
        # else:
        #     raise ValueError('wrong total_dim')
        return x,x_teacher,None

class LogitsAdaptor(nn.Module):
    def __init__(self,input_size,output_size,weights=None):
        super().__init__()
        self.ff = nn.Conv2d(input_size,output_size, kernel_size=1, stride=1, padding=0)

        if weights:
            self.weights = weights
        else:
            self.weights = [1,1,1,1,1]

    def forward(self,x_student,x_teacher,x_gt):

        x_student = resize(
            input=x_student,
            size=x_gt.shape[2:],
            mode='bilinear',
            align_corners=False)
        # x_student = self.ff(x_student)
        x_teacher = resize(
            input=x_teacher,
            size=x_gt.shape[2:],
            mode='bilinear',
            align_corners=False)  # [b,C,W,H]  

        x_gt = x_gt.squeeze(1)
        b, C, W, H = x_teacher.shape
        mask = torch.zeros(x_gt.shape).cuda()

        bg_mask = (x_gt == 255).detach()

        student_pd = torch.argmax(x_student,dim=1)
        teacher_pd = torch.argmax(x_teacher,dim=1)

        Sr = (student_pd == x_gt).detach() # student right
        Tr = (teacher_pd == x_gt).detach() # teacher right
        SrTr_mask = Sr & Tr
        SfTr_mask = (~Sr) & Tr
        SfTf_mask = (~Sr) & (~Tr) & (~bg_mask)
        SrTf_mask = Sr & (~Tr)

        masks = [bg_mask,SrTr_mask,SfTr_mask,SfTf_mask,SrTf_mask]
        for i in range(5):
            mask += self.weights[i] * masks[i]

        return x_student,x_teacher,mask

class DistillationLoss_(nn.Module):
    def __init__(self,distillation,tau):
        super().__init__()
        if 'T' in distillation:
            self.T = distillation['T']
        else:
            self.T = 1
        if 'weight' in distillation:
            self.weight = distillation['weight']
        else:
            self.weight = 1
        print('weight:', self.weight)
        print('T:', self.T)
        if distillation['loss_func'] == 'kl':
            self.kd_loss = KLdiv(self.T)
        elif distillation['loss_func'] == 'ca':
            self.kd_loss = CriterionChannelAwareLoss(self.T)
        elif distillation['loss_func'] == 'group':
            self.kd_loss = CriterionChannelAwareLossGroup(self.T)
        
        self.logits_weights = distillation['logits_weights']
        self.adaptors = nn.ModuleList()

        # self.conv_t_t = nn.ConvTranspose2d(150, 150, 4, stride=4)
        self.conv = nn.Conv2d(150,150,kernel_size=1)
        

        layers = distillation['layers']
        self.use_attn = distillation['use_attn']
        for student_layer,teacher_layers,channel_dim,dim in layers:
            student_dim,teacher_dim = channel_dim
            if self.use_attn:
                self.adaptors.append(AttnAdaptor(student_dim,teacher_dim,len(teacher_layers)))
            elif 'linear_pred' in student_layer:
                self.adaptors.append(LogitsAdaptor(student_dim,teacher_dim,self.logits_weights))
                print(f'add an logits adaptor of shape {student_dim} to {teacher_dim} in {student_layer}')
            else:
                self.adaptors.append(Adaptor(student_dim,teacher_dim,dim))
                print(f'add an adaptor of shape {student_dim} to {teacher_dim} in {student_layer}')

        self.layers = [student_name for student_name,_,_,_ in layers]
        
        # add gradients to weight of each layer's loss
        self.strategy = distillation['weights_init_strategy']
        if self.strategy=='equal':
            weights = [self.weight for i in range(len(layers))]
            weights = nn.Parameter(torch.Tensor(weights),requires_grad=False)
            self.weights = weights
        elif self.strategy=='self_adjust':
            weights = nn.Parameter(torch.Tensor([1 for i in range(3)]),requires_grad=True)
            self.weights = weights
        else:
            raise ValueError('Wrong weights init strategy')

        self.ce = nn.CrossEntropyLoss(reduction='none',ignore_index=255)

    def forward(self, soft, pred, losses,gt_semantic_seg=None):
        for i in range(len(pred)):
            adaptor = self.adaptors[i]

            if self.use_attn:
                pred[i], soft[i], attn = adaptor(pred[i], soft[i])
                for j in range(attn.shape[0]):
                    losses.update({'attn' + str(i) + 'layer' + str(j): attn[j].clone()})
            else:
                pred[i],soft[i],mask = adaptor(pred[i],soft[i][0],gt_semantic_seg)

            if self.strategy == 'equal':
                loss = self.weights[i] * self.kd_loss(pred[i], soft[i], mask)
                name = self.layers[i]
                losses.update({'loss_' + name: loss})
                if 'pred' in name:
                    decode_loss = (losses['decode.loss_seg']*mask).mean()
                    losses.update({'decode.loss_seg': decode_loss})
            elif self.strategy == 'self_adjust':
                loss = (1 / (self.weights[0] ** 2)) * \
                        self.kd_loss(pred[i], soft[i]) \
                        + torch.log(self.weights[0])
                name = self.layers[i]
                losses.update({'loss_' + name: loss})
                losses.update({'weight_' + name: self.weights[0]})

        if self.strategy == 'equal':
            pass
        elif self.strategy == 'self_adjust':
            losses['decode.loss_seg'] = (1 / (self.weights[1] ** 2)) * losses['decode.loss_seg'] + torch.log(
                self.weights[1])
            losses['aux.loss_seg'] = (1 / (self.weights[2] ** 2)) * losses['aux.loss_seg'] + torch.log(self.weights[2])

            losses.update({'weight_' + 'decode.loss_seg': self.weights[1]})
            losses.update({'weight_' + 'aux.loss_seg': self.weights[2]})
        return losses