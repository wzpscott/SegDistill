from functools import partial
from .losses import *
import re
from collections import OrderedDict

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from mmseg.ops import resize


class Extractor(nn.Module):
    def __init__(self, student, teacher, distillation):
        '''
        student: student model
        teacher: teacher model
        distillation: 
            a list of dicts containing layer information you want to apply distillation .
            a dict contains:
            student_layer: the name of student layer
            teacher_layer: the name of teacher layer
            loss_name: the name of loss function you want to apply to the ouputs of those two layers.
            loss_config: a dict containing config for the loss function(weight, temperature for SoftMax etc)
            channel_nums: a tuple of (student_channel_num,teacher_channel_num). optional
                        If specified, using a 1x1 conv layer to upsample student layer's channel num 
                        when student layer and teacher layer 's outputs have different channel shapes
        '''
        super().__init__()
        self.teacher_features = {}
        self.student_features = {}

        for i in range(len(distillation)):
            student_layer, teacher_layer = distillation[i]['student_layer'], distillation[i]['teacher_layer']

            for name, module in teacher.named_modules():
                if name == teacher_layer:
                    module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='teacher'))
                    print(f'teacher_layer :{teacher_layer} hooked!!!!')

            for name, module in student.named_modules():
                if name == student_layer:
                    module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='student'))
                    print(f'student_layer :{student_layer} hooked!!!!')

    def hook_fn_forward(self, module, input, output, name, type):
        if self.training == True:
            if type == 'student':
                self.student_features[name] = output
            if type == 'teacher':
                self.teacher_features[name] = output


class DistillationLoss(nn.Module):
    def __init__(self,distillation):
        super().__init__()

        for i in range(len(distillation)):
            loss_name = distillation[i]['loss_name'] 
            loss_config =  distillation[i]['loss_config']  
            criterion = eval(loss_name)(**loss_config) 
            distillation[i]['criterion'] = criterion

            if 'channel_nums' in distillation[i]:
                student_channel_num,teacher_channel_num = distillation[i]['channel_nums']
                distillation[i]['upsampler'] = Conv1d(student_channel_num,teacher_channel_num,kernel_size=1,dim=2).cuda()

        self.distillation = distillation
    def forward(self,student_features,teacher_features,gt_semantic_seg):
        distillation_losses = {}
        for i in range(len(self.distillation)):
            student_layer, teacher_layer = self.distillation[i]['student_layer'], self.distillation[i]['teacher_layer']
            x_student, x_teacher = student_features[student_layer], teacher_features[teacher_layer]
            if 'upsampler' in self.distillation[i]:
                x_student = self.distillation[i]['upsampler'](x_student)

            criterion = self.distillation[i]['criterion']
            loss = criterion(x_student, x_teacher,gt_semantic_seg)

            if 'inspect_mode' in self.distillation[i]:
                loss = loss.detach()

            distillation_losses[f'loss_{student_layer}<->{teacher_layer}'] = loss

        return distillation_losses


class Conv1d(nn.Module):
    # conv1d with optional dim
    def __init__(self,c_in,c_out,kernel_size,dim):
        super().__init__()
        self.conv1d = nn.Conv1d(c_in,c_out,kernel_size)
        self.dim = dim
    def forward(self,x):
        x = x.transpose(1,self.dim)
        x = self.conv1d(x)
        x = x.transpose(1,self.dim)
        return x
    
        
