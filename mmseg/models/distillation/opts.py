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

        teacher_layers = []
        student_layers = []

        for i in range(len(distillation)):
            student_layer, teacher_layer = distillation[i]['student_layer'], distillation[i]['teacher_layer']
            if isinstance(student_layer,list):
                student_layers += student_layer
            else:
                student_layers.append(student_layer)

            if isinstance(teacher_layer,list):
                teacher_layers += teacher_layer
            else:
                teacher_layers.append(teacher_layer)

        for name, module in teacher.named_modules():
            if name in teacher_layers:
                module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='teacher'))
                print(f'teacher_layer :{name} hooked!!!!')

        for name, module in student.named_modules():
            if name in student_layers:
                module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='student'))
                print(f'student_layer :{name} hooked!!!!')


        # for name, module in teacher.named_modules():
        #     print(name)
        # raise ValueError('dfs')


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
            loss_config = distillation[i]['loss_config'] 
            if isinstance(loss_config,tuple):
                loss_config = loss_config[0]
            criterion = eval(loss_name)(**loss_config) 
            distillation[i]['criterion'] = criterion

        self.distillation = distillation
    def forward(self,student_features,teacher_features,gt_semantic_seg,step,student,teacher):
        distillation_losses = {}
        for i in range(len(self.distillation)):
            student_layer, teacher_layer = self.distillation[i]['student_layer'], self.distillation[i]['teacher_layer']
            if isinstance(student_layer,list):
                attn_student,v_student = student_features[student_layer[0]],student_features[student_layer[1]]
                attn_teacher,v_teacher = teacher_features[teacher_layer[0]],teacher_features[teacher_layer[1]]
                criterion = self.distillation[i]['criterion']
                loss = criterion(attn_student,v_student,attn_teacher,v_teacher,student,teacher,gt_semantic_seg,step)
                n = self.distillation[i]['loss_name']
                loss_name = f'loss_{student_layer[0]}<->{teacher_layer}_{n}'
                distillation_losses[loss_name] = loss
            else:
                x_student, x_teacher = student_features[student_layer], teacher_features[teacher_layer]

                criterion = self.distillation[i]['criterion']
                loss = criterion(x_student,x_teacher,gt_semantic_seg,step)

                try:
                    loss_info = self.distillation[i]['loss_config']['transform_config']
                except:
                    loss_info = 'other'
                loss_name = f'loss_{student_layer}<->{teacher_layer}_{loss_info}'
                distillation_losses[loss_name] = loss

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
    


class ExtractorMT(nn.Module):
    def __init__(self, student, teachers, distillation):
        super().__init__()

        self.num_teacher = len(teachers)

        self.teacher_features = {}
        self.student_features = {}

        teacher_layers = []
        student_layers = []

        for i in range(len(distillation)):
            student_layer, teacher_layer = distillation[i]['student_layer'], distillation[i]['teacher_layer']
            if isinstance(student_layer,list):
                student_layers += student_layer
            else:
                student_layers.append(student_layer)

            if isinstance(teacher_layer,list):
                teacher_layers += teacher_layer
            else:
                teacher_layers.append(teacher_layer)

        for i,teacher in enumerate(teachers):
            for name, module in teacher.named_modules():
                if name in teacher_layers:
                    module.register_forward_hook(partial(self.hook_fn_forward, name=name+str(i), type='teacher'))
                    print(f'teacher_layer :{name} hooked!!!!')

        for name, module in student.named_modules():
            if name in student_layers:
                module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='student'))
                print(f'student_layer :{name} hooked!!!!')


    def hook_fn_forward(self, module, input, output, name, type):
        if self.training == True:
            if type == 'student':
                self.student_features[name] = output
            if type == 'teacher':
                self.teacher_features[name] = output
        
class DistillationLossMT(nn.Module):
    def __init__(self,distillation):
        super().__init__()
        for i in range(len(distillation)):
            loss_name = distillation[i]['loss_name'] 
            loss_config = distillation[i]['loss_config'] 
            if isinstance(loss_config,tuple):
                loss_config = loss_config[0]
            criterion = eval(loss_name)(**loss_config) 
            distillation[i]['criterion'] = criterion

        self.distillation = distillation
    def forward(self,student_features,teacher_features,gt_semantic_seg,step):
        distillation_losses = {}
        if len(teacher_features) != len(self.distillation):
            distillation = self.distillation[0]

            student_layer = distillation['student_layer']
            x_student = student_features[student_layer]

            # x_teacher = torch.cat([teacher_features[i].unsqueeze(0) for i in teacher_features],dim=0)
            x_teacher = [teacher_features[i] for i in teacher_features]
            criterion = distillation['criterion']
            loss = criterion(x_student,x_teacher,gt_semantic_seg,step)
            loss_name = f'loss_random'
            distillation_losses[loss_name] = loss
        else:
            for i,distillation in enumerate(self.distillation):
                student_layer = distillation['student_layer']
                x_student = student_features[student_layer]
                teacher_layer = distillation['teacher_layer']+str(i)
                x_teacher = teacher_features[teacher_layer]

                criterion = distillation['criterion']
                loss = criterion(x_student,x_teacher,gt_semantic_seg,step)
                loss_name = f'loss_{student_layer}<->{teacher_layer}_{i}'
                distillation_losses[loss_name] = loss

        return distillation_losses