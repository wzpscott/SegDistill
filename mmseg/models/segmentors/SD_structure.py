import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmseg.models.distillation.opts import *
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
import torch
from collections import OrderedDict
import numpy as np
import random
import copy
from ..distillation.losses import *


@SEGMENTORS.register_module()
class SDModule(BaseSegmentor):
    def __init__(self,
                 cfg_s, cfg_t,train_cfg,test_cfg,distillation,s_pretrain=None,t_pretrain=None):
        super().__init__()
        self.cfg_s = cfg_s
        self.cfg_t = cfg_t
        self.distillation = distillation

        self.student = builder.build_segmentor(
            cfg_s, train_cfg=train_cfg, test_cfg=test_cfg)
        if s_pretrain:
           self.student.load_state_dict(torch.load(
                s_pretrain)['state_dict'],strict=True) 

        self.teacher = builder.build_segmentor(
                    cfg_t, train_cfg=train_cfg, test_cfg=test_cfg)
        self.teacher.load_state_dict(torch.load(
                t_pretrain)['state_dict'],strict=True)
        self.teacher.eval()

        if len(distillation)>0:
            self.log_grad = True if 'log_grad' in distillation[0] else False
        else:
            self.log_grad = False
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.extractor = Extractor(self.student,self.teacher,self.distillation)
        self.distillation_loss = DistillationLoss(self.distillation)

        self.align_corners = False

        self.test_cfg = test_cfg
        self.test_mode = 'whole'

        self.cnt = 0


    def forward_train(self, img, img_metas, gt_semantic_seg):
        self.cnt += 1
        loss_dict = self.student(img, img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)
        if self.distillation != []:
            with torch.no_grad():
                _ = self.teacher(img, img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)
                del _
            student_features,teacher_features = self.extractor.student_features,self.extractor.teacher_features
            
            # import pickle as pkl
            # with open('/home/mist/SegformerDistillation/work_dirs/visualization/student_logits','wb') as f:
            #     pkl.dump(student_features,f)
            # with open('/home/mist/SegformerDistillation/work_dirs/visualization/teacher_logits','wb') as f:
            #     pkl.dump(teacher_features,f)
            # raise ValueError('dddd')
            distillation_loss_dict = self.distillation_loss(student_features,teacher_features,gt_semantic_seg,self.cnt,\
                                    self.student,self.teacher)

            loss_dict.update(distillation_loss_dict)

        
        return loss_dict

    def get_grads(self,loss):
        loss.backward(retain_graph=True)
        params = self.student.parameters()
        # print([i for i in params])
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        params = [i.grad.flatten().clone() for i in params]
        g = torch.cat(params)

        # for param in self.student.named_parameters():
        #     # print(param[0])
        #     g = []
        #     if 'decode_head.linear_pred' in param[0]:
        #         g.append(param[1].grad.flatten().clone())

        self.student.zero_grad()
        return g

    def _parse_losses(self,losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        if self.log_grad:
            for _key, _value in log_vars.items():
                if 'loss_seg' in _key:
                    loss_seg = log_vars[_key]
                elif 'channel' in _key:
                    loss_distill = log_vars[_key]
            grad_seg = self.get_grads(loss_seg)
            grad_distill = self.get_grads(loss_distill)
            cos = torch.sum(grad_seg*grad_distill)/(torch.norm(grad_seg)*torch.norm(grad_distill))
            deg = torch.acos(cos)
            log_vars['deg'] = deg*180/3.1416

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        seg_logit = self.student.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


@SEGMENTORS.register_module()
class SDModuleMT(BaseSegmentor):
    def __init__(self,
                 cfg_s, cfg_t,train_cfg,test_cfg,distillation,s_pretrain=None,t_pretrain=None):
        super().__init__()
        self.cfg_s = cfg_s
        self.cfg_t = cfg_t
        self.distillation = distillation

        self.student = builder.build_segmentor(
            cfg_s, train_cfg=train_cfg, test_cfg=test_cfg)
        if s_pretrain:
           self.student.load_state_dict(torch.load(
                s_pretrain)['state_dict'],strict=True) 

        self.teachers = []
        for i in range(len(cfg_t)):
            self.teachers.append(builder.build_segmentor(
                        cfg_t[i], train_cfg=train_cfg, test_cfg=test_cfg).cuda())
            self.teachers[i].load_state_dict(torch.load(
                    t_pretrain[i])['state_dict'],strict=True)
            self.teachers[i].eval()
            for param in self.teachers[i].parameters():
                param.requires_grad = False

        self.extractor = ExtractorMT(self.student,self.teachers,self.distillation)
        self.distillation_loss = DistillationLossMT(self.distillation)

        self.align_corners = False

        self.test_cfg = test_cfg
        self.test_mode = 'whole'

        self.cnt = 0


    def forward_train(self, img, img_metas, gt_semantic_seg):
        self.cnt += 1
        loss_dict = self.student(img, img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)
        for teacher in self.teachers:
            with torch.no_grad():
                _ = teacher(img, img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)
                del _
        student_features,teacher_features = self.extractor.student_features,self.extractor.teacher_features

        distillation_loss_dict = self.distillation_loss(student_features,teacher_features,gt_semantic_seg,self.cnt)

        loss_dict.update(distillation_loss_dict)

        
        return loss_dict

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        seg_logit = self.student.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
