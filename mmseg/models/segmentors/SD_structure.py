import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmseg.models.distillation.opts import Extractor,DistillationLoss_
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

@SEGMENTORS.register_module()
class SDModule_(BaseSegmentor):
    def __init__(self,
                 cfg, cfg_t,train_cfg,test_cfg,distillation,s_pretrain,t_pretrain):
        super(SDModule_, self).__init__()
        self.cfg_s = cfg
        self.cfg_t = cfg_t
        self.distillation = distillation
        if len(distillation.layers) == 0:
            self.use_teacher = False
        else:
            self.use_teacher = True
        
        self.teacher = builder.build_segmentor(
            cfg_t, train_cfg=train_cfg, test_cfg=test_cfg)
        self.teacher.load_state_dict(torch.load(
            t_pretrain)['state_dict'])
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = builder.build_segmentor(
            cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        self.student_init(strategy='use_pretrain',s_pretrain=s_pretrain,t_pretrain=t_pretrain)

        self.features = Extractor(self.student,self.teacher,distillation.layers)
        
        self.loss = DistillationLoss_(distillation = distillation,tau=1)
        self.align_corners = False
        self.test_mode = 'whole'

    def forward_train(self, img, img_metas, gt_semantic_seg):
        loss_dict = self.student(img, img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)
        if self.use_teacher:
            with torch.no_grad():
                _ = self.teacher(img, img_metas, return_loss=True, gt_semantic_seg=gt_semantic_seg)
            del _

        softs_fea = []
        preds_fea = []

        for i in range(len(self.features.teacher_features)):
            pred = self.features.student_features[i]
            soft = self.features.teacher_features[i]
            # print(soft[0].shape)
            # print(pred.shape)
            # print('------------------------------')
            softs_fea.append(soft)
            preds_fea.append(pred)
        # raise ValueError('____________________')
        loss_dict = self.loss(softs_fea, preds_fea, loss_dict)

        self.features.student_features = []
        self.features.teacher_features = []
        return loss_dict
    
    def student_init(self,strategy,s_pretrain=None,t_pretrain=None,distillation=None):
        if strategy == 'use_pretrain':# 使用预训练模型
            # 载入student的权重
            # 预训练模型的层名称没有‘backbone.’ 的前缀，因此在载入前需要增加前缀
            state_dict = torch.load(s_pretrain)
            new_keys = ['backbone.'+key for key in state_dict]
            d1 = dict( zip( list(state_dict.keys()), new_keys) )
            new_state_dict = {d1[oldK]: value for oldK, value in state_dict.items()}
            self.student.load_state_dict(new_state_dict,strict=False)
        elif strategy == 'use_teacher' :# 跳层初始化
            assert self.cfg_s['backbone']['embed_dim'] == self.cfg_t['backbone']['embed_dim']  # 需要维度一致

            state_dict = torch.load(t_pretrain)['state_dict']  # 载入teacher模型
            # student 和 teacher对应: 0->0 1->3 2->6 3->10 4->14 5->17
            mapping = {0:0,1:3,2:6,3:10,4:14,5:17}
            new_state_dict = OrderedDict()
            # print([k for k,v in state_dict.items()])
            for k,v in state_dict.items():
                if not k.startswith('backbone.layers.2'):
                    new_state_dict[k] = v
                elif str(k.split('.')[4]) in mapping.keys():
                    new_k = k.split('.')
                    new_k[4] = mapping[new_k[4]]
                    new_k = ''.join(new_k)

                    new_state_dict[new_k] = v
                    
            self.student.load_state_dict(new_state_dict,strict=False)
        else:
            raise ValueError('Wrong student init strategy')

    # def _parse_losses(self,losses):
    #     log_vars = OrderedDict()
    #     for loss_name, loss_value in losses.items():
    #         if isinstance(loss_value, torch.Tensor):
    #             log_vars[loss_name] = loss_value.mean()
    #         elif isinstance(loss_value, list):
    #             log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
    #         else:
    #             raise TypeError(
    #                 f'{loss_name} is not a tensor or list of tensors')
        
    #     return log_vars,self.distillation.parse_mode

    # def train_step(self, data_batch, optimizer, loss_name='all',backward=True, **kwargs):
    #     losses = self(**data_batch)
    #     log_vars,parse_mode = self._parse_losses(losses)

    #     outputs = dict(
    #         log_vars=log_vars,
    #         parse_mode=parse_mode,
    #         num_samples=len(data_batch['img'].data))

    #     loss = sum(_value for _key, _value in losses.items()
    #                         if 'loss' in _key)
    #     outputs['loss'] = loss
    #     return outputs
        

    def or_decomposition(self,u,v):
        if torch.count_nonzero(v) == 0:
            return u
        return u-(u.flatten()@v.flatten())/(v.flatten()@v.flatten())*v
    def get_grad(self,loss):
        loss.backward(retain_graph=True)
        grads = {}
        for name,param in self.student.named_parameters():
            if param.grad is None:
                grads[name] = torch.zeros(param.shape).cuda()
            else:
                grads[name] = param.grad.clone()
        self.student.zero_grad()
        return grads
    def flat_grad(self,grads):
        tmp = [v.flatten() for k,v in grads.items()]
        return torch.cat(tmp)
    def cos(self,u,v):
        return (u@v)/(torch.sqrt(v@v)*torch.sqrt(u@u))
    def mag(self,u):
        u = u.flatten()
        return torch.sqrt((u@u))
    def unit(self,u):
        if torch.count_nonzero(u) == 0:
            return u
        return u/self.mag(u)
    def set_grad(self,outputs):
        losses = outputs['log_vars']
        mode = outputs['parse_mode']
        loss_names = [k for k in losses if 'loss' in k]

        if mode == 'regular':
            loss = sum(_value for _key, _value in losses.items()
                    if 'loss' in _key)
            loss.backward()
        elif mode == 'PCGrad':
            g = {}
            g_pc = {}

            for i,loss_name in enumerate(loss_names):
                loss = losses[loss_name]
                loss.backward(retain_graph=True)

                g[loss_name] = {}
                g_pc[loss_name] = {}

                for name,param in self.student.named_parameters():
                    if param.grad is None:
                        g[loss_name][name] = torch.zeros(param.shape).cuda()
                    else:
                        g[loss_name][name] = copy.deepcopy(param.grad)
                    g_pc[loss_name][name] = g[loss_name][name]
                self.student.zero_grad()
                break

            for i in range(len(loss_names)):
                loss_i = loss_names[i]
                order = [i for i in range(len(loss_names))]
                random.shuffle(order)
                for j in order:
                    loss_j = loss_names[j]
                    if i == j:
                        continue

                    proj = 0
                    for k in g[loss_i]:
                        proj += torch.sum(g_pc[loss_i][k]*g[loss_j][k])
                    if proj.item() < 0:
                        mag = 0
                        for k in g[loss_i]:
                            mag += torch.sum(g[loss_j][k]*g[loss_j][k])

                        for k in g[loss_i]:
                            g_pc[loss_i][k] -= (proj / mag * g[loss_j][k])
            for name,param in self.student.named_parameters():
                for loss_name in g_pc:
                    param.grad += g_pc[loss_name][name]
        elif mode == 'PCGrad_decode':
            g = {}
            g_pc = {}

            for i,loss_name in enumerate(loss_names):
                loss = losses[loss_name]
                loss.backward(retain_graph=True)

                g[loss_name] = {}
                g_pc[loss_name] = {}

                for name,param in self.student.named_parameters():
                    if param.grad is None:
                        g[loss_name][name] = torch.zeros(param.shape).cuda()
                    else:
                        g[loss_name][name] = copy.deepcopy(param.grad)
                    g_pc[loss_name][name] = g[loss_name][name]
                self.student.zero_grad()


            for i in range(len(loss_names)):
                loss_i = loss_names[i]
                if loss_i == 'decode.loss_seg':
                    continue
                order = [i for i in range(len(loss_names))]
                random.shuffle(order)
                for j in order:
                    loss_j = loss_names[j]
                    if i == j:
                        continue

                    proj = 0
                    for k in g[loss_i]:
                        proj += torch.sum(g_pc[loss_i][k]*g[loss_j][k])
                    if proj.item() < 0:
                        mag = 0
                        for k in g[loss_i]:
                            mag += torch.sum(g[loss_j][k]*g[loss_j][k])

                        for k in g[loss_i]:
                            g_pc[loss_i][k] -= (proj / mag * g[loss_j][k])
            for name,param in self.student.named_parameters():
                for loss_name in g_pc:
                    param.grad += g_pc[loss_name][name]
        elif mode == 'SCKD':
            loss_names = [k for k in losses]

            decode_grads = torch.Tensor([]).cuda()
            decode_loss = losses['decode.loss_seg']
            decode_loss.backward(retain_graph=True)
            
            for name,param in self.student.named_parameters():
                if param.grad is None:
                    decode_grads = torch.cat([decode_grads,torch.zeros(param.shape).flatten().cuda()])
                else:
                    decode_grads = torch.cat([decode_grads,param.grad.flatten()])
            self.student.zero_grad()

            survive_losses = {}
            for loss_name in loss_names:
                if loss_name == 'decode.loss_seg' or 'loss' not in loss_name:
                    continue
                loss = losses[loss_name]
                loss.backward(retain_graph=True)
                loss_grads = torch.Tensor([]).cuda()

                for name,param in self.student.named_parameters():
                    if param.grad is None:
                        loss_grads = torch.cat([loss_grads,torch.zeros(param.shape).flatten().cuda()])
                    else:
                        loss_grads = torch.cat([loss_grads,param.grad.flatten()])
                if torch.sum(loss_grads * decode_grads) > 0:
                    survive_losses[loss_name] = loss

            loss = sum([v for k,v in survive_losses.items() if 'loss' in loss_name])
            loss += losses['decode.loss_seg']
            loss.backward()  
        elif mode == 'SCKD_param' :
            decode_grads = {}
            decode_loss = losses['decode.loss_seg']
            decode_loss.backward(retain_graph=True)
            for name,param in self.student.named_parameters():
                if param.grad is None:
                    continue
                else:
                    decode_grads[name] = param.grad
            self.student.zero_grad()

            distill_loss = sum([v for k,v in losses.items() if ('loss' in k) and ('decode' not in k) ])
            distill_loss.backward()
            for name,param in self.student.named_parameters():
                if param.grad is None:
                    continue
                elif name not in decode_grads:
                    continue 
                else:
                    decode_grad = decode_grads[name]
                    distill_grad = param.grad
                    if torch.sum(decode_grad * distill_grad).item() < 0:
                        param.grad = decode_grad
                    else:
                        param.grad = decode_grad + distill_grad
        elif mode == 'SCKD_sigmoid':
            grads = {}

            decode_grads = torch.Tensor([]).cuda()
            decode_loss = losses['decode.loss_seg']
            decode_loss.backward(retain_graph=True)
            
            for name,param in self.student.named_parameters():
                if param.grad is None:
                    decode_grads = torch.cat([decode_grads,torch.zeros(param.shape).flatten().cuda()])
                    grads['name'] == torch.zeros(param.shape).cuda()
                else:
                    decode_grads = torch.cat([decode_grads,param.grad.flatten()])
                    grads['name'] == param.grad
            self.student.zero_grad()

            survive_losses = {}
            for loss_name in loss_names:
                if loss_name == 'decode.loss_seg' or 'loss' not in loss_name:
                    continue
                loss = losses[loss_name]
                loss.backward(retain_graph=True)
                loss_grads = torch.Tensor([]).cuda()

                for name,param in self.student.named_parameters():
                    if param.grad is None:
                        loss_grads = torch.cat([loss_grads,torch.zeros(param.shape).flatten().cuda()])
                    else:
                        loss_grads = torch.cat([loss_grads,param.grad.flatten()])
                
                cos = torch.sum(loss_grads * decode_grads)/torch.sqrt((torch.sum(loss_grads * loss_grads)*\
                                                            torch.sum(decode_grads * decode_grads)))
                weight = F.sigmoid(cos)
                if weight > 0:
                    for name,param in self.student.named_parameters():
                        if param.grad is not None:
                            grads['name'] += param.grad
                self.student.zero_grad()
            for name,param in self.student.named_parameters():
                param.grad = grads[name]
        elif mode == 'dropout':
            p = 0.5
            survive_losses = {}
            for loss_name in loss_names:
                if 'loss' not in loss_name or 'decode' in loss_name:
                    continue
                if random.uniform(0,1) > p:
                    survive_losses[loss_name] = losses[loss_name]
            loss = sum((1/(1-p))*v for k,v in survive_losses.items() )
            loss += losses['decode.loss_seg']
            loss.backward()
        elif mode == 'grad_scale':
            decode_loss = losses['decode.loss_seg']
            soft_loss = losses['loss_decode_head.conv_seg']
            aux_loss = losses['aux.loss_seg']

            decode_grad = self.get_grad(decode_loss)
            soft_grad = self.get_grad(soft_loss)
            aux_grad = self.get_grad(aux_loss)

            decode_grad_flat = self.flat_grad(decode_grad)
            soft_grad_flat = self.flat_grad(soft_grad)
            aux_grad_flat = self.flat_grad(aux_grad)

            losses['mag_decode'] = self.mag(decode_grad_flat)
            losses['mag_soft'] = self.mag(soft_grad_flat)
            losses['mag_aux'] = self.mag(aux_grad_flat)
            w_decode = losses['mag_decode']/(losses['mag_decode']+losses['mag_soft']+losses['mag_aux'])
            w_soft = losses['mag_soft']/(losses['mag_decode']+losses['mag_soft']+losses['mag_aux'])
            w_aux = losses['mag_aux']/(losses['mag_decode']+losses['mag_soft']+losses['mag_aux'])

            losses['cos_decode_soft'] = self.cos(decode_grad_flat,soft_grad_flat)
            losses['cos_aux_soft'] = self.cos(soft_grad_flat,aux_grad_flat)
            losses['cos_decode_aux'] = self.cos(decode_grad_flat,aux_grad_flat)

            losses['mag_soft_or'] = self.mag(self.or_decomposition(soft_grad_flat,decode_grad_flat))
            losses['mag_aux_or'] = self.mag(self.or_decomposition(\
            self.or_decomposition(aux_grad_flat,decode_grad_flat),self.or_decomposition(soft_grad_flat,decode_grad_flat)\
            ))

            for name,param in self.student.named_parameters():
                decode = decode_grad[name]
                aux = aux_grad[name]
                soft = aux_grad[name]

                param.grad = torch.zeros(param.shape).cuda()

                if torch.count_nonzero(decode) > 0:
                    param.grad += w_decode*decode
                    print(1)
                else:
                    print(2)
                if torch.count_nonzero(aux) > 0:
                    param.grad += w_aux*aux
                if torch.count_nonzero(soft) > 0:
                    param.grad += w_soft*soft
        elif mode == 'adam':

            beta_1 = 0.5
            beta_2 = 0.5
            eps = 1e-8

            decode_loss = losses['decode.loss_seg']
            soft_loss = losses['loss_decode_head.conv_seg']
            aux_loss = losses['aux.loss_seg']

            names = ['decode','soft','aux']
            for i,l in enumerate([decode_loss,soft_loss,aux_loss]):
                l.backward(retain_graph=True)
                for name,param in self.student.named_parameters(): 
                    if param.grad is None:
                        g_t = torch.zeros(param.shape).cuda()
                    else:
                        g_t = param.grad
                    
                    self.m['m_'+names[i]][name] = beta_1*self.m['m_'+names[i]][name] + (1-beta_1)*g_t
                    self.v['v_'+names[i]][name] = beta_2*self.v['v_'+names[i]][name] + (1-beta_2)*(g_t**2)
                self.student.zero_grad()
            
            for name,param in self.student.named_parameters():
                param.grad = self.m['m_decode'][name] + \
                                self.m['m_soft'][name] +\
                                self.m['m_aux'][name]

            for k in self.m:
                # print(self.flat_grad(self.m[k]).unsqueeze(0).mean())
                outputs['log_vars'].update({k:self.flat_grad(self.m[k]).unsqueeze(0).mean()})
            for k in self.v:
                outputs['log_vars'].update({k:self.flat_grad(self.v[k]).unsqueeze(0).mean()})    
        elif mode == 'grad_inspect':
            mags = {}
            grads = {}
            for loss_name in losses:
                if 'loss' not in loss_name:
                    continue
                
                param_nums['nums_'+loss_name] = 0
                flat_grads = torch.Tensor([]).cuda()
                loss = losses[loss_name]
                loss.backward(retain_graph=True)
                for name,param in self.student.named_parameters():
                    if param.grad is None:
                        flat_grads = torch.cat([flat_grads,torch.zeros(param.shape).flatten().cuda()])
                    else:
                        flat_grads = torch.cat([flat_grads,param.grad.flatten()])
                        if not torch.count_nonzero(param.grad.flatten()) == 0:
                            self.param_nums['nums_'+loss_name] += param.grad.flatten().shape[0]

                        if name in grads:
                            grads[name] += param.grad
                        else:
                            grads[name] = param.grad

                mag = self.get_mag(flat_grads)
                mags['mag_'+loss_name] = mag/param_nums['nums_'+loss_name]

                self.student.zero_grad()
        
            outputs['log_vars'].update(mags)
            outputs['log_vars'].update(param_nums)

            for name,param in self.student.named_parameters():
                param.grad = grads[name]
                # param.grad = grads[name]/param_nums['nums_'+loss_name] 
                # if 'decode' not in name and 'aux' in name:
                #     param.grad *= 5
        else:
            raise NotImplementedError()
    def get_mag(self,grad):
        return torch.sqrt(torch.sum(grad*grad))

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pad_img = crop_img.new_zeros(
                    (crop_img.size(0), crop_img.size(1), h_crop, w_crop))
                pad_img[:, :, :y2 - y1, :x2 - x1] = crop_img
                pad_seg_logit = self.s_net.encode_decode(pad_img, img_meta)
                preds[:, :, y1:y2,
                x1:x2] += pad_seg_logit[:, :, :y2 - y1, :x2 - x1]
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return preds

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

        assert self.test_mode in ['slide', 'whole']
        img_meta = img_meta.data[0]
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        flip_direction = img_meta[0]['flip_direction']
        if flip:
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
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