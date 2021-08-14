from .losses import *
from .opts import DistillationLoss
from mmseg.models.distillation.opts import FeatureExtractor

class Sd_model(nn.Module):
    """
    Main class for Distillation
    """

    def __init__(self, cfg, cfg_t, s_net, t_net, s_feat, t_feat):
        super(Sd_model, self).__init__()

        self.teacher = t_net
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = s_net

        self.s_fea = s_feat
        self.t_fea = t_feat

        # for param in self.t_fea.parameters():
        #     param.requires_grad = False

        self.cfg_s = cfg
        self.loss = DistillationLoss(s_cfg=cfg, t_cfg=cfg_t)
        
    def train_step(self, batched_inputs, optimizer, **kwargs):
        loss_dict = self.student(img_metas=batched_inputs['img_metas'],img=batched_inputs['img'],gt_semantic_seg=batched_inputs['gt_semantic_seg'])
        with torch.no_grad():
            _ = self.teacher(img_metas=batched_inputs['img_metas'],img=batched_inputs['img'],gt_semantic_seg=batched_inputs['gt_semantic_seg'])
        del _

        softs_logits = []
        preds_logits = []
        softs_mask = []
        preds_mask = []
        softs_fea = []
        preds_fea = []

        for k in self.t_fea.feat:
            if 'logits' in k:
                ##apply distillation on the cls logits map
                for ind, soft in enumerate(self.t_fea.feat[k]):
                    softs_logits.append(soft)
                    preds_logits.append(self.s_fea.feat[k][ind])
            elif 'mask' in k:
                ##apply distillation on the instance mask
                for ind, soft in enumerate(self.t_fea.feat[k]):
                    softs_mask.append(soft)
                    preds_mask.append(self.s_fea.feat[k][ind])
            else:
                ##apply distillation on the fpn features
                for ind, soft in enumerate(self.t_fea.feat[k]):
                    softs_fea.append(soft)
                    preds_fea.append(self.s_fea.feat[k][ind])
            # else:
            #     print('Not Implemented')
            #     ##apply distillation on user define locations,
        if "off" not in self.cfg_s.distillation['logits']:
            logits_loss = self.loss(softs_logits, preds_logits, self.cfg_s.distillation, 'logits')
            loss_dict.update(logits_loss)
            del logits_loss
        # if "off" not in self.cfg_s.DISTILLATION.LOSS.MASK:
        #     mask_loss = self.loss(softs_mask, preds_mask, self.cfg_s.distillation, 'mask')
        #     loss_dict.update(mask_loss)
        #     del mask_loss
        if "off" not in self.cfg_s.distillation['fea']:
            FEA_loss = self.loss(softs_fea, preds_fea, self.cfg_s.distillation, 'fea')
            loss_dict.update(FEA_loss)
            del FEA_loss
        self.t_fea.feat = {}
        self.s_fea.feat = {}
        log_vars = [i for i in loss_dict if i != 'loss_CA_logits']
        loss_dict['loss'] = loss_dict['decode.loss_seg']
        self.get_log_info(loss_dict,log_vars,batched_inputs['img'].shape[0])

        return loss_dict

    def get_log_info(self,loss_dict,log_vars,num_samples):
        loss_dict['num_samples'] = num_samples
        loss_dict['log_vars'] = {}
        for var in log_vars:
            loss_dict['log_vars'][var] = loss_dict[var].detach().cpu()


    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

# class Sd_model(nn.Module):
#     """
#     Main class for Distillation
#     """

#     def __init__(self, cfg, cfg_t, s_net, t_net):
#         super(Sd_model, self).__init__()

#         self.teacher = t_net
#         self.teacher.eval()
#         for param in self.teacher.parameters():
#             param.requires_grad = False
#         self.student = s_net


#         EXT_S = FeatureExtractor(s_net, cfg, feat={})
#         EXT_T = FeatureExtractor(t_net, cfg_t, feat={})
#         self.s_fea = s_feat
#         self.t_fea = t_feat


#         self.cfg_s = cfg
#         self.loss = DistillationLoss(s_cfg=cfg, t_cfg=cfg_t)
        
#     def train_step(self, batched_inputs, optimizer, **kwargs):
#         loss_dict = self.student(batched_inputs)

#         with torch.no_grad():
#             _ = self.teacher(batched_inputs)
#         del _

#         softs_logits = []
#         preds_logits = []
#         softs_mask = []
#         preds_mask = []
#         softs_fea = []
#         preds_fea = []

#         for k in self.t_fea.feat:
#             if 'logits' in k:
#                 ##apply distillation on the cls logits map
#                 for ind, soft in enumerate(self.t_fea.feat[k]):
#                     softs_logits.append(soft)
#                     preds_logits.append(self.s_fea.feat[k][ind])
#             elif 'mask' in k:
#                 ##apply distillation on the instance mask
#                 for ind, soft in enumerate(self.t_fea.feat[k]):
#                     softs_mask.append(soft)
#                     preds_mask.append(self.s_fea.feat[k][ind])
#             else:
#                 ##apply distillation on the fpn features
#                 for ind, soft in enumerate(self.t_fea.feat[k]):
#                     softs_fea.append(soft)
#                     preds_fea.append(self.s_fea.feat[k][ind])
#             # else:
#             #     print('Not Implemented')
#             #     ##apply distillation on user define locations,
#         if "off" not in self.cfg_s.distillation['logits']:
#             logits_loss = self.loss(softs_logits, preds_logits, self.cfg_s.DISTILLATION.LOSS.LOGITS, 'logits')
#             loss_dict.update(logits_loss)
#             del logits_loss
#         if "off" not in self.cfg_s.DISTILLATION.LOSS.MASK:
#             mask_loss = self.loss(softs_mask, preds_mask, self.cfg_s.DISTILLATION.LOSS.MASK, 'mask')
#             loss_dict.update(mask_loss)
#             del mask_loss
#         if "off" not in self.cfg_s.DISTILLATION.LOSS.FEA:
#             FEA_loss = self.loss(softs_fea, preds_fea, self.cfg_s.DISTILLATION.LOSS.FEA, 'fea')
#             loss_dict.update(FEA_loss)
#             del FEA_loss
#         self.t_fea.feat = {}
#         self.s_fea.feat = {}
#         return loss_dict


#     def val_step(self, data_batch, **kwargs):
#         """The iteration step during validation.

#         This method shares the same signature as :func:`train_step`, but used
#         during val epochs. Note that the evaluation after training epochs is
#         not implemented with this method, but an evaluation hook.
#         """
#         output = self(**data_batch, **kwargs)
#         return output
