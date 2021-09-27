import torch.nn as nn
import os
import sys
import torch
import numpy as np
from torch.nn import functional as F


# from .opts import PercepNet, PercepRes


def kaiming_uniform_(tensor, a=0, multi=1.0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \sqrt{\frac{6}{(1 + a^2) \times \text{fan\_in}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (0 for ReLU
            by default)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        # >>> w = torch.empty(3, 5)
        # >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound * multi, bound * multi)


class PercepRes(nn.Module):

    def __init__(self, block, num_block, in_class):
        super().__init__()

        self.in_channels = 64
        self.conv_trans = nn.Conv2d(19, 3, 3, 1, 1, bias=False)
        self._initialize_weights()
        self.percep = resnet50(pretrained=True)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_class, 64, kernel_size=3, padding=1, bias=False),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True))
        # # we use a different inputsize than the original paper
        # # so conv2_x's stride is 1
        # self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        # self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        # self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        # self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                # nn.init.uniform_(m.weight, a=0, b=1)

    #
    # def _make_layer(self, block, out_channels, num_blocks, stride):
    #     """make resnet layers(by layer i didnt mean this 'layer' was the
    #     same as a neuron netowork layer, ex. conv layer), one layer may
    #     contain more than one residual block
    #
    #     Args:
    #         block: block type, basic block or bottle neck block
    #         out_channels: output depth channel number of this layer
    #         num_blocks: how many blocks per layer
    #         stride: the stride of the first block of this layer
    #
    #     Return:
    #         return a resnet layer
    #     """
    #
    #     # we have num_block blocks per layer, the first block
    #     # could be 1 or 2, other blocks would always be 1
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_channels, out_channels, stride))
    #         self.in_channels = out_channels * block.expansion
    #
    #     return nn.Sequential(*layers)

    def forward(self, x):
        out = []
        output = self.conv_trans(x)
        output = self.percep(output)
        # output = self.conv2_x(output)
        # output = self.conv3_x(output)
        # output = self.conv4_x(output)
        # output = self.conv5_x(output)
        out.append(output)
        return out


class PercepNet(nn.Module):
    def __init__(self, in_nc, nf, init='normal', multi_init=1.0):
        super(PercepNet, self).__init__()
        self.conv0_0 = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf, nf, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv1_0 = nn.Sequential(nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2_0 = nn.Sequential(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv3_0 = nn.Sequential(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2))
        # #
        self.conv4_0 = nn.Sequential(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                     nn.MaxPool2d(kernel_size=2, stride=2))

        self._initialize_weights(init, multi_init)

    def _initialize_weights(self, init, multi_init):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init == 'normal':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    kaiming_uniform_(m.weight, multi=multi_init, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = []
        fea = self.conv0_0(x)
        # output.append(fea)
        fea = self.conv1_0(fea)
        # output.append(fea)
        fea = self.conv2_0(fea)
        # output.append(fea)
        fea = self.conv3_0(fea)
        fea = self.conv4_0(fea)
        output.append(fea)
        return output


#
class VGGFeatureExtractor(nn.Module):
    def __init__(self, init='normal', multi_init=1.0):
        super(VGGFeatureExtractor, self).__init__()
        # vgg = PercepRes(BottleNeck, [2, 2, 2, 2], in_class=19)
        vgg = PercepNet(in_nc=19, nf=64, init=init, multi_init=multi_init)
        for param in vgg.parameters():
            param.requires_grad = False
        self.loss_network = vgg
        self.perception_loss = nn.MSELoss()

    def forward(self, out_s, out_t):
        # self.loss_network.eval()
        perception = 0
        with torch.no_grad():
            perception_ss = self.loss_network(out_s)
            perception_ts = self.loss_network(out_t)
        # perception = perception + self.perception_loss(perception_ts, perception_ss)
        for perception_s, perception_t in zip(perception_ss, perception_ts):
            if perception_t.shape == perception_s.shape:
                perception = perception + 0.1 * self.perception_loss(perception_t, perception_s)
            else:
                print('shape dismatch')
                perception = perception + 0.1 * self.perception_loss(perception_s, perception_s)
        return perception


class Hard_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Hard_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.sigmoid(energy)

        return attention


class CriterionAttnhard(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionAttnhard, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        # self.attn1 = Hard_Attn(2048, 'relu')
        self.attn = Hard_Attn(512, 'relu')
        self.criterion_sd = torch.nn.MSELoss(size_average=True)
        self.criterion_s = torch.nn.CrossEntropyLoss(size_average=True)

    def forward(self, preds, attn_h):
        m_batchsize, C, w, h = preds[2].size()
        graph_s = self.attn(preds[2])
        attn_s = torch.cat((1 - graph_s, graph_s)).view(2, m_batchsize, w * h, w * h).permute(1, 0, 2, 3)
        loss = self.criterion_s(attn_s, attn_h)

        return loss


class CriterionFeaSum(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionFeaSum, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.criterion_sd = torch.nn.MSELoss(size_average=True)

    def forward(self, preds, soft):
        cs, ct = preds[1].size(1), soft[1].size(1)
        graph_s = torch.sum(torch.abs(preds[1]), dim=1, keepdim=True) / cs
        graph_t = torch.sum(torch.abs(soft[1]), dim=1, keepdim=True) / ct
        loss_graph = self.criterion_sd(graph_s, graph_t)
        return loss_graph
        # torch.abs


class CriterionKDMSE(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, ignore_index=255, T=1):
        super(CriterionKDMSE, self).__init__()
        self.ignore_index = ignore_index
        self.T = T
        self.criterion_kd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        if preds.shape == soft.shape:
            loss2 = self.criterion_kd(F.softmax(preds / self.T, dim=1), F.softmax(soft / self.T, dim=1))
        else:
            print('dim not match')
            loss2 = self.criterion_kd(F.softmax(soft / self.T, dim=1), F.softmax(soft / self.T, dim=1))
        return loss2


class CriterionKD(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, ignore_index=255, T=1):
        super(CriterionKD, self).__init__()
        self.ignore_index = ignore_index
        self.T = T
        self.criterion_kd = torch.nn.CrossEntropyLoss()

    def forward(self, preds, soft):
        if preds.shape == soft.shape:
            # loss2 = self.criterion_kd(F.softmax(preds / self.T, dim=1), F.softmax(soft / self.T, dim=1))
            loss2 = self.criterion_kd(preds, torch.tensor(soft,dtype=torch.long))
        else:
            loss2 = self.criterion_kd(F.softmax(soft / self.T, dim=1), F.softmax(soft / self.T, dim=1))
            print('shape dismatch')

        return loss2

class CriterionCrossEntropy(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion_kd = torch.nn.KLDivLoss()
    def forward(self,pred,soft):
        # pred = pred.permute(0,2,3,1).contiguous() 
        # pred = pred.view(-1,pred.shape[-1])

        # soft = soft.permute(0,2,3,1).contiguous() 
        # soft = soft.view(-1,soft.shape[-1])

        loss = self.criterion_kd(F.log_softmax(pred, dim=1), F.softmax(soft, dim=1))

        return loss

class CriterionMSE(nn.Module):
    '''
    knowledge distillation loss
    '''

    def __init__(self, ignore_index=255, T=1, T_channle=256, S_channel=128):
        super(CriterionMSE, self).__init__()
        self.ignore_index = ignore_index
        self.T = T
        self.criterion_mse = torch.nn.MSELoss()
        self.adaptor = nn.Conv2d(S_channel, T_channle, kernel_size=1, stride=1, padding=0)

    def forward(self, preds, soft):
        ada_pred = self.adaptor(preds)
        loss2 = self.criterion_mse(ada_pred, soft)
        return loss2


class Cos_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / (nm + 0.000001)
        attention = self.softmax(norm_energy)  # BX (N) X (N)
        return attention


class Cos_Attn_sig(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn_sig, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Sigmoid()  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / (nm + 0.000001)
        attention = self.softmax(norm_energy)  # BX (N) X (N)
        return attention


class Cos_Attn_no(nn.Module):
    """ Self attention Layer"""

    def __init__(self, activation):
        super(Cos_Attn_no, self).__init__()
        # self.chanel_in = in_dim
        self.activation = activation
        self.softmax = nn.Sigmoid()  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = x.view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.view(m_batchsize, width * height, 1), q_norm.view(m_batchsize, 1, width * height))
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        norm_energy = energy / (nm + 0.000001)
        # attention = self.softmax(norm_energy)  # BX (N) X (N)
        return norm_energy


class CriterionLSCos(nn.Module):
    '''
    local structure with cosine similarity
    '''

    def __init__(self, ignore_index=255, location=0, use_weight=True):
        super(CriterionLSCos, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.location = location
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def f(self, logit):
        b, c, w, h = logit.shape
        filter_11 = torch.from_numpy(np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 1], [0, 1, 0], [0, 0, 0]],
             [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
             [[0, 0, 0], [0, 1, 0], [0, 0, 1]]])).expand(c, 8, 3, 3).float().cuda()
        filter_11 = torch.transpose(filter_11, 0, 1)

        filter_10 = torch.from_numpy(np.array(
            [[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 0, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
             [[0, 0, 0], [0, 0, 0], [0, 0, 1]]])).expand(c, 8, 3, 3).float().cuda()
        filter_10 = torch.transpose(filter_10, 0, 1)

        filter_01 = torch.from_numpy(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])).expand(8, c, 3, 3).float().cuda()

        # filter_11_dep = torch.from_numpy(np.array( [[[1, 0, 0], [0, 1, 0], [0, 0, 0]],[[0, 1, 0], [0, 1, 0], [0, 0, 0]],[[0, 0, 1], [0, 1, 0], [0, 0, 0]],[[0, 0, 0], [1, 1, 0], [0, 0, 0]],
        #             [[0, 0, 0], [0, 1, 1], [0, 0, 0]],[[0, 0, 0], [0, 1, 0], [1, 0, 0]],[[0, 0, 0], [0, 1, 0], [0, 1, 0]],[[0, 0, 0], [0, 1, 0], [0, 0, 1]]])).expand(c, 8, 3, 3).float().cuda()
        filter_11_dep = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])).expand(c, 1, 3, 3).float().cuda()
        xxaa = F.conv2d(logit ** 2, filter_11)
        aa = F.conv2d(logit ** 2, filter_10)
        xx = F.conv2d(logit ** 2, filter_01)
        kernel_set = np.array(
            [[[0, 1, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 1], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
             [[0, 0, 0], [0, 1, 0], [0, 0, 1]]])

        filter_11_dep = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])).expand(c, 1, 3, 3).float().cuda()
        xa = F.conv2d(logit, filter_11_dep, groups=c)
        xa2 = torch.sum(xa * xa, 1, keepdim=True)

        for kernel_ in kernel_set:
            filter_11_dep = torch.from_numpy(kernel_).expand(c, 1, 3, 3).float().cuda()
            xa = F.conv2d(logit, filter_11_dep, groups=c)

            tem = torch.sum(xa * xa, 1, keepdim=True)
            xa2 = torch.cat((xa2, tem), dim=1)
        map = (xa2 - xxaa) / 2 * torch.rsqrt(xx) * torch.rsqrt(aa)
        map.cuda()
        return map

    def forward(self, preds, soft):
        graph_s = self.f(preds[1])
        graph_t = self.f(soft[1])
        loss_graph = self.criterion_sd(graph_s, graph_t)
        return loss_graph


class CriterionSDcos_sig(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionSDcos_sig, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.attn = Cos_Attn_sig('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds)
        graph_t = self.attn(soft)
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph


class CriterionSDcos(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionSDcos, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.attn = Cos_Attn('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds)
        graph_t = self.attn(soft)
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph


class CriterionSDcos_no(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionSDcos_no, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.attn = Cos_Attn_no('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds)
        graph_t = self.attn(soft)
        if graph_s.shape == graph_t.shape:
            loss_graph = self.criterion_sd(graph_s, graph_t)
        else:
            print('shape dismatch')

            loss_graph = self.criterion_sd(graph_s, graph_s)

        # loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph


class CriterionSDcos_no_sp(nn.Module):
    '''
    structure distillation loss based on graph
    '''

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionSDcos_no_sp, self).__init__()
        self.ignore_index = ignore_index
        self.use_weight = use_weight
        self.attn = Cos_Attn_no('relu')
        # self.attn2 = Cos_Attn(320, 'relu')
        # self.criterion = torch.nn.NLLLoss(ignore_index=ignore_index)
        # # self.criterion_cls = torch.nn.BCEWithLogitsLoss()
        self.criterion_sd = torch.nn.MSELoss()

    def forward(self, preds, soft):
        # h, w = labels.size(1), labels.size(2)
        graph_s = self.attn(preds[0])
        graph_t = self.attn(soft[0])
        loss_graph = self.criterion_sd(graph_s, graph_t)

        return loss_graph


class CriterionChannelAwareLoss(nn.Module):
    '''
    channel-aware fore/back-ground distillation loss
    '''
    def __init__(self, tau=1.0):
        super(CriterionChannelAwareLoss, self).__init__()
        self.tau = tau
        self.KL = torch.nn.KLDivLoss(reduction='none')

    def forward(self, preds, soft, mask):
        preds_S, preds_T = preds, soft
        if mask is None:
            mask = 1

        softmax_pred_T = F.softmax(preds_T/self.tau, dim=2)
        softmax_pred_S = F.log_softmax(preds_S/self.tau, dim=2)
        if len(preds_S.shape) == 4:
            N,num_head,WH,WH_ = preds_S.shape
            # preds_S = preds_S.reshape(N,num_head,-1)
            # preds_T = preds_T.reshape(N,num_head,-1)
            softmax_pred_T = F.softmax(preds_T/self.tau, dim=-1)
            softmax_pred_S = F.log_softmax(preds_S/self.tau, dim=-1)
            loss = self.KL(softmax_pred_S,softmax_pred_T).sum()
            return loss/(N*num_head*WH)
        else:
            N, C, WH = preds_S.shape
            mask = mask.squeeze(1)
            preds_S *= mask
            preds_T *= mask
            softmax_pred_T = F.softmax(preds_T/self.tau, dim=2)
            softmax_pred_S = F.log_softmax(preds_S/self.tau, dim=2)

            r = sum(mask)/mask.flatten().shape[0]

            loss = self.KL(softmax_pred_S,softmax_pred_T).sum()*r
            return loss/(N*C)


class KLdiv(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
        self.KL = torch.nn.KLDivLoss(reduction='none')
    def forward(self, preds, soft, mask):

        preds_S, preds_T = preds, soft
        attn_weight = 0.2 if len(preds_S.shape) == 4 else 1
        if mask is None:
            mask = torch.ones(preds_S.shape[0],preds_S.shape[2]).cuda()
        if len(preds_S.shape) == 4:
            N,C,W,H = preds_S.shape
            preds_S = preds_S.reshape(N,C,W*H)
            preds_T = preds_T.reshape(N,C,W*H)
            mask = mask.reshape(N,W*H)
        N, C, WH = preds_S.shape
        softmax_pred_T = F.softmax(preds_T/self.tau, dim=1)
        softmax_pred_S = F.log_softmax(preds_S/self.tau, dim=1)

        loss = self.KL(softmax_pred_S,softmax_pred_T).sum(dim=1)
        # loss = (loss*mask).sum()*attn_weight
        # return loss/(N*WH)
        loss = (loss*mask).mean()*attn_weight
        return loss

class CriterionChannelAwareLossGroup(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
        self.KL = torch.nn.KLDivLoss(reduction='none')

    def forward(self, preds, soft, mask, G=5):
        preds_S, preds_T = preds, soft
        if mask is None:
            mask = torch.ones(preds_S.shape[0],preds_S.shape[2]).cuda()
        if len(preds_S.shape) == 4:
            N,C,W,H = preds_S.shape
            preds_S = preds_S.reshape(N,C,W*H)
            preds_T = preds_T.reshape(N,C,W*H)
            mask = mask.reshape(N,W*H)

        N, C, WH = preds_S.shape
        if C % G==0:
            preds_S = preds_S.reshape(N,C//G,G*WH)
            preds_T = preds_T.reshape(N,C//G,G*WH)

        
        softmax_pred_T = F.softmax(preds_T/self.tau, dim=2)
        softmax_pred_S = F.log_softmax(preds_S/self.tau, dim=2)

        loss = self.KL(softmax_pred_S,softmax_pred_T).sum()

        return loss/(N*C)*G

class SaLoss(nn.Module):
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
        self.KL = KLdiv(tau)
        self.CA =CriterionChannelAwareLoss(tau)
        weights = nn.Parameter(torch.Tensor([2.71 for i in range(2)]),requires_grad=True)
        torch.nn.init.uniform_(weights,1,2)
        self.weights = weights

    def forward(self, preds, soft, mask):
        ca_loss = self.CA(preds, soft, mask)
        kl_loss = self.KL(preds, soft, mask)
        loss =  (1 / (self.weights[0] ** 2)) *ca_loss + (1 / (self.weights[1] ** 2)) *kl_loss + torch.log(self.weights[0]) + torch.log(self.weights[1])
        return loss