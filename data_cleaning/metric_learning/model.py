#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:54:17 2021

@author: nuvilabs
"""
import torch
import torch.nn as nn
import torchvision.models as models
import os
import math
from scipy.special import binom
import torch.backends.cudnn as cudnn
from torch.autograd import Variable


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output




class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332,  s=64., m1=1, m2=0., m3=0., K=1):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum * K))
        # initial kernel
        print(f'K={K}')
        self.k = K
        self.m1 = m1
        # ==========================LSoftmaxLinear=====================================
        self.divisor = math.pi / self.m1
        self.coeffs = binom(m1, range(0, m1 + 1, 2))
        self.cos_exps = range(self.m1, -1, -2)
        self.sin_sq_exps = range(len(self.cos_exps))
        self.signs = [1]
        for i in range(1, len(self.sin_sq_exps)):
            self.signs.append(self.signs[-1] * -1)
        self.iteration = 0
        # ===============================================================
        self.beta = 200
        self.beta_min = 0
        self.scale = 0.99
        self.m3 = m3
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m2  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m2)
        self.sin_m = math.sin(m2)
        self.mm = self.sin_m * m2  # issue 1
        self.threshold = math.cos(math.pi - m2)

    def find_k(self, cos, eps=1e-7):
        theta = torch.acos(cos)
        k = (theta / self.divisor).floor().detach()
        return k

    def forward(self, embbedings, label, combine=False):
        nB = len(embbedings)
        idx_ = torch.arange(0, nB, dtype=torch.long)
        beta = max(self.beta, self.beta_min)
        self.kernel_norm = l2_norm(self.kernel, axis=0) if self.m > 0 or combine else self.kernel
        logit = torch.mm(embbedings, self.kernel_norm)
        if combine:
            return self.s * logit
        if self.k > 1:
            logit = logit.view(-1, self.classnum, self.k)
            logit, _ = logit.max(axis=2)

        output = logit * 1.0
        if self.m1 > 1:
            logit_target = logit[idx_, label]
            # cos(theta) = w * x / ||w||*||x||

            w_target_norm = self.kernel[:, label].norm(p=2, dim=0)
            x_norm = embbedings.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            sin2_theta = 1 - cos_theta_target**2
            coeffs = Variable(embbedings.data.new(self.coeffs))
            cos_exps = Variable(embbedings.data.new(self.cos_exps))
            sin_sq_exps = Variable(embbedings.data.new(self.sin_sq_exps))
            signs = Variable(embbedings.data.new(self.signs))
            cos_terms = cos_theta_target.unsqueeze(1) ** cos_exps.unsqueeze(0)
            sin_sq_terms = (sin2_theta.unsqueeze(1)
                            ** sin_sq_exps.unsqueeze(0))

            cosm_terms = (signs.unsqueeze(0) * coeffs.unsqueeze(0)
                          * cos_terms * sin_sq_terms)
            cosm = cosm_terms.sum(1)
            k = self.find_k(cos_theta_target)

            ls_target = w_target_norm * x_norm * (((-1)**k * cosm) - 2*k)

            output[idx_, label] = (beta * output[idx_, label] + ls_target) / (1 + beta)
            self.beta = self.beta * self.scale

        elif self.m1 == 1. and self.m != 0:
            cos_theta = logit.clamp(-1, 1)
            cos_theta_2 = torch.pow(cos_theta, 2)
            sin_theta_2 = 1 - cos_theta_2
            sin_theta = torch.sqrt(sin_theta_2)
            cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
            cos_theta_m -= self.m3
            # this condition controls the theta+m should in range [0, pi]
            #      0<=theta+m<=pi
            #     -m<=theta<=pi-m
            cond_v = cos_theta - self.threshold
            cond_mask = cond_v <= 0
            keep_val = (cos_theta - self.mm)  # when theta not in [0,pi], use cosface instead
            cos_theta_m[cond_mask] = keep_val[cond_mask]
            output[idx_, label] = cos_theta_m[idx_, label]
            output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


class EmbedderWrapper:

    def __init__(self, args, n_classes):
        print('==> Building model..')

        self.module = models.resnet50(pretrained=True)
        num_ftrs = self.module.fc.in_features
        self.module.fc = nn.Linear(num_ftrs, args.low_dim)
        self.head = None
        self.training = False
        self.device = None
        self.dim = args.low_dim
        self.if_norm = args.arcface
        if args.arcface:
            print("Arcface head")
            self.head = Arcface(embedding_size=args.low_dim, classnum=n_classes, K=args.subcenters,
                                m1=args.m_1, m2=args.m_2)

    def to_device(self, device):
        if self.head:
            self.head.to(device)
        if device == 'cuda':
            self.module = torch.nn.DataParallel(self.module, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
        self.module.to(device)
        self.device = device

    def get_parameters(self, args):
        if args.arcface:
            return [{'params': self.head.kernel}, {'params': self.module.parameters()}]
        return self.module.parameters()

    def resume(self, args, name):
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+name)
        self.module.load_state_dict(checkpoint['net'])
        if args.arcface and 'head' in checkpoint.keys():
            self.head.kernel = nn.Parameter(checkpoint['head']['kernel'])


    def save_model(self, args):
        torch.save(self.module.module, args.instance+'_model.pth')
        if self.head:
            torch.save(self.head.kernel, args.instance+'_kernel.pt')

    def train(self, mode=True):
        self.training = mode
        self.module.train()

    def eval(self, mode=False):
        self.training = mode
        self.module.eval()

    def __call__(self, x, targets=None, test=False):
        features = self.module(x)
        norm_features = l2_norm(features) if self.if_norm or test else features
        outputs = norm_features
        if targets is not None:
            assert self.head is not None and targets is not None
            outputs = self.head(norm_features, targets)

        return outputs




