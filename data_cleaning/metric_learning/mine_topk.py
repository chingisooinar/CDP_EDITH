#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:01:14 2021

@author: chingis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:51:25 2021

@author: nuvilabs
"""

import torch
import time

from lib.utils import AverageMeter
from model import l2_norm
import math



def kNN(net, testloader, K=None, reference_loader=None):
    net.eval()
    net_time = AverageMeter()
    cls_time = AverageMeter()
    total = 0
    testsize = testloader.dataset.__len__()

    if reference_loader:
        reference_features = torch.zeros(len(reference_loader.dataset.samples), net.dim).cuda().t()

        transform_bak = reference_loader.dataset.transform
        reference_loader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(
            reference_loader.dataset, batch_size=256, shuffle=False, num_workers=1)
        for batch_idx, (inputs, targets, indexes) in enumerate(temploader):
            bs = inputs.size(0)
            features = net(inputs, test=True)
            reference_features[:, batch_idx * bs:batch_idx *
                           bs + bs] = features.data.t()
        reference_loader.dataset.transform = transform_bak

    indices = []
    threshold = math.cos(math.pi * 70 / 180)
    with torch.no_grad():

        for batch_idx, (inputs, targets, indexes) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(non_blocking=True)
            bs = inputs.size(0)
            features = net(inputs, test=True)
            net_time.update(time.time() - end)
            end = time.time()


            dist = torch.mm(features, reference_features)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            assert yd.max() <= 1.05 and yd.min() >= -1
            indices.extend(yi[yd > threshold].flatten().detach().cpu().numpy().tolist())
    return list(set(indices))
