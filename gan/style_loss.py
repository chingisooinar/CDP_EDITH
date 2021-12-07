#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:00:09 2021

@author: chingis
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
def gram_matrix(input):
    """
    A gram matrix is the result of multiplying a given matrix by its transposed matrix. 
    """
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        

    def forward(self, input_map, target_map):
        G = gram_matrix(input_map)
        target = gram_matrix(target_map).detach()
        loss = F.mse_loss(G, target)
        return loss