import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .RUnet import RUNet

class DRUNet(nn.Module):
  """Double RUNet"""
  def __init__(self, in_channels, num_feats, dropout=0.25, act_type='prelu', batch_norm=False, weight_norm=False):
    super(DRUNet, self).__init__()
    
    self.runet_land = RUNet(in_channels, num_feats, dropout, act_type, batch_norm, weight_norm)
    self.runet_sea = RUNet(in_channels, num_feats, dropout, act_type, batch_norm, weight_norm)
    

  def forward(self, land, sea):
    a = self.runet_land(land)
    b = self.runet_sea(sea)
    x = a + b
    
    return x
