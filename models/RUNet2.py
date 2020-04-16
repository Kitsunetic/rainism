import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBlock, DeconvBlock, MeanShift, activation


class CALayer(nn.Module):
  def __init__(self, num_feats, batch_norm=False, weight_norm=False, reduction=16):
    super(CALayer, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.conv_du = nn.Sequential(
      ConvBlock(num_feats, num_feats//reduction, 1, bias=True, act_type='relu', batch_norm=batch_norm, weight_norm=weight_norm),
      ConvBlock(num_feats//reduction, num_feats, 1, bias=True, act_type='sigmoid', batch_norm=batch_norm, weight_norm=weight_norm),
    )

  def forward(self, x):
    y = self.avg_pool(x)
    y = self.conv_du(y)
    return x * y


class ResidualBlock(nn.Module):
  def __init__(self, num_feats, act_type='relu', batch_norm=False, weight_norm=False):
    super(ResidualBlock, self).__init__()
    self.conv = nn.Sequential(
      ConvBlock(num_feats, num_feats, 3, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm),
      ConvBlock(num_feats, num_feats, 3, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm),
      CALayer(num_feats, weight_norm=weight_norm)
    )

  def forward(self, x):
    return self.conv(x) + x


class RG(nn.Module):
  """Residual Group"""
  def __init__(self, num_feats, num_layers, act_type='relu', batch_norm=False, weight_norm=False):
    super(RG, self).__init__()
    
    m = []
    for i in range(num_layers):
      m.append(ResidualBlock(num_feats, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm))
    m.append(ConvBlock(num_feats, num_feats, 3, act_type=None, batch_norm=batch_norm, weight_norm=weight_norm))
    self.conv = nn.Sequential(*m)

  def forward(self, x):
    return self.conv(x) + x


class RUNet2(nn.Module):
  def __init__(self, in_channels, num_feats, num_layers,
               act_type=None, batch_norm=False, weight_norm=False):
    super(RUNet2, self).__init__()
    
    self.conv_in = ConvBlock(in_channels, num_feats, 3, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm)
    
    # 40 -> 20
    self.encoder1 = nn.Sequential(
      ConvBlock(1*num_feats, 2*num_feats, 3, stride=2, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm),
      ResidualBlock(2*num_feats, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm)
    )
    # 20 -> 10
    self.encoder2 = nn.Sequential(
      ConvBlock(2*num_feats, 4*num_feats, 3, stride=2, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm),
      ResidualBlock(4*num_feats, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm)
    )
    
    self.body = ResidualBlock(4*num_feats, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm)
    
    # 10 -> 20
    self.rg1 = [RG(4*num_feats, num_layers, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm) for _ in range(4)]
    self.rg1 = nn.Sequential(*self.rg1)
    self.decoder1 = DeconvBlock(4*num_feats, 2*num_feats, 3, stride=2, padding=1, output_padding=1, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm)
    # 20 -> 40
    self.rg2 = [RG(2*num_feats, num_layers, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm) for _ in range(2)]
    self.rg2 = nn.Sequential(*self.rg2)
    self.decoder2 = DeconvBlock(2*num_feats, 1*num_feats, 3, stride=2, padding=1, output_padding=1, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm)
    
    self.rg3 = [RG(1*num_feats, num_layers, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm) for _ in range(1)]
    self.rg3 = nn.Sequential(*self.rg3)
    
    self.conv_out = ConvBlock(num_feats, 1, 1, act_type='relu')

  def forward(self, x):
    f1 = self.conv_in(x)
    
    f2 = self.encoder1(f1)
    f3 = self.encoder2(f2)
    x = f3
    
    x = self.decoder1(self.rg1(x) + f3)
    x = self.decoder2(self.rg2(x) + f2)
    x = self.rg3(x) + f1
    
    x = self.conv_out(x)
    
    return x
