import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBlock, DeconvBlock, MeanShift, activation


class CALayer(nn.Module):
  """
  Used in EDRN
  """
  def __init__(self, channel, weight_norm=False, reduction=16):
    super(CALayer, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.conv_du = nn.Sequential(
      ConvBlock(channel, channel//reduction, 1, bias=True, act_type='relu', weight_norm=weight_norm),
      ConvBlock(channel//reduction, channel, 1, bias=True, act_type='sigmoid', weight_norm=weight_norm),
    )

  def forward(self, x):
    y = self.avg_pool(x)
    y = self.conv_du(y)
    return x * y


class ResidualBlock(nn.Module):
  """
  Used in EDRN, EDSR
  """
  def __init__(self, in_channels, grow_rate, weight_norm=False):
    super(ResidualBlock, self).__init__()
    self.conv = nn.Sequential(
      ConvBlock(in_channels, grow_rate, 3, act_type='relu', weight_norm=weight_norm),
      ConvBlock(grow_rate, in_channels, 3, act_type='relu', weight_norm=weight_norm),
      CALayer(in_channels, weight_norm=weight_norm)
    )

  def forward(self, x):
    return self.conv(x) + x


class RG(nn.Module):
  """
  Used in EDRN
  """
  def __init__(self, grow_rate, num_conv_layers, weight_norm=False):
    super(RG, self).__init__()
    conv_residual = []
    self.conv_residual = nn.Sequential(*[ResidualBlock(grow_rate, grow_rate, weight_norm=weight_norm) for _ in range(num_conv_layers)])
    self.last_conv = ConvBlock(grow_rate, grow_rate, 3, act_type=None, weight_norm=weight_norm)

  def forward(self, x):
    h = self.conv_residual(x)
    h = self.last_conv(h)
    return h + x


class EDRN(nn.Module):
  def __init__(self, in_channels, out_channels, grow_rate, scale, mean_shift=True, rgb_mean=(0.4313, 0.4162, 0.3861), rgb_std=(1.0, 1.0, 1.0), batch_norm=False, weight_norm=False):
    super(EDRN, self).__init__()
    
    D, C, G = 4, 10, 16
    
    if mean_shift:
      self.sub_mean = MeanShift(rgb_mean, rgb_std, -1)
      self.add_mean = MeanShift(rgb_mean, rgb_std, 1)
    
    self.SFENet = ConvBlock(in_channels, grow_rate, 3, weight_norm=weight_norm)
    self.encoder1 = ConvBlock(1*grow_rate, 2*grow_rate, 3, stride=2, conv_type='CBA', act_type='relu', batch_norm=batch_norm, weight_norm=weight_norm)
    self.encoder2 = ConvBlock(2*grow_rate, 4*grow_rate, 3, stride=2, conv_type='CBA', act_type='relu', batch_norm=batch_norm, weight_norm=weight_norm)
    self.decoder1 = DeconvBlock(4*grow_rate, 2*grow_rate, 3, padding=1, output_padding=1, stride=2, conv_type='CBA', act_type='relu', batch_norm=batch_norm, weight_norm=weight_norm)
    self.decoder2 = DeconvBlock(2*grow_rate, 1*grow_rate, 3, padding=1, output_padding=1, stride=2, conv_type='CBA', act_type='relu', batch_norm=batch_norm, weight_norm=weight_norm)
    # Additional deconv for upscale image
    #self.decoder3 = DeconvBlock(1*grow_rate, 1*grow_rate, 3, padding=1, output_padding=1, stride=2, conv_type='CBA', act_type='relu', batch_norm=batch_norm, weight_norm=weight_norm)
    
    self.RG0 = [RG(4*grow_rate, C, weight_norm=weight_norm) for _ in range(D//1)]
    self.RG0.append(ConvBlock(4*grow_rate, 4*grow_rate, 3, weight_norm=weight_norm))
    self.RG1 = [RG(2*grow_rate, C, weight_norm=weight_norm) for _ in range(D//2)]
    self.RG1.append(ConvBlock(2*grow_rate, 2*grow_rate, 3, weight_norm=weight_norm))
    self.RG2 = [RG(1*grow_rate, C, weight_norm=weight_norm) for _ in range(D//4)]
    self.RG2.append(ConvBlock(1*grow_rate, 1*grow_rate, 3, weight_norm=weight_norm))
    self.RG0 = nn.Sequential(*self.RG0)
    self.RG1 = nn.Sequential(*self.RG1)
    self.RG2 = nn.Sequential(*self.RG2)
    
    self.restoration = ConvBlock(grow_rate, out_channels, 3, weight_norm=weight_norm)

  def forward(self, x):
    x = self.sub_mean(x)
    
    # encoders
    f1 = self.SFENet(x)
    f2 = self.encoder1(f1)
    f3 = self.encoder2(f2)
    x = f3
    
    # add residual
    x = self.decoder1(self.RG0(x) + f3)
    x = self.decoder2(self.RG1(x) + f2)
    #x = self.decoder3(self.RG2(x) + f1)
    x = self.RG2(x) + f1
    
    x = self.restoration(x)
    x = self.add_mean(x)
    
    return x
