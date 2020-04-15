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
  def __init__(self, num_feats, batch_norm=False, weight_norm=False):
    super(ResidualBlock, self).__init__()
    self.conv = nn.Sequential(
      ConvBlock(num_feats, num_feats, 3, act_type='relu', batch_norm=batch_norm, weight_norm=weight_norm),
      ConvBlock(num_feats, num_feats, 3, act_type='relu', batch_norm=batch_norm, weight_norm=weight_norm),
      CALayer(num_feats, weight_norm=weight_norm)
    )

  def forward(self, x):
    return self.conv(x) + x


def DownSampleBlock(num_feats, dropout):
  m = [nn.BatchNorm2d(num_feats), nn.MaxPool2d(2)]
  
  if dropout > 0:
    m.append(nn.Dropout2d(dropout))
  
  return nn.Sequential(*m)


def UpSampleBlock(in_channels, out_channels):
  return DeconvBlock(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)


def EncoderBlock(in_channels, out_channels, batch_norm=False, weight_norm=False):
  return nn.Sequential(
    ConvBlock(in_channels, out_channels, 3, batch_norm=batch_norm, weight_norm=weight_norm),
    ResidualBlock(out_channels, batch_norm=batch_norm, weight_norm=weight_norm)
  )


def DecoderBlock(num_feats, dropout, batch_norm=False, weight_norm=False):
  return nn.Sequential(
    nn.Dropout2d(dropout),
    ResidualBlock(num_feats, batch_norm=batch_norm, weight_norm=weight_norm)
  )


class RUNet(nn.Module):
  def __init__(self, in_channels, num_feats, dropout=0.25, act_type='prelu', batch_norm=False, weight_norm=False, mean_shift=False, color_mean=None, color_std=None):
    super(RUNet, self).__init__()
    
    self.mean_shift = mean_shift
    
    # mean shift
    if self.mean_shift:
      self.sub_mean = MeanShift(color_mean, color_std, -1)
      self.add_mean = MeanShift([30.9175], [0.6971], 1)
    
    # 40 -> 20
    self.encoder1 = EncoderBlock(in_channels, num_feats, batch_norm=batch_norm, weight_norm=weight_norm)
    self.down1    = DownSampleBlock(num_feats, dropout)
    # 20 -> 10
    self.encoder2 = EncoderBlock(1*num_feats, 2*num_feats, batch_norm=batch_norm, weight_norm=weight_norm)
    self.down2    = DownSampleBlock(2*num_feats, dropout)
    
    # 
    self.center   = EncoderBlock(2*num_feats, 4*num_feats, batch_norm=batch_norm, weight_norm=weight_norm)
    
    self.up2      = UpSampleBlock(4*num_feats, 2*num_feats)
    self.decoder2 = DecoderBlock(2*num_feats, dropout, batch_norm=batch_norm, weight_norm=weight_norm)
    self.up1      = UpSampleBlock(2*num_feats, num_feats)
    self.decoder1 = DecoderBlock(num_feats, dropout, batch_norm=batch_norm, weight_norm=weight_norm)
    
    self.tail = nn.Sequential(
      nn.Dropout2d(dropout),
      ResidualBlock(num_feats, batch_norm=batch_norm, weight_norm=weight_norm),
      ConvBlock(num_feats, 1, 1, act_type='relu')
    )

  def forward(self, x):
    if self.mean_shift:
      x = self.sub_mean(x)
    
    h1 = self.encoder1(x) # f
    x = self.down1(h1)
    h2 = self.encoder2(x) # 2f
    x = self.down2(h2)
    
    x = self.center(x) # 4f
    
    x = self.up2(x) # 2f
    x = self.decoder2(x) + h2 # 2f
    x = self.up1(x) # f
    x = self.decoder1(x) + h1 # f
    
    x = self.tail(x)
    
    if self.mean_shift:
      x = self.add_mean(x)
    
    return x
