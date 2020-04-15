import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBlock, DeconvBlock, MeanShift, activation


class UNet1(nn.Module):
  def __init__(self, num_feats, act_type='relu', dropout=0.25, batch_norm=True, mean_shift=False, color_mean=None, color_std=None):
    super(UNet1, self).__init__()
    
    self.mean_shift = mean_shift
    
    if self.mean_shift:
      self.sub_mean = MeanShift(color_mean, color_std, -1)
      #self.add_mean = MeanShift(color_mean, color_std, 1)
    
    # encoder
    # 40x40 -> 20x20
    self.downblock1 = nn.Sequential(
      ConvBlock(9, 1*num_feats, 3, act_type=act_type),
      ConvBlock(1*num_feats, 1*num_feats, 3, act_type=act_type)
    )
    self.downtran1 = nn.Sequential(
      nn.BatchNorm2d(1*num_feats),
      nn.MaxPool2d(2),
      nn.Dropout2d(dropout)
    )
    # 20x20 -> 10x10
    self.downblock2 = nn.Sequential(
      ConvBlock(1*num_feats, 2*num_feats, 3, act_type=act_type),
      ConvBlock(2*num_feats, 2*num_feats, 3, act_type=act_type)
    )
    self.downtran2 = nn.Sequential(
      nn.BatchNorm2d(2*num_feats),
      nn.MaxPool2d(2),
      nn.Dropout2d(dropout)
    )
    
    # 10x10 -> 10x10
    self.code = ConvBlock(2*num_feats, 4*num_feats, 3, act_type=act_type)
    
    # 10x10 -> 20x20
    self.upblock1 = DeconvBlock(4*num_feats, 2*num_feats, 3, stride=2, padding=1, output_padding=1)
    self.uptran1 = nn.Sequential(
      nn.Dropout2d(dropout),
      ConvBlock(4*num_feats, 2*num_feats, 3, act_type=act_type),
      ConvBlock(2*num_feats, 2*num_feats, 3, act_type=act_type, batch_norm=batch_norm)
    )
    # 20x20 -> 40x40
    self.upblock2 = DeconvBlock(2*num_feats, 1*num_feats, 3, stride=2, padding=1, output_padding=1)
    self.uptran2 = nn.Sequential(
      nn.Dropout2d(dropout),
      ConvBlock(2*num_feats, 1*num_feats, 3, act_type=act_type),
      ConvBlock(1*num_feats, 1*num_feats, 3, act_type=act_type, batch_norm=batch_norm),
      nn.Dropout2d(dropout)
    )
    
    self.conv_out = ConvBlock(num_feats, 1, 1, act_type=act_type)
    
  def forward(self, x):
    if self.mean_shift:
      x = self.sub_mean(x)
    
    # encoder
    h1 = self.downblock1(x) # in -> 1
    x = self.downtran1(h1)
    h2 = self.downblock2(x) # 1 -> 2
    x = self.downtran2(h2)
    
    # center
    x = self.code(x) # 2 -> 4
    
    # decoder
    x = self.upblock1(x) # 4 -> 2
    h2 = torch.cat([x, h2], dim=1) # 2+2 = 4
    x = self.uptran1(h2) # 4 -> 2
    
    x = self.upblock2(x) # 2 -> 1
    h1 = torch.cat([x, h1], dim=1) # 1+1 = 2
    x = self.uptran2(h1) # 2 -> 1
    
    # output
    x = self.conv_out(x)
    
    #if self.mean_shift:
    #  x = self.add_mean(x)
    
    return x
