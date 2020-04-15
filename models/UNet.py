import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_valid_padding(kernel_size: int, dilation: int) -> int:
  kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
  padding = (kernel_size-1) // 2
  return padding


def activation(act_type='relu', inplace=True, n_parameters=1, slope=0.2) -> nn.Module:
  if not act_type:
    return None
  elif act_type == 'relu':
    act = nn.ReLU(True)
  elif act_type == 'lrelu':
    act = nn.LeakyReLU(0.2, True)
  elif act_type == 'prelu':
    act = nn.PReLU(1, 0.2)
  elif act_type == 'sigmoid':
    act = nn.Sigmoid()
  else:
    raise NotImplementedError(act_type)
  return act


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, padding=0, valid_padding=True, dilation=1, bias=True, act_type=None, conv_type='CAB', batch_norm=False, weight_norm=False) -> nn.Sequential:
  if valid_padding:
    padding = get_valid_padding(kernel_size, dilation)
  
  conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)
  if weight_norm: conv = nn.utils.weight_norm(conv)
  m = [conv]
  if conv_type == 'CAB':
    if act_type: m.append(activation(act_type))
    if batch_norm: m.append(nn.BatchNorm2d(out_channels))
  elif conv_type == 'CBA':
    if batch_norm: m.append(nn.BatchNorm2d(out_channels))
    if act_type: m.append(activation(act_type))
  else:
    raise NotImplementedError(f'conv_type must be one of [\'CAB\', \'CBA\'], but \'{conv_type}\'')
  return nn.Sequential(*m)


def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, bias=True, conv_type='CBA', act_type='relu', batch_norm=False, weight_norm=False) -> nn.Sequential:
  # TODO: Why DeconvBlock doesn't have valid_padding?
  deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias, output_padding=output_padding)
  if weight_norm:
    deconv = nn.utils.weight_norm(deconv)
  m = [deconv]
  if conv_type == 'CAB':
    if act_type: m.append(activation(act_type))
    if batch_norm: m.append(nn.BatchNorm2d(out_channels))
  elif conv_type == 'CBA':
    if batch_norm: m.append(nn.BatchNorm2d(out_channels))
    if act_type: m.append(activation(act_type))
  else:
    raise NotImplementedError(f'conv_type must be one of [\'CAB\', \'CBA\'], but \'{conv_type}\'')
  return nn.Sequential(*m)


class MeanShift(nn.Conv2d):
  def __init__(self, color_mean, color_std, sign=-1):
    c = min(len(color_mean), len(color_std))
    super(MeanShift, self).__init__(c, c, 1)
    
    mean = torch.Tensor(color_mean[:c])
    std = torch.Tensor(color_std[:c])
    
    self.weight.data = torch.eye(c).view(c, c, 1, 1)
    self.weight.data.div_(std.view(c, 1, 1, 1))
    self.bias.data = sign * mean
    self.bias.data.div_(std)
    
    for p in self.parameters():
      p.requires_grad = False


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
