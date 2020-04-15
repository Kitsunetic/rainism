import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBlock, DeconvBlock, MeanShift, activation


class FeedbackBlock(nn.Module):
  """
  Used in SRFBN
  """
  def __init__(self, num_feats, num_groups, scale, act_type='relu', batch_norm=False, weight_norm=False):
    super(FeedbackBlock, self).__init__()
    
    if   scale == 1:  stride = 1; padding = 2; kernel_size = 5
    elif scale == 2:  stride = 2; padding = 2; kernel_size = 6
    elif scale == 3:  stride = 3; padding = 2; kernel_size = 7
    elif scale == 4:  stride = 4; padding = 2; kernel_size = 8
    elif scale == 8:  stride = 8; padding = 2; kernel_size = 12
    else: raise NotImplementedError(f'scale must be one of [1, 2, 3, 4, 8] but {scale}')
    
    self.num_groups = num_groups
    
    self.compress_in = ConvBlock(2*num_feats, num_feats, 1, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm)
    
    up_blocks = []
    down_blocks = []
    uptran_blocks = []
    downtran_blocks = []
    for idx in range(self.num_groups):
      up_blocks.append(DeconvBlock(num_feats, num_feats, kernel_size, stride, padding, act_type=act_type, batch_norm=batch_norm))
      down_blocks.append(ConvBlock(num_feats, num_feats, kernel_size, stride, padding, act_type=act_type, batch_norm=batch_norm))
      if idx > 0:
        uptran_blocks.append(ConvBlock(num_feats*(idx+1), num_feats, 1, 1, act_type=act_type, batch_norm=batch_norm))
        downtran_blocks.append(ConvBlock(num_feats*(idx+1), num_feats, 1, 1, act_type=act_type, batch_norm=batch_norm))
    self.up_blocks = nn.Sequential(*up_blocks)
    self.down_blocks = nn.Sequential(*down_blocks)
    self.uptran_blocks = nn.Sequential(*uptran_blocks)
    self.downtran_blocks = nn.Sequential(*downtran_blocks)

    self.compress_out = ConvBlock(num_groups*num_feats, num_feats, 1, act_type=act_type, batch_norm=batch_norm)
    
    self.should_reset = True
    self.last_hidden = None

  def forward(self, x):
    if self.should_reset:
      self.last_hidden = torch.zeros(x.size()).cuda()
      self.last_hidden.copy_(x)
      self.should_reset = False
    
    x = torch.cat((x, self.last_hidden), dim=1)
    x = self.compress_in(x)
    
    lr_feats, hr_feats = [x], []
    LD_L = torch.cat(tuple(lr_feats), 1)
    LD_H = self.up_blocks[0](LD_L)
    hr_feats.append(LD_H)
    LD_H = torch.cat(tuple(hr_feats), 1)
    LD_L = self.down_blocks[0](LD_H)
    lr_feats.append(LD_L)
    
    for i in range(1, self.num_groups):
      LD_L = torch.cat(tuple(lr_feats), 1)
      LD_L = self.uptran_blocks[i-1](LD_L)
      LD_H = self.up_blocks[i](LD_L)
      hr_feats.append(LD_H)
      
      LD_H = torch.cat(tuple(hr_feats), 1)
      LD_H = self.downtran_blocks[i-1](LD_H)
      LD_L = self.down_blocks[i](LD_H)
      lr_feats.append(LD_L)
    
    del hr_feats
    output = torch.cat(tuple(lr_feats[1:]), 1)
    output = self.compress_out(output)
    self.last_hidden = output
    return output
  
  def reset_state(self):
    self.should_reset = True


class SRFBN(nn.Module):
  def __init__(self, in_channels, num_feats, num_steps, num_groups, 
               act_type='prelu', batch_norm=False, weight_norm=False, 
               mean_shift=False, color_mean=None, color_std=None):
    super(SRFBN, self).__init__()
    
    stride = 1
    padding = 2
    kernel_size = 5

    self.num_steps = num_steps
    self.num_feats = num_feats
    self.mean_shift = mean_shift
    
    if self.mean_shift:
      self.sub_mean = MeanShift(color_mean, color_std, -1)
    
    self.conv_inter = ConvBlock(in_channels, 1, 1, act_type='relu', batch_norm=batch_norm, weight_norm=weight_norm)
    
    # LR feature extraction block
    self.conv_in = ConvBlock(in_channels, 4*num_feats, 3, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm)
    self.feat_in = ConvBlock(4*num_feats, num_feats, 1, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm)
    
    # basic block
    self.block = FeedbackBlock(num_feats, num_groups, 1, act_type=act_type, batch_norm=batch_norm, weight_norm=weight_norm)

    self.out = DeconvBlock(num_feats, num_feats, kernel_size, stride=stride, padding=padding, act_type='prelu', batch_norm=batch_norm, weight_norm=weight_norm)
    self.conv_out = ConvBlock(num_feats, 1, 3, act_type='relu', batch_norm=batch_norm, weight_norm=weight_norm)

  def forward(self, x):
    self._reset_state()
    
    if self.mean_shift:
      x = self.sub_mean(x)
		# uncomment for pytorch 0.4.0
    # inter_res = self.upsample(x)
		
		# comment for pytorch 0.4.0
    inter_res = self.conv_inter(x)

    x = self.conv_in(x)
    x = self.feat_in(x)

    outs = []
    for _ in range(self.num_steps):
      h = self.block(x) # feedback block store state each step
      h = self.out(h)
      h = self.conv_out(h)

      h = torch.add(inter_res, h)
      outs.append(h)

    #return outs # return output of every timesteps
    return h

  def _reset_state(self):
    self.block.reset_state()
