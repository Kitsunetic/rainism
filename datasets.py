import os
import re
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def custom_transforms(x):
  # Random horizontal rotation
  if random.random() > 0.5:
    x = cv2.flip(x, 0)
  
  # Random vertical rotation
  if random.random() > 0.5:
    x = cv2.flip(x, 1)
  
  return x


class LetsGoHikingDataset(torch.utils.data.Dataset):
  def __init__(self, dataset_path, is_train=True):
    super(LetsGoHikingDataset, self).__init__()
    
    self.is_train = is_train
    
    files = os.listdir(dataset_path)
    files = filter(lambda x: x.endswith('.npy'), files)
    files = list(files)
    
    # Get orbit and subset number
    self.data = []
    for fname in files:
      fpath = os.path.join(dataset_path, fname)
      groups = re.search('subset_(\\d+)_(\\d+)\\.npy', fname).groups()
      if len(groups) != 2:
        continue
      
      orbit = int(groups[0])
      subset = int(groups[1])
      self.data.append((fpath, orbit, subset))
    
    self.tf = transforms.ToTensor()

  def __getitem__(self, index):
    fpath, orbit, subset = self.data[index]
    data = np.load(fpath).astype(np.float32)
    """
    # Data Specifications
    0~8  | 밝기 온도 (단위: K, 10.65GHz~89.0GHz)
      9  | 지표 타입 (앞자리 0: Ocean, 앞자리 1: Land, 앞자리 2: Coastal, 앞자리 3: Inland Water)
     10  | GMI 경도
     11  | GMI 위도
     12  | DPR 경도
     13  | DPR 위도
     14  | 강수량 (mm/h, 결측치는 -9999.xxx 형태의 float 값으로 표기) (TARGET)
    """
    data = custom_transforms(data)
    data = self.tf(data)
    
    temperature = data[:9, ...]
    surface = data[9, ...]
    surface = surface.view((1, *surface.shape))
    surface[surface > 20] = 0
    surface[surface > 10] -= 10
    out = torch.cat([temperature, surface], dim=0)
    
    if self.is_train:
      target = data[14, ...]
      target = target.view((1, *target.shape))
      
      # Remove NaN
      target[target <= -9000] = 0
      
      return target, out
    else:
      return orbit, subset, out

  def __len__(self):
    return len(self.data)
