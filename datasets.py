import os
import re

import numpy as np
import torch
from torchvision import transforms


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
    
    self.tf = transforms.Compose([
      transforms.RandomCrop(35),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.RandomRotation(15),
      transforms.ToTensor()
    ])
    
    """
    self.tf_temperature = transforms.ToTensor()
    self.tf_surface = transforms.ToTensor()
    self.tf_gmi_longitude = transforms.ToTensor()
    self.tf_gmi_latitude = transforms.ToTensor()
    self.tf_dpr_longitude = transforms.ToTensor()
    self.tf_dpr_latitude = transforms.ToTensor()
    if self.is_train:
      self.tf_target = transforms.ToTensor()
    """

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
    """
    temperature = data[..., 0:9].astype(np.float32)
    surface_type = data[..., 9].astype(np.float32)
    gmi_longitude = data[..., 10].astype(np.float32)
    gmi_latitude = data[..., 11].astype(np.float32)
    dpr_longitude = data[..., 12].astype(np.float32)
    dpr_latitude = data[..., 13].astype(np.float32)
    
    # Transform
    temperature = self.tf_temperature(temperature)
    surface_type = self.tf_surface(surface_type)
    gmi_longitude = self.tf_gmi_longitude(gmi_longitude)
    gmi_latitude = self.tf_gmi_latitude(gmi_latitude)
    dpr_longitude = self.tf_dpr_longitude(dpr_longitude)
    dpr_latitude = self.tf_dpr_latitude(dpr_latitude)
    
    if self.is_train:
      target = self.tf_target(target)
      target = data[..., 14].astype(np.float32)
      # Remove NaN
      target[target <= -9999] = 0
    
      return target, temperature, surface_type, gmi_longitude, gmi_latitude, dpr_longitude, dpr_latitude
    else:
      return orbit, subset, temperature, surface_type, gmi_longitude, gmi_latitude, dpr_longitude, dpr_latitude
    """
    data = self.tf(data)
    
    if self.is_train:
      target = data[14, ...]
      data = data[:13, ...]
      return target, data
    else:
      return orbit, subset, data

  def __len__(self):
    return len(self.data)
