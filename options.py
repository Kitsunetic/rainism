import json
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import datasets
import models
import utils


def _load_dataset(opts) -> Tuple[DataLoader, DataLoader, DataLoader]:
  if opts['type'] == 'LetsGoHikingDataset':
    dataset = datasets.LetsGoHikingDataset(opts['train_path'])
    testset = datasets.LetsGoHikingDataset(opts['test_path'], is_train=False)
  else:
    raise NotImplementedError('Unknown dataset type ' + opts['type'])
  
  dataset_size = len(dataset)
  validset_size = int(dataset_size * opts['rate_valid'])
  trainset_size = dataset_size - validset_size
  trainset, validset = random_split(dataset, (trainset_size, validset_size))
  
  trainloader = DataLoader(trainset, **opts['kwargs'])
  validloader = DataLoader(validset, **opts['kwargs'])
  
  test_opts = opts['kwargs']
  test_opts['batch_size'] = 128
  testloader = DataLoader(testset, **test_opts)
  
  return trainloader, validloader, testloader

def _load_model(opts) -> nn.Module:
  if opts['type'] == 'UNet1':
    model = models.UNet1(**opts['kwargs'])
  elif opts['type'] == 'SRFBN':
    model = models.SRFBN(**opts['kwargs'])
  elif opts['type'] == 'RUNet':
    model = models.RUNet(**opts['kwargs'])
  else:
    raise NotImplementedError('Unknown network type ' + opts['type'])
  
  if opts['checkpoint_path']:
    print('Load checkpoint', opts['checkpoint_path'])
    with open(opts['checkpoint_path'], 'rb') as f:
      state_dict = torch.load(f)
      model.load_state_dict(state_dict)
  
  model = nn.DataParallel(model)
  
  return model

def _load_criterion(opts) -> nn.Module:
  if opts['type'] == 'L1':
    criterion = nn.L1Loss()
  elif opts['type'] == 'L2':
    criterion = nn.MSELoss()
  else:
    raise NotImplementedError('Unknown criterion type ' + opts['type'])
  
  return criterion

def _load_optimizer(opts, parameters) -> torch.optim.Optimizer:
  optimizer = getattr(torch.optim, opts['type'])(parameters, **opts['kwargs'])
  
  if opts['checkpoint_path']:
    print('Load checkpoint', opts['checkpoint_path'])
    with open(opts['checkpoint_path'], 'rb') as f:
      state_dict = torch.load(f)
      optimizer.load_state_dict(state_dict)
  
  return optimizer

def load_environment(opts) -> Tuple[Dict, DataLoader, DataLoader, DataLoader, nn.Module, nn.Module, torch.optim.Optimizer]:
  trainloader, validloader, testloader = _load_dataset(opts['dataset'])
  model = _load_model(opts['network']).cuda()
  criterion = _load_criterion(opts['criterion']).cuda()
  optimizer = _load_optimizer(opts['optimizer'], model.parameters())
  
  return opts, trainloader, validloader, testloader, model, criterion, optimizer

def load_json(json_path) -> Tuple[Dict, DataLoader, DataLoader, DataLoader, nn.Module, nn.Module, torch.optim.Optimizer]:
  with open(json_path, 'r') as f:
    opts = json.load(f)
  
  return load_environment(opts)
