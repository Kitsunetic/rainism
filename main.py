import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import datasets
import models
import options
import utils


def train(dataloader, model, criterion, optimizer, desc='') -> float:
  torch.set_grad_enabled(True)
  
  loss_list = []
  with tqdm(desc=desc, total=len(dataloader), ncols=128) as t:
    for target, data in dataloader:
      target = target.cuda()
      input = data[:, :9, ...].cuda()
      
      pred = model(input)
      
      optimizer.zero_grad()
      loss = criterion(target, pred)
      loss.backward()
      optimizer.step()
      
      # update tqdm
      loss_list.append(loss.item())
      mean_loss = sum(loss_list) / len(loss_list)
      t.set_postfix_str('loss %.4f'%mean_loss)
      t.update()
  
  return mean_loss

def valid(dataloader, model, criterion, optimizer, desc='') -> float:
  torch.set_grad_enabled(False)
  
  loss_list = []
  with tqdm(desc=desc, total=len(dataloader), ncols=128) as t:
    for target, data in dataloader:
      target = target.cuda()
      input = data[:, :9, ...].cuda()
      
      pred = model(input)
      
      loss = criterion(target, pred)
      
      # update tqdm
      loss_list.append(loss.item())
      mean_loss = sum(loss_list) / len(loss_list)
      t.set_postfix_str('vloss %.4f'%mean_loss)
      t.update()
  
  return mean_loss

def test(dataloader, model, submit_path, desc=''):
  """Create submission *.csv file"""
  torch.set_grad_enabled(False)
  
  with open(submit_path, 'w') as f:
    f.write('id')
    for i in range(1, 1601):
      f.write(f',px_{i}')
    f.write('\n')
    
    with tqdm(desc=desc, total=len(dataloader), ncols=128) as t:
      for orbit, subset, data in dataloader:
        input = data[:, :9, ...].cuda()
        
        preds = model(input) # n x 1 x 40 x 40
        
        batch_size = preds.shape[0]
        for i in range(batch_size):
          pred = preds[i].cpu().detach()
          pred = pred.flatten() # 1600
          pred = pred.tolist()
          submit_id = '%06d_%02d'%(orbit[i], subset[i])
          
          f.write(submit_id)
          for i in range(1600):
            f.write(',%.4f'%pred[i])
          f.write('\n')
          f.flush()
        
        t.update()


def save_checkpoint(dirroot_ckpt, model, optimizer, epoch, loss):
  now = datetime.now().strftime('%y%m%d')
  
  model_name = 'ckpt-%s-epoch%d-loss%.4f-model.pth'%(now, epoch, loss)
  model_path = os.path.join(dirroot_ckpt, model_name)
  print('Save model', model_name)
  with open(model_path, 'wb') as f:
    torch.save(model.module.state_dict(), f)
  
  optim_name = 'ckpt-%s-epoch%d-loss%.4f-optim.pth'%(now, epoch, loss)
  optim_path = os.path.join(dirroot_ckpt, optim_name)
  print('Save optimizer', optim_name)
  with open(optim_path, 'wb') as f:
    torch.save(optimizer.state_dict(), f)


def main():
  # load environment
  json_path = sys.argv[1]
  opts, trainloader, validloader, testloader, model, criterion, optimizer = options.load_json(json_path)
  os.makedirs(opts['result_path'], exist_ok=True)
  
  # set random seed
  np.random.seed(opts['seed'])
  random.seed(opts['seed'])
  torch.manual_seed(opts['seed'])
  torch.cuda.manual_seed_all(opts['seed'])
  
  # iterate epochs
  if not opts['test_only']:
    min_loss = 1e9
    for epoch in range(opts['start_epoch'], opts['finish_epoch']+1):
      loss = train(trainloader, model, criterion, optimizer, '[%03d/%03d] train'%(epoch, opts['finish_epoch']))
      loss = valid(validloader, model, criterion, optimizer, '[%03d/%03d] valid'%(epoch, opts['finish_epoch']))
      if loss < min_loss:
        min_loss = loss
        save_checkpoint(opts['result_path'], model, optimizer, epoch, loss)
  
  now = datetime.now().strftime('%y%m%d-%H%M')
  submit_name = 'submit-%s.csv'%(now)
  submit_path = os.path.join(opts['result_path'], submit_name)
  test(testloader, model, submit_path, 'Create submit')
  print('Save submit file', submit_name)


if __name__ == "__main__":
  main()
