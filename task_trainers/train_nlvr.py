'''
 adapted from code with the following copyright:
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import copy
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from itertools import zip_longest

from models.blip_nlvr import blip_nlvr

import utils
from utils import cosine_lr_schedule, warmup_lr_schedule, count_parameters
from data import create_dataset, create_sampler, create_loader, create_zsl_dataset

import loralib as lora

def train(model, data_loader, optimizer, epoch, device, config, agent):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 10
 
    for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if not isinstance(batch_data[1], list):
            image0, image1, text, targets = batch_data
        else:
            image0, pos, neg, idx = batch_data
            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            targets = torch.zeros((len(text,)), dtype=torch.int64)
            targets[:len(pos)] = 1
            image1 = None

        if image1 is not None:
            images = torch.cat([image0, image1], dim=0)
        else:
            images = image0
        images, targets = images.to(device), targets.to(device)   

        loss = model(images, text, targets=targets, train=True, agent=agent)   
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())  
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def train_zsl(model, data_loader,zsl_data_loader,zsl_datasets,samplers,num_workers,optimizer, epoch, device, config, agent):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('zero_shot_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 10
    for i, (batch_data, zsl_batch_data) in enumerate(
            zip_longest(data_loader, zsl_data_loader, fillvalue=(None, None, None, None))):
        if all(item is None for item in batch_data) :
            break
        if all(item is None for item in zsl_batch_data):
            break

        #task data loss cal
        image0, pos, neg, idx = batch_data
        text = pos + neg
        image0 = image0.repeat(2, 1, 1, 1)
        targets = torch.zeros((len(text, )), dtype=torch.int64)
        targets[:len(pos)] = 1
        images = image0
        images, targets = images.to(device), targets.to(device)
        loss1 = model(images, text, targets=targets, train=True, agent=agent)

        if agent.train_distill_type == 'zsl-single' or  agent.train_distill_type == 'ema-zsl-single' or  agent.train_distill_type == 'adv_text_zsl' :
            # zsl data loss cal
            if agent.random ==True:
                image0, pos, neg, idx = zsl_batch_data
                random.shuffle(pos)
                random.shuffle(neg)
                text = pos + neg
                image0 = image0.repeat(2, 1, 1, 1)
                targets = torch.ones((len(text, )), dtype=torch.int64)#do not invlove in calculation
                images = image0
                images, targets = images.to(device), targets.to(device)
                loss2, losses_log = model(images, text, targets=targets, train=True, agent=agent, train_zsl=True)
            else:
                image0, pos, neg, idx = zsl_batch_data
                text = pos
                targets = torch.ones((len(text, )), dtype=torch.int64)#do not invlove in calculation
                images = image0
                images, targets = images.to(device), targets.to(device)
                loss2,losses_log = model(images, text, targets=targets, train=True, agent=agent,train_zsl = True)
        elif agent.train_distill_type == 'zsl-cons'or  agent.train_distill_type == 'ema-zsl-cons' or  agent.train_distill_type == 'adv_text_cons':
            if agent.random ==True:
                image0, pos, neg, idx = zsl_batch_data
                random.shuffle(pos)
                random.shuffle(neg)
                text = pos + neg
                image0 = image0.repeat(2, 1, 1, 1)

                targets = torch.zeros((len(text, )), dtype=torch.int64)#do not invlove in calculation
                targets[:len(pos)] = 1
                images = image0
                images, targets = images.to(device), targets.to(device)
                loss2, losses_log = model(images, text, targets=targets, train=True, agent=agent, train_zsl=True)
            else:
                image0, pos, neg, idx = zsl_batch_data
                text = pos + neg
                image0 = image0.repeat(2, 1, 1, 1)
                targets = torch.zeros((len(text, )), dtype=torch.int64)
                targets[:len(pos)] = 1
                images = image0
                images, targets = images.to(device), targets.to(device)
                loss2,losses_log = model(images, text, targets=targets, train=True, agent=agent,train_zsl = True)

        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        metric_logger.update(zero_shot_loss=loss2.item())

        if i % print_freq == 0 and i != 0:
            print(f"Step: {i}, Loss: {loss.item():.4f}, ce-Loss: {loss1.item():.4f}, zero-shot-Loss: {loss2.item():.4f}, task-Loss: {losses_log}")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, device, config, agent):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        if not isinstance(batch_data[1], list):
            image0, image1, text, targets = batch_data
        else:
            image0, pos, neg, idx = batch_data
            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            targets = torch.zeros((len(text, )), dtype=torch.int64)
            targets[:len(pos)] = 1
            image1 = None
        if image1 is not None:
            images = torch.cat([image0, image1], dim=0)
        else:
            images = image0
        images, targets = images.to(device), targets.to(device)   
        
        prediction = model(images, text, targets=targets, train=False, agent=agent)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def multi_task_evaluate(model, data_loader, device, config, agent):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        if not isinstance(batch_data[1], list):
            image0, image1, text, targets = batch_data
        else:
            image0, pos, neg, idx = batch_data
            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            targets = torch.zeros((len(text,)), dtype=torch.int64)
            targets[:len(pos)] = 1
            image1 = None
        if image1 is not None:
            images = torch.cat([image0, image1], dim=0)
        else:
            images = image0
        images, targets = images.to(device), targets.to(device)

        predictions = []
        for iT in range(agent.get_num_tasks()):
            agent.prep_model4task(iT)
            prediction = model(images, text, targets=targets, train=False, agent=agent)
            if isinstance(prediction, tuple):
                fuse_weights = prediction[1].detach().cpu()
                prediction = prediction[0]
            predictions.append(prediction.detach().cpu())
        agent.prep_model4task(-1)


        if agent.fuse_type in ['last']:
            prediction = torch.stack([x.softmax(dim=-1) for x in predictions], dim=-1)
        else:
            prediction = torch.stack(predictions, dim=-1)

        if agent.fuse_type in ['last']:
            prediction = prediction[:, :, -1]
        else:
            raise NotImplementedError(f'Unsupported fuse type: {agent.fuse_type}')

        _, pred_class = prediction.max(1)
        accuracy = (targets.cpu() == pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

        
def main(args, config, eval=False,test_ema=False):

    agent = args['agent']


    # change all "args." to args[""]
    args['result_dir'] = os.path.join(args['out_dir'], 'result')
    if utils.is_main_process(): Path(args['result_dir']).mkdir(parents=True, exist_ok=True)
    device = args['device']

    #### Dataset #### 
    print("Creating dataset")
    dataset_pass_dict = {'training_data_sample':args['training_data_sample']}
    datasets = create_dataset(config['dataset'], config, dataset_pass_dict)

    if args['distributed']:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,False,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    batch_size=[config['batch_size_train'][agent.task_id],config['batch_size_test'],config['batch_size_test']]
    train_loader, val_loader, test_loader = create_loader(datasets,samplers,batch_size=batch_size,
                                                          num_workers=[args['num_workers'], args['num_workers'], args['num_workers']],is_trains=[True,False,False],
                                                          collate_fns=[None,None,None])

    # agent
    agent = args['agent']

    #### Model #### 
    print("Creating model")
    model, head_not_loaded = blip_nlvr(pretrained=args['pretrained'], image_size=config['image_size'],
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent, single_image_model=('vl-checklist' in config['dataset']))

    model = model.to(device)   
    
    model_without_ddp = model
    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']], find_unused_parameters=True)
        model_without_ddp = model.module

    if not eval and agent.ema:
        if  args['ema_lora'] == 'continual':
            print('continual training lora ')
            pass
        elif  args['ema_lora'] == 'zero' or agent.train_distill_type == 'grassmann':
            print('initial lora with defination')
            lora.lora_initial(model_without_ddp.text_encoder)
            lora.lora_initial(model_without_ddp.visual_encoder)
        elif args['ema_lora'] == 'ema':
            print('initial lora with ema')
            lora.lora_initial_ema(model_without_ddp.text_encoder)
            lora.lora_initial_ema(model_without_ddp.visual_encoder)

    
    if agent.freeze_encoders:
        
        param_to_optim = []
        
        # task heads
        if (not agent.freeze_heads) or head_not_loaded:
            print('Training head')
            param_to_optim += list(model_without_ddp.cls_head.parameters())
        else:
            print('Locking head')
            for p in model_without_ddp.cls_head.parameters():
                p.requires_grad = False

        if agent.lora:
            param_to_optim += list(model_without_ddp.text_encoder.parameters())
            param_to_optim += list(model_without_ddp.visual_encoder.parameters())
            lora.mark_only_lora_as_trainable(model_without_ddp.text_encoder)
            lora.mark_only_lora_as_trainable(model_without_ddp.visual_encoder)
        
        # optimizer
        optimizer = torch.optim.AdamW(params=param_to_optim, lr=config['init_lr'], weight_decay=config['weight_decay'])
        nparam = count_parameters(param_to_optim)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
        nparam = count_parameters(model.parameters())

    # print num trainable params    
    print(f'trainable_parameters = {nparam}')

    # init agent
    if not eval: agent.update_model(model_without_ddp)

    print("Start training")
    start_time = time.time()
    best = 0
    best_epoch = 0

    # flag for no training
    if not eval and args['eval_every'] < 0:
        if utils.is_main_process():  
            torch.save({'model':model_without_ddp.state_dict()}, args['model_save_path'])
        return

    start_epoch = 0

    #load checkpint of current task
    for epoch in range(start_epoch, config['max_epoch']):
        load_file = os.path.join(args['out_dir'], 'checkpoint_%02d.pth'%epoch)
        if os.path.exists(load_file):
            checkpoint = torch.load(load_file)
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best = checkpoint['best']
            best_epoch = checkpoint['best_epoch']

    if test_ema:
        fuse_type = agent.fuse_type
        print('evaluate ema')
        agent = args['agent']
        agent.fuse_type = 'ema'
        print("Creating model")
        model, head_not_loaded = blip_nlvr(pretrained=args['pretrained'], image_size=config['image_size'],
                                           vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                           vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent,
                                           single_image_model=('vl-checklist' in config['dataset']))
        model = model.to(device)
        if args['distributed']:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']],
                                                              find_unused_parameters=True)

        eval_func = evaluate

        test_stats = eval_func(model, test_loader, device, config, agent)

        agent.fuse_type = fuse_type

        if utils.is_main_process():
            return test_stats['acc']
        else:
            return -0.1

    else:
        if agent.task_id != 0:
            print("Creating zsl dataset")
            dataset_pass_dict = {'training_data_sample': args['training_data_sample']}
            zsl_datasets = create_zsl_dataset(config['dataset'], config, dataset_pass_dict)

            if args['distributed']:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()
                samplers = create_sampler(zsl_datasets, [True, False, False], num_tasks, global_rank)
            else:
                samplers = [None, None, None]

            batch_size = [config['batch_size_train'][agent.task_id], config['batch_size_test'], config['batch_size_test']]
            zsl_train_loader, _, _ = create_loader(zsl_datasets, samplers, batch_size=batch_size,
                                                   num_workers=[args['num_workers'], args['num_workers'],
                                                                args['num_workers']], is_trains=[True, False, False],
                                                   collate_fns=[None, None, None])

        best = 0
        for epoch in range(start_epoch, config['max_epoch']):
            if not eval:
                if args['distributed']:
                    train_loader.sampler.set_epoch(epoch)

                cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

                if agent.task_id != 0:
                    train_stats = train_zsl(model, train_loader,zsl_train_loader,zsl_datasets,samplers,args['num_workers'], optimizer, epoch, device, config, agent)
                else:
                    train_stats = train(model, train_loader, optimizer, epoch,  device, config, agent)

                if agent.ema and (epoch + 1) % args['ema_frequency'] == 0:
                    frequency = args['ema_frequency']
                    if args['ema'] == 'epoch':
                        ema_alpha = args['ema_alpha']
                        print(f'epoch EMA begins,current_alpha = {ema_alpha},task_id = {agent.task_id},ema_frequency = {frequency}')
                        lora.update_ema_epoch_lora(model_without_ddp.text_encoder, args['ema_alpha'], agent.task_id,agent.update_both)
                        lora.update_ema_epoch_lora(model_without_ddp.visual_encoder, args['ema_alpha'], agent.task_id,agent.update_both)

                        val_stats = evaluate(model, val_loader, device, config, agent)
                        if args['save_frequency'] == 'best':
                            if float(val_stats['acc']) > best:
                                best = float(val_stats['acc'])
                                torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])
                        elif args['save_frequency'] == 'every':
                            torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])
                    else:
                        pass


            if eval or (epoch + 1) % args['eval_every'] == 0:
                eval_func = evaluate
                if agent.multi:
                    eval_func = multi_task_evaluate

                val_stats = eval_func(model, val_loader, device, config, agent)
                test_stats = eval_func(model, test_loader, device, config, agent)

                if utils.is_main_process():
                    if eval:
                        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                                     **{f'test_{k}': v for k, v in test_stats.items()},
                                     }
                        with open(os.path.join(args['out_dir'], "log.txt"), "a") as f:
                            f.write(json.dumps(log_stats) + "\n")

                    else:
                        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                     **{f'val_{k}': v for k, v in val_stats.items()},
                                     **{f'test_{k}': v for k, v in test_stats.items()},
                                     'epoch': epoch,
                                     }


                        if float(val_stats['acc']) > best :
                            best = float(val_stats['acc'])
                            best_epoch = epoch
                            if not agent.ema:
                                if args['save_frequency'] == 'best':
                                    torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])

                        if not agent.ema and args['save_frequency'] == 'every':
                            torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])


                        with open(os.path.join(args['out_dir'], "log.txt"), "a") as f:
                            f.write(json.dumps(log_stats) + "\n")

                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'config': config,
                            'epoch': epoch,
                            'best': best,
                            'best_epoch': best_epoch,
                        }
                        torch.save(save_obj, os.path.join(args['out_dir'], 'checkpoint_%02d.pth' % epoch))
                        epoch_old = epoch - 1
                        old_file = os.path.join(args['out_dir'], 'checkpoint_%02d.pth' % epoch_old)
                        if os.path.isfile(old_file):
                            os.remove(old_file)

                        print(f'Finished epoch {epoch} best epoch is {best_epoch} with acc {best}')

            dist.barrier()
            torch.cuda.empty_cache()
            if eval:
                if utils.is_main_process():
                    return test_stats['acc']
                else:
                    return -0.1
