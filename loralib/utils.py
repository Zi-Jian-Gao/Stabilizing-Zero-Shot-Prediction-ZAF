#  ------------------------------------------------------------------------------------------
#  adapted from code with the following copyright:
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer

import torch.nn.init as init
import math


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        #     print(f'Freezing {n}')
        # else:
        #     if p.requires_grad:
        #         print(f'Optimizing {n}')
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def update_ema_task_lora(model: nn.Module,task_id, bias: str = 'none') -> None:
    lora0_params = {}
    # 第一遍遍历：收集lora0的参数
    for name, param in model.named_parameters():
        if 'lora_A.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_A.0', 'lora_A.1')] = param.clone()
        if 'lora_B.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_B.0', 'lora_B.1')] = param.clone()

    # 第二遍遍历：将lora0的参数复制到lora1
    for name, param in model.named_parameters():
        if name in lora0_params:
            # 将lora0的参数直接复制给lora1
            if task_id == 0:
                param.data = lora0_params[name].data
            else:
                param.data = 1/(task_id+1) * lora0_params[name].data + (task_id)/ (task_id+1) * param.data

def update_ema_epoch_lora(model: nn.Module,alpha,task_id,update_both, bias: str = 'none') -> None:
    lora0_params = {}
    # 第一遍遍历：收集lora0的参数
    for name, param in model.named_parameters():
        if 'lora_A.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_A.0', 'lora_A.1')] = param.clone()

        if task_id == 0:
            if 'lora_B.0' in name:
                # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
                lora0_params[name.replace('lora_B.0', 'lora_B.1')] = param.clone()
        #lora B一直用任务0的
        elif update_both:
            if 'lora_B.0' in name:
                # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
                lora0_params[name.replace('lora_B.0', 'lora_B.1')] = param.clone()


    # 第二遍遍历：将lora0的参数复制到lora1
    for name, param in model.named_parameters():
        if name in lora0_params:
            # 将lora0的参数直接复制给lora1
            if task_id == 0:
                param.data = lora0_params[name].data
            else:
                param.data = (1-alpha) * lora0_params[name].data + alpha * param.data



def update_ema_epoch_lora_B_merge(model: nn.Module,alpha,task_id,update_both, bias: str = 'none') -> None:
    lora0_params = {}
    # 第一遍遍历：收集lora0的参数
    for name, param in model.named_parameters():
        if 'lora_A.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_A.0', 'lora_A.1')] = param.clone()

        # if task_id == 0:
        if 'lora_B.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_B.0', 'lora_B.1')] = param.clone()
        #
        # elif update_both:
        #     if 'lora_B.0' in name:
        #         # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
        #         lora0_params[name.replace('lora_B.0', 'lora_B.1')] = param.clone()


    # 第二遍遍历：将lora0的参数复制到lora1
    for name, param in model.named_parameters():
        if name in lora0_params:
            # 将lora0的参数直接复制给lora1
            if task_id == 0:
                param.data = lora0_params[name].data
            else:
                if 'lora_B.1' in name and update_both:
                    param.data = (1 - alpha) * lora0_params[name].data + alpha * param.data
                elif 'lora_B.1' in name and not update_both:
                    param.data = lora0_params[name].data + param.data
                else:
                    param.data = (1-alpha) * lora0_params[name].data + alpha * param.data


def update_ema_epoch_mix_lora(model: nn.Module,alpha,task_id, bias: str = 'none') -> None:
    lora0_params = {}
    # 第一遍遍历：收集lora0的参数
    for name, param in model.named_parameters():
        if 'lora_A.0' in name:
            # 以新的键存储参数，将'.0'替换为'.2'以便后续匹配lora2
            lora0_params[name.replace('lora_A.0', 'lora_A.2')] = param.clone()
        if 'lora_B.0' in name:
            # 以新的键存储参数，将'.0'替换为'.2'以便后续匹配lora2
            lora0_params[name.replace('lora_B.0', 'lora_B.2')] = param.clone()

    # 第二遍遍历：将lora0的参数复制到lora1
    for name, param in model.named_parameters():
        if name in lora0_params:
            # 将lora0的参数直接复制给lora1
            if task_id == 0:
                param.data = lora0_params[name].data
            else:
                param.data = (1-alpha) * lora0_params[name].data + alpha * param.data

def update_ema_task_mix_lora(model: nn.Module,task_id, bias: str = 'none') -> None:
    lora2_params = {}
    # 第一遍遍历：收集lora2的参数
    for name, param in model.named_parameters():
        if 'lora_A.2' in name:
            # 以新的键存储参数，将'.2'替换为'.1'以便后续匹配lora1
            lora2_params[name.replace('lora_A.2', 'lora_A.1')] = param.clone()
        if 'lora_B.2' in name:
            # 以新的键存储参数，将'.2'替换为'.1'以便后续匹配lora1
            lora2_params[name.replace('lora_B.2', 'lora_B.1')] = param.clone()

    # 第二遍遍历：将lora0的参数复制到lora1
    for name, param in model.named_parameters():
        if name in lora2_params:
            # 将lora0的参数直接复制给lora1
            if task_id == 0:
                param.data = lora2_params[name].data
            else:
                param.data = 1/(task_id+1) * lora2_params[name].data + (task_id)/ (task_id+1) * param.data

def lora_initial(model: nn.Module, bias: str = 'none') -> None:
    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if 'lora_A.0' in name:
            # 使用Kaiming/He均匀初始化来初始化lora_A.0
            init.kaiming_uniform_(param, a=math.sqrt(5))
        if 'lora_B.0' in name:
            # 为lora_B.0初始化为零
            init.zeros_(param)

def lora_initial_ema(model: nn.Module, bias: str = 'none') -> None:
    lora1_params = {}
    # 第一遍遍历：收集lora1的参数
    for name, param in model.named_parameters():
        if 'lora_A.1' in name:
            # 以新的键存储参数，将'.1'替换为'.0'以便后续匹配lora0
            lora1_params[name.replace('lora_A.1', 'lora_A.0')] = param.clone()
        if 'lora_B.1' in name:
            # 以新的键存储参数，将'.1'替换为'.0'以便后续匹配lora0
            lora1_params[name.replace('lora_B.1', 'lora_B.0')] = param.clone()

    # 第二遍遍历：将lora1的参数复制到lora0
    for name, param in model.named_parameters():
        if name in lora1_params:
            # 将lora0的参数直接复制给lora1
            param.data = lora1_params[name].data




def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
