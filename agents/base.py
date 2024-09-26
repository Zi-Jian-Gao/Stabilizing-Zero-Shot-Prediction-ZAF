import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import numpy as np
from pathlib import Path
import shutil
import os
import utils
import torch.distributed as dist


#这里面是lora的各种setting和utils
class Base:
    def __init__(self, agent_config):
        super(Base, self).__init__()
        self.config = agent_config
        self.init_model_ckpt = self.config['pretrained']
        self.oracle = self.config['oracle']
        self.mu = self.config['mu']
        self.prompting = False
        self.sep_dec_prompts = False
        self.prompt_type = None
        self.lora = False  # for LoRa
        self.r = None  # for LoRa
        self.multi = False  # for LoRa
        self.ada_weights = False  # for LoRa

        # 该参数控制在后面进行evaluate时的推理方式
        #对于只有一个lora直接推理即可
        #对于multi_lora和adv_lora,last是看前面所有lora的共同的输出
        #对于ema_lora，ema是只看第二个ema lora的结果
        self.fuse_type = 'max'  # for LoRa

        #random sample for single and cons sample pairs
        self.random = False

        self.train_fuse_type = 'current'  # for LoRa
        self.train_distill_type = None  # for LoRa
        self.model_task_id = 1e7  # for LoRa


        #首先根据该ema参数来设置lora的num，第二个为ema_lora,第一个为task-specific lora
        #其次控制lora的更新方式task-specific lora正常更新(要注意这个时候只看这单个lora！不需要ema_lora的输出)
        #ema_lora在每个epoch后通过ema更新 （注意task0必须直接复制）
        #注意还要控制parameter的梯度,设置为第一个为true，第二个为false
        self.ema = False  # for EMA_LoRa

        self.update_both = True

        self.type = self.config['type']

        self.layer_keys = None
        self.args = self.config['global_args']

        # saving/loading results and checkpoints
        self.output_dir = self.config['output_dir']
        self.task_model_dir = os.path.join(self.output_dir, 'task_models')
        if utils.is_main_process(): Path(self.task_model_dir).mkdir(parents=True, exist_ok=True)
        dist.barrier()

        # pre-training model ckpt
        if self.init_model_ckpt is not None and self.init_model_ckpt != 'None':
            pre_check_file = os.path.join(self.task_model_dir, '_pre.pth')
            if utils.is_main_process(): shutil.copyfile(self.init_model_ckpt,pre_check_file)
            dist.barrier()

            # dict of ckpts
            self.model_ckpt_history = {'pretrained':pre_check_file}
            
            # list of ckpts
            self.model_ckpt_list = []
            self.model_ckpt_list.append(pre_check_file)
        else:
            self.model_ckpt_history = {}
            self.model_ckpt_list = []
        self.model_ckpt_load = copy.deepcopy(self.model_ckpt_list)
        
        # other dirs
        self.task_dir_dict = {}
        self.task_config_dict = {}

        # dynamic
        self.tasks = []
        self.task_id = 0
        self.current_task = 'init'

        # memory
        self.coreset = []

    def get_num_tasks(self):
        if self.ema:
            if  self.type !='mix':
            #第一个为持续update的lora，第二个为ema_lora
                return 2
            else:
                return 3
        return len(self.model_ckpt_list) - (1 if 'pretrained' in self.model_ckpt_history else 0)



#对ema没用，只有AdvTextMultiLoRa有用
    def prep_model4task(self, task_num=-1, force=False):
        if (task_num < 0) and (not force):
            self.model_task_id = 1e7
        else:
            self.model_task_id = task_num

    def increment_task(self, task_str, task_config):
        
        # add task
        self.tasks.append(task_str)

        # create task directory
        self.task_dir_dict[task_str] = os.path.join(self.output_dir, task_str)
        if utils.is_main_process(): Path(self.task_dir_dict[task_str]).mkdir(parents=True, exist_ok=True)
        self.task_dir = self.task_dir_dict[task_str]
        
        # add ckpt files
        self.model_ckpt_history[task_str] = os.path.join(self.task_model_dir, task_str+'.pth')
        if not self.oracle: self.model_ckpt_load = copy.deepcopy(self.model_ckpt_list)
        self.model_ckpt_list.append(self.model_ckpt_history[task_str])

        # save task config
        self.task_config_dict[task_str] = task_config

    def finish_task (self):
        self.task_id += 1

    def regularize(self, state_dict):
        return torch.zeros((1,), requires_grad=True).cuda()

    def update_model(self, model):
        pass

class Naive(Base):
    def __init__(self, agent_config):
        super(Naive, self).__init__(agent_config)

    def regularize(self, state_dict):
        return torch.zeros((1,), requires_grad=True).cuda()