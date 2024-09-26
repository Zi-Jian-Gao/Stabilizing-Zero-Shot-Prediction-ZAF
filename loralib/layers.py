#  ------------------------------------------------------------------------------------------
#  adapted from code with the following copyright:
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
        agent=None
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.agent = agent  # global context
        numA = self.get_num_adapters()
        if numA > 1:
            self.merge_weights = False

        if self.ada_weights_enabled():
            self.lora_ada_weights = nn.ParameterList([nn.Parameter(self.weight.new_ones((i + 1,))) for i in range(numA)])

    def get_ada_weight(self, ix):
        if self.ada_weights_enabled():
            ada_row = min(self.agent.model_task_id, len(self.lora_ada_weights) - 1)
            return self.lora_ada_weights[ada_row][ix]
        return 1

    def ada_weights_enabled(self):
        return (self.get_num_adapters() > 1) and self.agent.ada_weights

    def get_num_adapters(self):
        numA = 1
        if (self.agent is not None) and self.agent.multi:
            numA = self.agent.get_num_tasks()
        if (self.agent is not None) and self.agent.ema:
            numA = self.agent.get_num_tasks()
        return numA

    def apply_lock_policy(self):
        #在这里要修改一下，ema的情况下multi-lora一直训练的是第一个所以应该训练
        numA = self.get_num_adapters()
        if self.agent.ema == True:
            self.lora_A[0].requires_grad = True
            self.lora_B[0].requires_grad = True
            self.lora_A[1].requires_grad = False
            self.lora_B[1].requires_grad = False
            if self.agent.type =='mix':
                self.lora_A[2].requires_grad = False
                self.lora_B[2].requires_grad = False
            if self.ada_weights_enabled():
                pass
        else:
            if numA > 1:
                for i in range(numA - 1):
                    self.lora_A[i].requires_grad = False
                    self.lora_B[i].requires_grad = False
                    if self.ada_weights_enabled():
                        self.lora_ada_weights[i].requires_grad = False

    def should_exec(self, ix):
        numA = self.get_num_adapters()
        if (numA == 1) and ((self.agent is None) or (ix <= self.agent.model_task_id)):
            return True
        # if hasattr(self.agent, 'model_task_id'):
        if ix > self.agent.model_task_id:
            return False
        return True

class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        agent = None,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights, agent=agent)
        # Actual trainable parameters
        if r > 0:
            numA = self.get_num_adapters()
            if numA == 1:
                self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
                self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            else:
                self.lora_A = nn.ParameterList([nn.Parameter(self.weight.new_zeros((r, num_embeddings))) for i in range(numA)])
                self.lora_B = nn.ParameterList([nn.Parameter(self.weight.new_zeros((embedding_dim, r))) for i in range(numA)])
                self.apply_lock_policy()
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            if isinstance(self.lora_A, nn.ParameterList):
                numA = self.get_num_adapters()
                #emalora一开始就有两个lora，所以应该要在这里加一个初始化
                if self.agent.ema == True:
                    for i in range(numA):
                        nn.init.zeros_(self.lora_A[i])
                        nn.init.normal_(self.lora_B[i])
                else:
                    assert (len(self.lora_A) == numA) and (len(self.lora_B) == numA)
                    for i in range(numA):
                        nn.init.zeros_(self.lora_A[i])
                        nn.init.normal_(self.lora_B[i])
            else:
                nn.init.zeros_(self.lora_A)
                nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    #改forward,和EMA适配
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                if isinstance(self.lora_A, nn.ParameterList):
                    # 再加一个判断，在测试EMA状态的时候（self.fuse_type = 'ema'），则看第二个lora即ema_lora的输出
                    # 否则测当前训练时候第一个lora的训练时候的准确率
                    if self.agent.ema ==True:
                        if self.agent.fuse_type in ['ema']:
                            # EMA，在ema推理的时候只看第二个ema_lora的
                            after_A = F.embedding(
                                x, self.lora_A[1].T, self.padding_idx, self.max_norm,
                                self.norm_type, self.scale_grad_by_freq, self.sparse
                            )
                            result += (after_A @ self.lora_B[1].T) * self.scaling
                        else:
                            # 正常ema训练的时候看第一个lora的
                            after_A = F.embedding(
                                x, self.lora_A[0].T, self.padding_idx, self.max_norm,
                                self.norm_type, self.scale_grad_by_freq, self.sparse
                            )
                            result += (after_A @ self.lora_B[0].T) * self.scaling

                    else:
                        numA = self.get_num_adapters()
                        assert (len(self.lora_A) == numA) and (len(self.lora_B) == numA)
                        for i in range(numA):
                            if self.should_exec(i):
                                after_A = F.embedding(
                                    x, self.lora_A[i].T, self.padding_idx, self.max_norm,
                                    self.norm_type, self.scale_grad_by_freq, self.sparse
                                )
                                # result += (after_A @ self.lora_B[i].T) * self.scaling
                                result += (after_A @ self.lora_B[i].T) * self.scaling * self.get_ada_weight(i)
                else:
                    if self.should_exec(0):
                        after_A = F.embedding(
                            x, self.lora_A.T, self.padding_idx, self.max_norm,
                            self.norm_type, self.scale_grad_by_freq, self.sparse
                        )
                        result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        agent=None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights, agent=agent)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            numA = self.get_num_adapters()
            if numA == 1:
                self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
                self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            else:
                self.lora_A = nn.ParameterList([nn.Parameter(self.weight.new_zeros((r, in_features))) for i in range(numA)])
                self.lora_B = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_features, r))) for i in range(numA)])
                self.apply_lock_policy()
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            if isinstance(self.lora_A, nn.ParameterList):
                numA = self.get_num_adapters()
                #emalora一开始就有两个lora，所以应该要在这里加一个初始化
                if self.agent.ema == True:
                    for i in range(numA):
                        nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
                        nn.init.zeros_(self.lora_B[i])
                else:
                    assert (len(self.lora_A) == numA) and (len(self.lora_B) == numA)
                    for i in range(numA):
                        nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
                        nn.init.zeros_(self.lora_B[i])
            else:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                if isinstance(self.lora_A, nn.ParameterList):
                    # 再加一个判断，在测试EMA状态的时候（self.fuse_type = 'ema'），则看第二个lora即ema_lora的输出
                    # 否则测当前训练时候第一个lora的训练时候的准确率
                    if self.agent.ema == True:
                        if self.agent.fuse_type in ['ema']:
                            # EMA，在ema推理的时候只看第二个ema_lora的
                            result += (self.lora_dropout(x) @ self.lora_A[1].T @ self.lora_B[1].T) * self.scaling
                        else:
                            # 正常ema训练的时候看第一个lora的
                            result += (self.lora_dropout(x) @ self.lora_A[0].T @ self.lora_B[0].T) * self.scaling
                    else:
                        numA = self.get_num_adapters()
                        assert (len(self.lora_A) == numA) and (len(self.lora_B) == numA)
                        for i in range(numA):
                            if self.should_exec(i):
                                # result += (self.lora_dropout(x) @ self.lora_A[i].T @ self.lora_B[i].T) * self.scaling
                                result += (self.lora_dropout(x) @ self.lora_A[i].T @ self.lora_B[i].T) * self.scaling * self.get_ada_weight(i)
                else:
                    if self.should_exec(0):
                        result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

#no usage
class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), 
                    self.lora_B.data.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False
    
    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), 
                    self.lora_B.data.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), 
                    self.lora_B.unsqueeze(-1), 
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result
            
#no usage
class Conv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False
    
    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x, 
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return nn.Conv2d.forward(self, x)
