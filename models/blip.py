'''
 adapted from code with the following copyright:
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from models.vit import VisionTransformer, interpolate_pos_embed
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
import math

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file

from functools import partial
from models.vit import Block as SA_Block
from timm.models.layers import trunc_normal_

def init_tokenizer(multi_lingual=False):
    if multi_lingual:
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0, agent=None):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate,
                                           agent=agent
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate,
                                           agent=agent
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model, url_or_filename_list, args=None):
    ############################################
    # IF UPDATING THIS FUNCTION, BLIP NLVR HAS 
    # UNIQUE LOAD_CHECKPOINT THAT NEEDS TO BE
    # CHANGED AS WELL!
    ############################################
    if not isinstance(url_or_filename_list, list):
        url_or_filename_list = [url_or_filename_list]

    for url_or_filename in url_or_filename_list:
        if url_or_filename is not None and url_or_filename != 'None':
            if is_url(url_or_filename):
                cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
                checkpoint = torch.load(cached_file, map_location='cpu') 
            elif os.path.isfile(url_or_filename):        
                checkpoint = torch.load(url_or_filename, map_location='cpu') 
            else:
                raise RuntimeError(f'checkpoint url or path ({url_or_filename}) is invalid')
                
            state_dict = checkpoint['model']
            
            state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
            if 'visual_encoder_m.pos_embed' in model.state_dict().keys() and 'visual_encoder_m.pos_embed' in state_dict.keys():
                state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                                model.visual_encoder_m)

            if isinstance(model.tokenizer, list):
                blip_w = state_dict['text_encoder.embeddings.word_embeddings.weight']
                if model.text_encoder.embeddings.word_embeddings.weight.shape != blip_w.shape: # it may be that we are loading a model that already has the corrected embedding layer
                    toks_w = [(blip_w if x[-1] == 'blip' else x[1].word_embeddings.weight) for x in model.tokenizer]
                    new_weights = torch.cat(toks_w, dim=0).detach()
                    state_dict['text_encoder.embeddings.word_embeddings.weight'] = new_weights
                    state_dict['text_encoder_m.embeddings.word_embeddings.weight'] = new_weights

            if args is not None:
                if args['flush_queue']:
                    for key in list(state_dict.keys()):
                        if 'queue' in key:
                            print(f'Deleting {key} from checkpoint')
                            del state_dict[key]

            mdsd = model.state_dict()
            sdk = state_dict.keys()
            for key in mdsd.keys():
                if key in sdk:
                    if state_dict[key].shape!=mdsd[key].shape:
                        del state_dict[key]
                elif 'lora_' in key:
                    # it could be that the model has a sequence of loras while the saved model has a single lora for the same
                    key_ = '.'.join(key.split('.')[:-1])
                    if ('lora_' in key_) and (key_ in sdk):  # this means we stripped a number being the ModuleList index
                        state_dict[key] = state_dict[key_]
                        del state_dict[key_]

            if False:
                if url_or_filename != url_or_filename_list[0]:
                    diff_w = []
                    missing_w = []
                    for key in mdsd.keys():
                        if key in state_dict:
                            if not torch.all(mdsd[key].eq(state_dict[key])):
                                d = torch.max(torch.abs(mdsd[key] - state_dict[key]))
                                diff_w.append((key, d))
                        else:
                            missing_w.append(key)
                    # print(f'Missing: {missing_w}')
                    print(f'Different: {diff_w}')

            msg = model.load_state_dict(state_dict,strict=False)
            print('load checkpoint from %s'%url_or_filename)  
    return model,msg