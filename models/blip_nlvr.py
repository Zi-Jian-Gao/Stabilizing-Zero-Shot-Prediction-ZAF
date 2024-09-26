'''
 adapted from code with the following copyright:
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from models.med import BertConfig
from models.nlvr_encoder import BertModel
from models.med import BertModel as BertModelSingleImageEHS
from models.vit import interpolate_pos_embed
from models.blip import create_vit, init_tokenizer, is_url

from timm.models.hub import download_cached_file

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import os

class BLIP_NLVR(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,  
                 agent = None,
                 single_image_model = False
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1, agent=agent)
        self.tokenizer = init_tokenizer()

        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.single_image_model = single_image_model
        if self.single_image_model:
            self.text_encoder = BertModelSingleImageEHS(config=med_config, add_pooling_layer=False, agent=agent)
        else:
            self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)
                    
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                )  

    def forward(self, image, text, targets, train=True, agent=None, feature_forward=False, train_zsl = False):

        origin_text = text

        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        if not self.single_image_model:
            image0_embeds, image1_embeds = torch.split(image_embeds,targets.size(0))

        blip_enc_token_id = None # currently the sequence of tokenizers is required to have blip as one of them, otherwise a new token needs to be found
        if not isinstance(self.tokenizer, list):
            text = self.tokenizer(text, padding='longest', return_tensors="pt").to(image.device)
            text_input_ids = text.input_ids
            text_attention_mask = text.attention_mask
            blip_enc_token_id = self.tokenizer.enc_token_id
        else:
            text_ = []
            for tok in self.tokenizer:
                text_.append(tok[0](text, padding='longest', return_tensors="pt").to(image.device))
                text_[-1].input_ids[text_[-1].input_ids != 0] += tok[2]
                if tok[-1] == 'blip':
                    blip_enc_token_id = tok[0].enc_token_id + tok[2]
            text_input_ids = torch.cat([x.input_ids for x in text_], dim=1)
            text_attention_mask = torch.cat([x.attention_mask for x in text_], dim=1)

        assert blip_enc_token_id is not None
        text_input_ids[:,0] = blip_enc_token_id
        # print(text.input_ids[:,0:10])   

        if self.single_image_model:
            output = self.text_encoder(text_input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
        else:
            output = self.text_encoder(text_input_ids,
                                       attention_mask = text_attention_mask,
                                       encoder_hidden_states = [image0_embeds,image1_embeds],
                                       encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                                 image_atts[image0_embeds.size(0):]],
                                       return_dict = True,
                                      )
        #current lora feature

        hidden_state = output.last_hidden_state[:,0,:]        
        prediction = self.cls_head(hidden_state)

        if feature_forward:
            return [image_embeds.detach(), hidden_state.detach()]

        eps = 1e-7
        if train:
            if not train_zsl:
                loss = F.cross_entropy(prediction, targets)

                if agent.train_distill_type is not None:
                    if agent.train_distill_type == 'adv_text' or  agent.train_distill_type == 'adv_text_zsl':
                        assert agent.args.freeze_text_emb #without this it is not clear how we transfer text in soft form between task models
                        alpha = agent.args.loss_alpha
                        if agent.args.auto_scale_alpha and (agent.task_id > 0):
                            alpha /= agent.task_id
                        with torch.no_grad():
                            base_text_emb = self.text_encoder._embed_only(text_input_ids).detach()
                        for task_id in range(agent.task_id):
                            if agent.args.adv_last_only:
                                if task_id < (agent.task_id - agent.args.adv_num_last):
                                    continue
                            step_sz = agent.args.adv_step_sz
                            num_steps = agent.args.num_adv_iters
                            ix2use = (targets > 0)
                            adv_text_emb = base_text_emb[ix2use].detach()
                            for iStep in range(num_steps):
                                adv_text_emb.requires_grad = True
                                _, txt_emb = self.get_task_feats(image[ix2use], adv_text_emb, text_attention_mask[ix2use], task_id, agent, no_detach=True, text_is_emb=True)
                                del _
                                adv_pred = self.cls_head(txt_emb).softmax(dim=1)
                                if iStep == 0:
                                    orig_prev_pred = adv_pred[:, 1].detach()
                                if iStep < (num_steps - 1):
                                    adv_loss =  - adv_pred[:, 1].mean()
                                    if agent.args.adv_pos:
                                        adv_loss = - adv_loss
                                    grad = torch.autograd.grad(adv_loss, adv_text_emb, create_graph=False)
                                    adv_text_emb = adv_text_emb + step_sz * grad[0].sign()
                                adv_text_emb = adv_text_emb.detach()
                            adv_pred = adv_pred[:,1].detach()
                            adv_pred = torch.stack([adv_pred, orig_prev_pred], dim=-1)
                            del orig_prev_pred
                            _, cur_adv_txt_emb = self.get_task_feats(image[ix2use], adv_text_emb, text_attention_mask[ix2use], 1e7, agent, no_detach=True, text_is_emb=True)
                            cur_adv_pred = self.cls_head(cur_adv_txt_emb).softmax(dim=-1)
                            _prediction = prediction.softmax(dim=-1)
                            cur_adv_pred = torch.stack([cur_adv_pred[:, 1], _prediction[ix2use, 1]], dim=-1)
                            loss += alpha * torch.abs((adv_pred[:, 1] - adv_pred[:, 0]) - (cur_adv_pred[:, 1] - cur_adv_pred[:, 0])).mean()

                    elif agent.train_distill_type == 'dis_ema' and agent.task_id !=0:
                        assert agent.args.freeze_text_emb #without this it is not clear how we transfer text in soft form between task models
                        alpha = agent.args.loss_alpha
                        if agent.args.auto_scale_alpha and (agent.task_id > 0):
                            alpha /= agent.task_id

                        #current lora feature: hidden_state

                        #get ema lora feature

                        fuse_type = agent.fuse_type
                        agent.fuse_type = 'ema'
                        with torch.no_grad():
                            image_embeds = self.visual_encoder(image)
                            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                            if not self.single_image_model:
                                image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))
                            blip_enc_token_id = None  # currently the sequence of tokenizers is required to have blip as one of them, otherwise a new token needs to be found
                            if not isinstance(self.tokenizer, list):
                                text = self.tokenizer(origin_text, padding='longest', return_tensors="pt").to(image.device)
                                text_input_ids = text.input_ids
                                text_attention_mask = text.attention_mask
                                blip_enc_token_id = self.tokenizer.enc_token_id
                            else:
                                text_ = []
                                for tok in self.tokenizer:
                                    text_.append(tok[0](text, padding='longest', return_tensors="pt").to(image.device))
                                    text_[-1].input_ids[text_[-1].input_ids != 0] += tok[2]
                                    if tok[-1] == 'blip':
                                        blip_enc_token_id = tok[0].enc_token_id + tok[2]
                                text_input_ids = torch.cat([x.input_ids for x in text_], dim=1)
                                text_attention_mask = torch.cat([x.attention_mask for x in text_], dim=1)
                            assert blip_enc_token_id is not None
                            text_input_ids[:, 0] = blip_enc_token_id
                            # print(text.input_ids[:,0:10])

                            if self.single_image_model:
                                output = self.text_encoder(text_input_ids,
                                                           attention_mask=text_attention_mask,
                                                           encoder_hidden_states=image_embeds,
                                                           encoder_attention_mask=image_atts,
                                                           return_dict=True,
                                                           )
                            else:
                                output = self.text_encoder(text_input_ids,
                                                           attention_mask=text_attention_mask,
                                                           encoder_hidden_states=[image0_embeds, image1_embeds],
                                                           encoder_attention_mask=[image_atts[:image0_embeds.size(0)],
                                                                                   image_atts[image0_embeds.size(0):]],
                                                           return_dict=True,
                                                           )


                            ema_hidden_state = output.last_hidden_state[:, 0, :]

                        # 调整回到训练模式,确保下个task训练正常
                        agent.fuse_type = fuse_type

                        loss += alpha * torch.dist(hidden_state, ema_hidden_state, 2).mean()

                    elif agent.train_distill_type == 'dis_pre_ema' and agent.task_id !=0:
                        assert agent.args.freeze_text_emb #without this it is not clear how we transfer text in soft form between task models
                        alpha = agent.args.loss_alpha
                        if agent.args.auto_scale_alpha and (agent.task_id > 0):
                            alpha /= agent.task_id

                        #current lora feature: hidden_state

                        #get ema lora feature

                        fuse_type = agent.fuse_type
                        agent.fuse_type = 'ema'
                        with torch.no_grad():
                            image_embeds = self.visual_encoder(image)
                            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                            if not self.single_image_model:
                                image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))
                            blip_enc_token_id = None  # currently the sequence of tokenizers is required to have blip as one of them, otherwise a new token needs to be found
                            if not isinstance(self.tokenizer, list):
                                text = self.tokenizer(origin_text, padding='longest', return_tensors="pt").to(image.device)
                                text_input_ids = text.input_ids
                                text_attention_mask = text.attention_mask
                                blip_enc_token_id = self.tokenizer.enc_token_id
                            else:
                                text_ = []
                                for tok in self.tokenizer:
                                    text_.append(tok[0](text, padding='longest', return_tensors="pt").to(image.device))
                                    text_[-1].input_ids[text_[-1].input_ids != 0] += tok[2]
                                    if tok[-1] == 'blip':
                                        blip_enc_token_id = tok[0].enc_token_id + tok[2]
                                text_input_ids = torch.cat([x.input_ids for x in text_], dim=1)
                                text_attention_mask = torch.cat([x.attention_mask for x in text_], dim=1)
                            assert blip_enc_token_id is not None
                            text_input_ids[:, 0] = blip_enc_token_id
                            # print(text.input_ids[:,0:10])

                            if self.single_image_model:
                                output = self.text_encoder(text_input_ids,
                                                           attention_mask=text_attention_mask,
                                                           encoder_hidden_states=image_embeds,
                                                           encoder_attention_mask=image_atts,
                                                           return_dict=True,
                                                           )
                            else:
                                output = self.text_encoder(text_input_ids,
                                                           attention_mask=text_attention_mask,
                                                           encoder_hidden_states=[image0_embeds, image1_embeds],
                                                           encoder_attention_mask=[image_atts[:image0_embeds.size(0)],
                                                                                   image_atts[image0_embeds.size(0):]],
                                                           return_dict=True,
                                                           )


                            ema_hidden_state = output.last_hidden_state[:, 0, :]

                        # 调整回到训练模式,确保下个task训练正常
                        agent.fuse_type = fuse_type

                        ema_prediction = self.cls_head(ema_hidden_state)

                        loss_kd = _KD_loss(
                            prediction,
                            ema_prediction,
                            2,
                        )

                        loss += alpha * loss_kd
                    elif agent.train_distill_type == 'grassmann' and agent.task_id != 0:
                        visual_distance_loss = count_encoder_lora_distance(self.visual_encoder,image)
                        text_distance_loss = count_encoder_lora_distance(self.text_encoder,image)

                        loss = loss - (visual_distance_loss + text_distance_loss) / 2



                    else:
                        pass
                    # raise NotImplementedError(f'Unsupported train distill type: {agent.train_distill_type}')
            else:
                #cal zsl loss
                if agent.train_distill_type is not None:
                    if agent.train_distill_type == 'zsl-single' or  agent.train_distill_type == 'adv_text_zsl':
                        total_loss = torch.zeros((1,), requires_grad=True).to(image.device)
                        losses = []
                        alpha = 1.0
                        # if agent.args.auto_scale_alpha and (agent.task_id > 0):
                        #     alpha /= agent.task_id

                        agent.prep_model4task(agent.task_id - 1, force=True)
                        with torch.no_grad():
                            image_embeds = self.visual_encoder(image)
                            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                            if not self.single_image_model:
                                image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))
                            blip_enc_token_id = None  # currently the sequence of tokenizers is required to have blip as one of them, otherwise a new token needs to be found
                            if not isinstance(self.tokenizer, list):
                                text = self.tokenizer(origin_text, padding='longest', return_tensors="pt").to(
                                    image.device)
                                text_input_ids = text.input_ids
                                text_attention_mask = text.attention_mask
                                blip_enc_token_id = self.tokenizer.enc_token_id
                            else:
                                text_ = []
                                for tok in self.tokenizer:
                                    text_.append(
                                        tok[0](text, padding='longest', return_tensors="pt").to(image.device))
                                    text_[-1].input_ids[text_[-1].input_ids != 0] += tok[2]
                                    if tok[-1] == 'blip':
                                        blip_enc_token_id = tok[0].enc_token_id + tok[2]
                                text_input_ids = torch.cat([x.input_ids for x in text_], dim=1)
                                text_attention_mask = torch.cat([x.attention_mask for x in text_], dim=1)
                            assert blip_enc_token_id is not None
                            text_input_ids[:, 0] = blip_enc_token_id
                            # print(text.input_ids[:,0:10])

                            if self.single_image_model:
                                output = self.text_encoder(text_input_ids,
                                                           attention_mask=text_attention_mask,
                                                           encoder_hidden_states=image_embeds,
                                                           encoder_attention_mask=image_atts,
                                                           return_dict=True,
                                                           )
                            else:
                                output = self.text_encoder(text_input_ids,
                                                           attention_mask=text_attention_mask,
                                                           encoder_hidden_states=[image0_embeds, image1_embeds],
                                                           encoder_attention_mask=[
                                                               image_atts[:image0_embeds.size(0)],
                                                               image_atts[image0_embeds.size(0):]],
                                                           return_dict=True,
                                                           )

                            ema_hidden_state = output.last_hidden_state[:, 0, :]

                            zsl_prediction = self.cls_head(ema_hidden_state)

                        cur_result =  prediction.softmax(dim=-1)
                        past_result = zsl_prediction.softmax(dim=-1)
                        loss = alpha * torch.abs(past_result[:, 1] - cur_result[:, 1]).mean()
                        total_loss += loss
                        losses.append(loss.item())
                        agent.prep_model4task(-1)

                        return total_loss, losses

                    elif agent.train_distill_type == 'ema-zsl-single':
                        alpha = agent.args.loss_alpha
                        if agent.args.auto_scale_alpha and (agent.task_id > 0):
                            alpha /= agent.task_id
                        fuse_type = agent.fuse_type
                        agent.fuse_type = 'ema'
                        with torch.no_grad():
                            image_embeds = self.visual_encoder(image)
                            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                            if not self.single_image_model:
                                image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))
                            blip_enc_token_id = None  # currently the sequence of tokenizers is required to have blip as one of them, otherwise a new token needs to be found
                            if not isinstance(self.tokenizer, list):
                                text = self.tokenizer(origin_text, padding='longest', return_tensors="pt").to(
                                    image.device)
                                text_input_ids = text.input_ids
                                text_attention_mask = text.attention_mask
                                blip_enc_token_id = self.tokenizer.enc_token_id
                            else:
                                text_ = []
                                for tok in self.tokenizer:
                                    text_.append(
                                        tok[0](text, padding='longest', return_tensors="pt").to(image.device))
                                    text_[-1].input_ids[text_[-1].input_ids != 0] += tok[2]
                                    if tok[-1] == 'blip':
                                        blip_enc_token_id = tok[0].enc_token_id + tok[2]
                                text_input_ids = torch.cat([x.input_ids for x in text_], dim=1)
                                text_attention_mask = torch.cat([x.attention_mask for x in text_], dim=1)
                            assert blip_enc_token_id is not None
                            text_input_ids[:, 0] = blip_enc_token_id
                            # print(text.input_ids[:,0:10])

                            if self.single_image_model:
                                output = self.text_encoder(text_input_ids,
                                                           attention_mask=text_attention_mask,
                                                           encoder_hidden_states=image_embeds,
                                                           encoder_attention_mask=image_atts,
                                                           return_dict=True,
                                                           )
                            else:
                                output = self.text_encoder(text_input_ids,
                                                           attention_mask=text_attention_mask,
                                                           encoder_hidden_states=[image0_embeds, image1_embeds],
                                                           encoder_attention_mask=[
                                                               image_atts[:image0_embeds.size(0)],
                                                               image_atts[image0_embeds.size(0):]],
                                                           return_dict=True,
                                                           )

                            ema_hidden_state = output.last_hidden_state[:, 0, :]

                            zsl_prediction = self.cls_head(ema_hidden_state)


                        agent.fuse_type = fuse_type

                        cur_result =  prediction.softmax(dim=-1)
                        past_result = zsl_prediction.softmax(dim=-1)
                        loss = alpha * torch.abs(past_result[:, 1] - cur_result[:, 1]).mean()

                        return loss, loss.item()
                    elif agent.train_distill_type == 'zsl-cons':
                        total_loss = torch.zeros((1,), requires_grad=True).to(image.device)
                        losses = []  # 创建一个列表来存储每个 task_id 的损失值

                        alpha = agent.args.loss_alpha
                        if agent.args.auto_scale_alpha and (agent.task_id > 0):
                            alpha /= agent.task_id
                        with torch.no_grad():
                            base_text_emb = self.text_encoder._embed_only(text_input_ids).detach()
                        for task_id in range(agent.task_id):
                            agent.prep_model4task(task_id, force=True)
                            with torch.no_grad():
                                image_embeds = self.visual_encoder(image)
                                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                                if not self.single_image_model:
                                    image0_embeds, image1_embeds = torch.split(image_embeds, targets.size(0))
                                blip_enc_token_id = None  # currently the sequence of tokenizers is required to have blip as one of them, otherwise a new token needs to be found
                                if not isinstance(self.tokenizer, list):
                                    text = self.tokenizer(origin_text, padding='longest', return_tensors="pt").to(
                                        image.device)
                                    text_input_ids = text.input_ids
                                    text_attention_mask = text.attention_mask
                                    blip_enc_token_id = self.tokenizer.enc_token_id
                                else:
                                    text_ = []
                                    for tok in self.tokenizer:
                                        text_.append(
                                            tok[0](text, padding='longest', return_tensors="pt").to(image.device))
                                        text_[-1].input_ids[text_[-1].input_ids != 0] += tok[2]
                                        if tok[-1] == 'blip':
                                            blip_enc_token_id = tok[0].enc_token_id + tok[2]
                                    text_input_ids = torch.cat([x.input_ids for x in text_], dim=1)
                                    text_attention_mask = torch.cat([x.attention_mask for x in text_], dim=1)
                                assert blip_enc_token_id is not None
                                text_input_ids[:, 0] = blip_enc_token_id
                                # print(text.input_ids[:,0:10])

                                if self.single_image_model:
                                    output = self.text_encoder(text_input_ids,
                                                               attention_mask=text_attention_mask,
                                                               encoder_hidden_states=image_embeds,
                                                               encoder_attention_mask=image_atts,
                                                               return_dict=True,
                                                               )
                                else:
                                    output = self.text_encoder(text_input_ids,
                                                               attention_mask=text_attention_mask,
                                                               encoder_hidden_states=[image0_embeds, image1_embeds],
                                                               encoder_attention_mask=[
                                                                   image_atts[:image0_embeds.size(0)],
                                                                   image_atts[image0_embeds.size(0):]],
                                                               return_dict=True,
                                                               )

                                ema_hidden_state = output.last_hidden_state[:, 0, :]

                                zsl_prediction = self.cls_head(ema_hidden_state)

                            cur_result =  prediction.softmax(dim=-1)
                            past_result = zsl_prediction.softmax(dim=-1)

                            cons_num = int(past_result.size(0)/2)

                            past_pos_result_dev = past_result[:cons_num][:,1] - past_result[cons_num:][:,1]
                            cur_pos_result_dev = cur_result[:cons_num][:,1] - cur_result[cons_num:][:,1]

                            loss = alpha * torch.abs(past_pos_result_dev - cur_pos_result_dev).mean()
                            total_loss += loss
                            losses.append(loss.item())
                            agent.prep_model4task(-1)

                        return total_loss, losses

            return loss
        else:
            return prediction

    def _get_task_feats_(self, image, text_input_ids, text_attention_mask, no_detach=False, text_is_emb=False):
        image_embeds_q = self.visual_encoder(image)
        if not no_detach:
            image_embeds_q = image_embeds_q.detach()
        image_atts_q = torch.ones(image_embeds_q.size()[:-1], dtype=torch.long).to(image.device)
        text_embeds_q = self.text_encoder(text_input_ids if (not text_is_emb) else None,
                                          encoder_embeds=(text_input_ids if text_is_emb else None),
                                          attention_mask=text_attention_mask,
                                          encoder_hidden_states=image_embeds_q,
                                          encoder_attention_mask=image_atts_q,
                                          return_dict=True,
                                          ).last_hidden_state[:, 0, :]
        if not no_detach:
            text_embeds_q = text_embeds_q.detach()
        image_embeds_q = image_embeds_q.mean(dim=1)
        if not no_detach:
            image_embeds_q = image_embeds_q.detach()
        return image_embeds_q, text_embeds_q
    def get_task_feats(self, image, text_input_ids, text_attention_mask, task_id, agent, no_detach=False, text_is_emb=False):
        agent.prep_model4task(task_id, force=True)
        if not no_detach:
            with torch.no_grad():
                image_embeds_q, text_embeds_q = self._get_task_feats_(image, text_input_ids, text_attention_mask,  no_detach, text_is_emb)
        else:
            image_embeds_q, text_embeds_q = self._get_task_feats_(image, text_input_ids, text_attention_mask,  no_detach, text_is_emb)
        agent.prep_model4task(-1)
        return image_embeds_q, text_embeds_q
    
def blip_nlvr(pretrained='',**kwargs):
    model = BLIP_NLVR(**kwargs)
    head_not_loaded = True
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
        head_not_loaded = False
        for k in msg.missing_keys:
            if 'cls_head' in k:
                head_not_loaded = True
    return model, head_not_loaded

def load_checkpoint(model, url_or_filename_list):

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
                raise RuntimeError('checkpoint url or path is invalid')
            state_dict = checkpoint['model']
            
            state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 

            if hasattr(model, 'single_image_model') and model.single_image_model:
                for key in list(state_dict.keys()):
                    if 'crossattention.self' in key:
                        new_key0 = key.replace('self0', 'self')
                        # new_key1 = key.replace('self','self1')
                        state_dict[new_key0] = state_dict[key]
                        # state_dict[new_key1] = state_dict[key]
                    elif 'crossattention.output.dense' in key:
                        new_key0 = key.replace('dense0', 'dense')
                        # new_key1 = key.replace('dense','dense1')
                        state_dict[new_key0] = state_dict[key]
                        # state_dict[new_key1] = state_dict[key]
                # pass
            else:
                for key in list(state_dict.keys()):
                    if 'crossattention.self.' in key:
                        new_key0 = key.replace('self','self0')
                        new_key1 = key.replace('self','self1')
                        state_dict[new_key0] = state_dict[key]
                        state_dict[new_key1] = state_dict[key]
                    elif 'crossattention.output.dense.' in key:
                        new_key0 = key.replace('dense','dense0')
                        new_key1 = key.replace('dense','dense1')
                        state_dict[new_key0] = state_dict[key]
                        state_dict[new_key1] = state_dict[key]
                 
            if isinstance(model.tokenizer, list):
                blip_w = state_dict['text_encoder.embeddings.word_embeddings.weight']
                if model.text_encoder.embeddings.word_embeddings.weight.shape != blip_w.shape: # it may be that we are loading a model that already has the corrected embedding layer
                    toks_w = [(blip_w if x[-1] == 'blip' else x[1].word_embeddings.weight) for x in model.tokenizer]
                    new_weights = torch.cat(toks_w, dim=0).detach()
                    state_dict['text_encoder.embeddings.word_embeddings.weight'] = new_weights

            mdsd = model.state_dict()
            sdk = state_dict.keys()
            for key in mdsd.keys():
                if key in sdk:
                    if state_dict[key].shape != mdsd[key].shape:
                        del state_dict[key]
                elif 'lora_' in key:
                    # it could be that the model has a sequence of loras while the saved model has a single lora for the same
                    key_ = '.'.join(key.split('.')[:-1])
                    if ('lora_' in key_) and (
                            key_ in sdk):  # this means we stripped a number being the ModuleList index
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
            

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def grassmann_distance_torch(A, B):

    Q_A, _ = torch.linalg.qr(A)
    Q_B, _ = torch.linalg.qr(B)

    M = torch.matmul(Q_A.T, Q_B)

    # 步骤3: 计算M的奇异值分解，得到奇异值
    _, S, _ = torch.linalg.svd(M)

    epsilon = 1e-10
    sin_theta_squared = 1.0 - (S + epsilon) ** 2

    distance = torch.sqrt(torch.sum(sin_theta_squared))

    print("Q_A:", torch.isnan(Q_A).any())
    print("Q_B:", torch.isnan(Q_B).any())
    print("M:", torch.isnan(M).any())
    print("S:", torch.isnan(S).any())
    print("sin_theta_squared:", torch.isnan(sin_theta_squared).any())

    return distance

def count_encoder_lora_distance(lora_model,image):
    count = 0
    lora1_params = {}
    with torch.no_grad():
        for name, param in lora_model.named_parameters():
            if 'lora_A.1' in name:
                # 以新的键存储参数，将'.1'替换为'.0'以便后续匹配lora0
                lora1_params[name.replace('lora_A.1', 'lora_A.0')] = param.clone()

    total_distance = torch.zeros((1,), requires_grad=True).to(image.device)

    for name, param in lora_model.named_parameters():
        if name in lora1_params:
            count += 1
            # 计算该lora的距离，不进行原地操作
            current_distance = grassmann_distance_torch(param, lora1_params[name])
            # 累加到total_distance中，不使用原地加法
            total_distance = torch.add(total_distance, current_distance, alpha=1)

    # 计算平均距离，这里不需要梯度
    avg_distance = total_distance / count
    # print(avg_distance)

    return avg_distance

def ortho_penalty(self, t):
    return ((t @ t.T - torch.eye(t.shape[0])) ** 2).mean()
