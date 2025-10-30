'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import warnings
warnings.filterwarnings("ignore")

from vit import VisionTransformer, interpolate_pos_embed
#from med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertLMHeadModel, BertConfig, BertModel #fix:Megha: By eliminating the med bert issues  
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file
import logging

class BLIP_Base(nn.Module):
    def __init__(self,                 
                 med_config = '../config/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                 
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)  

        
    def forward(self, image, caption, mode):
        
        assert mode in ['image', 'text', 'multimodal'], "mode parameter must be image, text, or multimodal"
        text = self.tokenizer(caption, return_tensors="pt").to(image.device) 
        
        if mode=='image':    
            # return image features
            image_embeds = self.visual_encoder(image)             
            return image_embeds
        
        elif mode=='text':
            # return text features
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')  
            return text_output.last_hidden_state
        
        elif mode=='multimodal':
            # return multimodel features
            image_embeds = self.visual_encoder(image)    
            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)      
            
            text.input_ids[:,0] = self.tokenizer.enc_token_id
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )              
            return output.last_hidden_state
        
        
        
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = '../config/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """            
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        med_config.is_decoder = True    #Fix: Megha: transformers >= 4.25 compatibility
        med_config.add_cross_attention = True #Fix: Megha: transformers >= 4.25 compatibility
        self.text_decoder = BertLMHeadModel(config=med_config)    
        print("Decoder:", med_config.is_decoder)  #debug print
        print("Cross Attention:", med_config.add_cross_attention) #debug print
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        
    def forward(self, image, caption):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss
        
        return loss_lm
        
    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
        
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device) 
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 

        if sample:
            #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  do_sample=True,
                                                  top_p=top_p,
                                                  num_return_sequences=1,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, 
                                                  repetition_penalty=1.1,                                            
                                                  **model_kwargs)
        else:
            #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids,
                                                  max_length=max_length,
                                                  min_length=min_length,
                                                  num_beams=num_beams,
                                                  eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id,     
                                                  repetition_penalty=repetition_penalty,
                                                  **model_kwargs)            
            
        captions = []    
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            captions.append(caption[len(self.prompt):])
        return captions
    

def blip_decoder(pretrained='',**kwargs):
    print("inside blip_decoder() function")
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        #assert(len(msg.missing_keys)==0)
        if len(msg.missing_keys) > 0:
            print("⚠️ Missing keys:", msg.missing_keys)
        if len(msg.unexpected_keys) > 0:
            print("⚠️ Unexpected keys:", msg.unexpected_keys)
    return model    
    
def blip_feature_extractor(pretrained='',**kwargs):
    model = BLIP_Base(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model        

def init_tokenizer():
    print("debug: inside init_tokenizer() function")
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        cache_dir='/pscratch/sd/m/megha89/ddl-project/hf_cache',  # ✅ local cache path
        local_files_only=False                        # can still download if missing
    )
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    #tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    #tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    # ✅ FIX 2: Modern tokenizer versions have better handling of special tokens
    print("debug1: inside init_tokenizer() function")
    special_tokens_dict = {
        'bos_token': '[DEC]',
        'additional_special_tokens': ['[ENC]']
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    print("debug2: inside init_tokenizer() function")
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    print("debug3: inside init_tokenizer() function")
    return tokenizer



def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    if vit=='base':
        vision_width = 768
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = VisionTransformer(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

"""
The fix: for assertionerror
def load_checkpoint(model, url_or_filename, tokenizer=None):
    logger = logging.getLogger(__name__)

    def is_url(u):
        from urllib.parse import urlparse
        p = urlparse(u)
        return p.scheme in ("http", "https")

    if is_url(url_or_filename):
        from timm.models.hub import download_cached_file
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')

    state_dict = checkpoint.get('model', checkpoint)

    # ---- keep your pos_embed interpolation if present ----
    if 'visual_encoder.pos_embed' in state_dict and 'visual_encoder.pos_embed' in model.state_dict():
        try:
            state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(
                state_dict['visual_encoder.pos_embed'],
                model.visual_encoder
            )
            logger.info("Interpolated visual_encoder.pos_embed to match model.")
        except Exception as e:
            logger.warning(f"pos_embed interpolation failed: {e}")

    if 'visual_encoder_m.pos_embed' in state_dict and 'visual_encoder_m.pos_embed' in model.state_dict():
        try:
            state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(
                state_dict['visual_encoder_m.pos_embed'],
                model.visual_encoder_m
            )
            logger.info("Interpolated visual_encoder_m.pos_embed to match model.")
        except Exception as e:
            logger.warning(f"visual_encoder_m pos_embed interpolation failed: {e}")

    # Remove keys with mismatching shapes to avoid load errors (we'll report them)
    model_state = model.state_dict()
    keys_to_delete = []
    for k, v in list(state_dict.items()):
        if k in model_state:
            if isinstance(v, torch.Tensor) and v.shape != model_state[k].shape:
                logger.warning(f"Checkpoint key {k} has shape {v.shape}, model expects {model_state[k].shape}. Skipping this key.")
                keys_to_delete.append(k)
        else:
            # keep non-matching keys too; they will be reported by load_state_dict
            pass
    for k in keys_to_delete:
        del state_dict[k]

    # Load with strict=False
    msg = model.load_state_dict(state_dict, strict=False)

    # Report
    if len(msg.missing_keys) > 0:
        logger.warning(f"Missing keys when loading pretrained checkpoint: {len(msg.missing_keys)}")
        for k in msg.missing_keys[:50]:
            logger.warning(f"  MISSING: {k}")
        if len(msg.missing_keys) > 50:
            logger.warning(f"  ... and {len(msg.missing_keys)-50} more")

    if len(msg.unexpected_keys) > 0:
        logger.warning(f"Unexpected keys in checkpoint: {len(msg.unexpected_keys)}")
        for k in msg.unexpected_keys[:50]:
            logger.warning(f"  UNEXPECTED: {k}")
        if len(msg.unexpected_keys) > 50:
            logger.warning(f"  ... and {len(msg.unexpected_keys)-50} more")

    # If tokenizer provided, try to handle embedding size mismatch by resizing the model embeddings
    try:
        # Many models use model.get_input_embeddings()/resize_token_embeddings API
        if tokenizer is not None:
            # If text decoder present and embedding mismatch, resize
            if hasattr(model, "text_decoder"):
                td = model.text_decoder
                if hasattr(td, "resize_token_embeddings"):
                    current_vocab = td.get_input_embeddings().weight.shape[0]
                    desired_vocab = len(tokenizer)
                    if current_vocab != desired_vocab:
                        logger.info(f"Resizing text decoder token embeddings: {current_vocab} -> {desired_vocab}")
                        td.resize_token_embeddings(desired_vocab)
            else:
                # try top-level model (for BLIP_Base or other cases)
                if hasattr(model, "resize_token_embeddings"):
                    current_vocab = model.get_input_embeddings().weight.shape[0]
                    desired_vocab = len(tokenizer)
                    if current_vocab != desired_vocab:
                        logger.info(f"Resizing model token embeddings: {current_vocab} -> {desired_vocab}")
                        model.resize_token_embeddings(desired_vocab)
    except Exception as e:
        logger.warning(f"Failed to resize token embeddings automatically: {e}")

    logger.info(f"Loaded checkpoint from {url_or_filename}")
    return model, msg
"""

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg
    
