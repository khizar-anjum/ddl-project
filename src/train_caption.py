'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''

import argparse
import os
#import ruamel.yaml as YAML
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from blip import blip_decoder
import utils
from utils import cosine_lr_schedule
from coco_data import create_dataset, create_sampler, create_loader
from utils_data import save_result, coco_caption_eval

def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Caption Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i, (image, caption, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)       
        
        loss = model(image, caption)      
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

"""
@torch.no_grad()
def evaluate(model, data_loader, device, config):

    #Evaluate BLIP captioning model on validation set.

    #Computes both validation loss and generated captions, 
    #logs validation loss via metric_logger.
    
    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'
    print_freq = 10

    result = []
    total_loss = 0.0
    count = 0

    for i, (image, caption, image_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)
        
        # Forward pass for loss computation
        text = model.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(device)
        image_embeds = model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        decoder_targets = text.input_ids.masked_fill(text.input_ids == model.tokenizer.pad_token_id, -100)
        decoder_targets[:, :model.prompt_length] = -100

        output = model.text_decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True,
        )

        loss = output.loss
        total_loss += loss.item()
        count += 1
        metric_logger.update(valid_loss=loss.item())

        # Caption generation
        captions = model.generate(
            image,
            sample=False,
            num_beams=config['num_beams'],
            max_length=config['max_length'],
            min_length=config['min_length']
        )

        for caption_text, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption_text})

    avg_valid_loss = total_loss / count if count > 0 else 0.0
    metric_logger.synchronize_between_processes()
    print(f"\nValidation Loss: {avg_valid_loss:.4f}")

    return result, avg_valid_loss
"""

@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    for image, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)       
        
        captions = model.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
        
        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
  
    return result

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset = create_dataset('caption_coco', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset,val_dataset], [True,False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, val_loader = create_loader([train_dataset, val_dataset],samplers,
                                                          batch_size=[config['batch_size']]*2,num_workers=[4,4],
                                                          is_trains=[True, False], collate_fns=[None,None])         

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])
    print("Model is created")
    model = model.to(device)   
    
    print("Make Model ddp distributed ")
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    print("Model is made distributed")

    print("setting the optimizer...")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    best = 0
    best_epoch = 0
    # 🔹 Initialize lists to track losses
    train_losses = []
    val_losses = []

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats = train(model, train_loader, optimizer, epoch, device) 
            train_loss = float(train_stats['loss'])
            train_losses.append(train_loss)  # 🔹 store train loss

        val_result = evaluate(model_without_ddp, val_loader, device, config)  
        #val_result, val_loss = evaluate(model, val_loader, device, config)
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')        
        #val_losses.append(val_loss)  # 🔹 store validation loss
        
        #test_result = evaluate(model_without_ddp, test_loader, device, config)  
        #test_result_file = save_result(test_result, args.result_dir, 'test_epoch%d'%epoch, remove_duplicate='image_id')  

        if utils.is_main_process():   
            coco_val = coco_caption_eval(config['ann_root'],val_result_file,'val')
            #coco_test = coco_caption_eval(config['coco_gt_root'],test_result_file,'test')
            
            if args.evaluate:            
                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},
                             #**{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                   
            else:             
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

                if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                    best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                    best_epoch = epoch                
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in coco_val.eval.items()},
                             #**{f'test_{k}': v for k, v in coco_test.eval.items()},                       
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "metric.json"),"w") as json_file:
                    json.dump(log_stats, json_file, indent=2)
                #with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    #f.write(json.dumps(log_stats) + "\n")     
        """
        # 🔹 Plot Train and Validation Loss Curves
        if utils.is_main_process():
            plt.figure(figsize=(8,6))
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training vs Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'train_val_loss_curve.png'))
            plt.close()
            print(f"✅ Saved train/val loss curve to {os.path.join(args.output_dir, 'train_val_loss_curve.png')}")
        """
        if args.evaluate: 
            break
        dist.barrier()     

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../config/caption_coco.yaml')
    parser.add_argument('--output_dir', default='../output/Caption_coco')        
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
    #config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    #yaml = YAML(typ='safe')   # or 'rt' for round-trip mode
    #with open(args.config, 'r') as f:
    #    config = yaml.load(f)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
