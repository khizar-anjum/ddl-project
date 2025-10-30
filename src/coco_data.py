import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

#from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_caption_eval, coco_karpathy_retrieval_eval
#from data.nocaps_dataset import nocaps_eval
#from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
#from data.vqa_dataset import vqa_dataset
#from data.nlvr_dataset import nlvr_dataset
#from data.pretrain_dataset import pretrain_dataset
from randaugment import RandomAugment

import os
import json

from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from utils_data import pre_caption

class coco_train(Dataset):
    def __init__(self, transform, train_image_root, ann_root, max_words=30, prompt=''):
        '''
        image_root (string): Root directory of images (e.g. coco/train2014/)
        ann_root (string): directory containing captions_train2014.json
        '''
        filename = 'captions_train2014.json'
        ann_path = os.path.join(ann_root, filename)

        # --- Load official COCO JSON ---
        coco_data = json.load(open(ann_path, 'r'))

        # Build mapping from image_id -> filename
        imgid2file = {img['id']: img['file_name'] for img in coco_data['images']}

        # Create flat list of {image_id, caption, image}
        self.annotation = []
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            image = imgid2file[image_id]
            self.annotation.append({
                'image_id': image_id,
                'caption': caption,
                'image': image
            })
        # ✅ Limit to first 1000 samples for faster experiments #added by Megha
        #self.annotation = self.annotation[:1000]

        self.transform = transform
        self.image_root = train_image_root
        self.max_words = max_words
        self.prompt = prompt

        # Build image ID lookup for evaluation consistency
        self.img_ids = {}
        for n, ann in enumerate(self.annotation):
            self.img_ids.setdefault(ann['image_id'], n)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        return image, caption, self.img_ids[ann['image_id']]

class coco_caption_eval(Dataset):
    def __init__(self, transform, val_image_root, ann_root, split='val'):
        '''
        image_root (string): Root directory of images (e.g. coco/val2014/)
        ann_root (string): directory containing captions_val2014.json
        '''
        filename = 'captions_val2014.json'
        ann_path = os.path.join(ann_root, filename)

        coco_data = json.load(open(ann_path, 'r'))
        imgid2file = {img['id']: img['file_name'] for img in coco_data['images']}

        self.annotation = []
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            image = imgid2file[image_id]
            self.annotation.append({
                'image_id': image_id,
                'caption': ann['caption'],
                'image': image
            })
        
        #✅ Limit to first 1000 samples for validation as well
        #self.annotation = self.annotation[:1000]
        #with open('/pscratch/sd/m/megha89/ddl-project/data/annotations/captions_val2014_1000S.json', "w") as f:
        #    json.dump(self.annotation, f, indent=2)

        self.transform = transform
        self.image_root = val_image_root
        #self.prompt = 'A picture of'
        #self.max_words= 30

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        img_id = ann['image_id']
        #caption = self.prompt + pre_caption(ann['caption'], self.max_words)
        return image, img_id
                            
def create_dataset(dataset, config, min_scale=0.5):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])  
        
    if dataset=='pretrain':
        #dataset = pretrain_dataset(config['train_file'], config['laion_path'], transform_train)              
        return dataset  
    
    elif dataset=='caption_coco':   
        train_dataset = coco_train(transform_train, config['train_image_dir'], config['ann_root'], prompt=config['prompt'])
        val_dataset = coco_caption_eval(transform_test, config['val_image_dir'], config['ann_root'], 'val')
        #test_dataset = coco_karpathy_caption_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return train_dataset, val_dataset #, test_dataset
    
    elif dataset=='nocaps':   
        #val_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        #test_dataset = nocaps_eval(transform_test, config['image_root'], config['ann_root'], 'test')   
        return val_dataset #, test_dataset   
    
    elif dataset=='retrieval_coco':          
        #train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'])
        #val_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        #test_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset #, test_dataset    
    
    elif dataset=='retrieval_flickr':          
        #train_dataset = flickr30k_train(transform_train, config['image_root'], config['ann_root'])
        #val_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val') 
        #test_dataset = flickr30k_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          
        return train_dataset, val_dataset #, test_dataset     
    
    elif dataset=='vqa': 
        #train_dataset = vqa_dataset(transform_train, config['ann_root'], config['vqa_root'], config['vg_root'], 
        #                            train_files = config['train_files'], split='train') 
        #test_dataset = vqa_dataset(transform_test, config['ann_root'], config['vqa_root'], config['vg_root'], split='test')
        return train_dataset #, test_dataset
    
    elif dataset=='nlvr': 
        #train_dataset = nlvr_dataset(transform_train, config['image_root'], config['ann_root'],'train')
        #val_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'val')
        #test_dataset = nlvr_dataset(transform_test, config['image_root'], config['ann_root'],'test')     
        return train_dataset, val_dataset #, test_dataset   
    
    
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

