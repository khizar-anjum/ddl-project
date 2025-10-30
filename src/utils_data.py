import re
import json
import os

import torch
import torch.distributed as dist

import utils

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file



from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def fix_coco_json(path_in, path_out):
    with open(path_in, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):  # if karpathy format (list of dicts)
        print("Karpathy format detected — converting to COCO format.")
        images = []
        annotations = []
        for i, item in enumerate(data):
            images.append({"id": item['image_id']})
            annotations.append({
                "id": i,
                "image_id": item['image_id'],
                "caption": item['caption']
            })
        data = {
            "info": {},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "type": "captions"
        }
    else:
        # if missing required COCO fields
        data.setdefault("info", {})
        data.setdefault("licenses", [])
        data.setdefault("type", "captions")

    with open(path_out, 'w') as f:
        json.dump(data, f)

    print(f"✅ Fixed and saved: {path_out}")

def dict_to_list(input_file, output_file):

    #input_file = 'fixed-coco-style-result.json'
    #output_file = 'coco_result_final.json'

    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract the actual list of objects
    if isinstance(data, dict):
        for key in ['annotations', 'results', 'images', 'captions']:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError(f"Cannot find valid list in {input_file}")

    # Now data must be a list of {image_id, caption}
    if not isinstance(data, list):
        raise ValueError("Result file must be a list of {image_id, caption}")

    # Ensure each item has proper keys
    clean_data = []
    for d in data:
        if isinstance(d, dict):
            if 'image_id' in d and 'caption' in d:
                clean_data.append({'image_id': int(d['image_id']), 'caption': str(d['caption'])})
            else:
                print("⚠️ Skipping malformed entry:", d)
        else:
            print("⚠️ Skipping non-dict entry:", d)

    with open(output_file, 'w') as f:
        json.dump(clean_data, f)

    print(f"✅ Fixed result file written to {output_file}, total entries: {len(clean_data)}")

def coco_caption_eval(coco_gt_root, results_file, split):
    #urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
    #        'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    #filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    #download_url(urls[split],coco_gt_root)
    filename = 'captions_val2014.json'
    #filename = 'coco_karpathy_val_gt.json'
    #filename= 'captions_val2014_1000S.json'
    annotation_file = os.path.join(coco_gt_root,filename)    
    # create coco object and coco_result object
    fix_coco_json(annotation_file,'fixed_coco_style_val_annotation.json')
    #coco = COCO(annotation_file)
    fix_coco_json(results_file,'fixed-coco-style-result.json')
    #fix_coco_results('fixed-coco-style-result.json')
    dict_to_list('fixed-coco-style-result.json','coco_result_final.json')
    data = json.load(open('coco_result_final.json'))
    print(type(data), len(data), data[0])
    #coco_result = coco.loadRes(results_file)
    coco= COCO('fixed_coco_style_val_annotation.json')
    coco_result = coco.loadRes('coco_result_final.json')
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval
