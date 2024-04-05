import os
import sys
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="/home/ubuntu/stzhao/LLaVA/finetuned_llm_weight/LLaVA-Lightning-7B-vicuna-v1-1")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--conv-mode", type=str, default="llava_v1")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cuda", type=str, default=None)
parser.add_argument("--ds", type=str, default=None)
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--noise_step", type=int, default=None)
parser.add_argument("--use_cog", type=str, default=False)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

sys.path.insert(0,'/home/ubuntu/stzhao/Causal-CoG')

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
from datasets import Dataset
import random
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from inference_engine.llava_interface import build
import torch.backends.cudnn as cudnn

from PIL import Image
import math

ds_collections = {
    'mme': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/mme.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'mmbench_dev': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/mmbench_dev.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'seedbench': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/seedbench.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'pope_adversarial': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/pope_adversarial.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'pope_random': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/pope_random.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'pope_popular': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/pope_popular.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'vsr': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/vsr.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'winoground': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/winoground_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'okvqa': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/okvqa_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'vqav2': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/vqav2_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'vizwiz': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/vizwiz_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
    'gqa': {
        'test_file_path': '/home/ubuntu/stzhao/Causal-CoG/test_data/gqa_reformeval.json',
        'record_logits_path': '/home/ubuntu/stzhao/Causal-CoG/logits_recorded',
        'results_path': '/home/ubuntu/stzhao/Causal-CoG/reults',
        'max_new_length': 10,
        'prompt': {}
    },
}


all_options = ['A', 'B', 'C', 'D', 'E', 'F']

class VQADataset:

    def __init__(self, annotation_file_path):
        self.annotation_file = json.load(open(annotation_file_path, "r"))

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, idx):

        item = self.annotation_file[idx]
        x = {}
        # x['image'] = Image.open(item['image_path'])
        x['choices'] = item['choices']
        x['question'] = item['question']
        x['question_id'] = item['question_id']
        x['data_type'] = "image"
        x['data_path'] = item['image_path']
        x['gt_answers'] = item['gt_answers']

        return x
    

class DataCollator(object):

    def __init__(self):
        pass


    def __call__(self, x):


        data_batched = {}
        data_batched['choices'] = [sample['choices'] for sample in x]
        data_batched['question'] = [sample['question'] for sample in x]
        data_batched['question_id'] = [sample['question_id'] for sample in x]
        data_batched['data_path'] = [sample['data_path'] for sample in x]
        data_batched['data_type'] = [sample['data_type'] for sample in x]
        data_batched['gt_answers'] = [sample['gt_answers'] for sample in x]

        return data_batched





def setup_seeds(args):
    # seed = config.run_cfg.seed + get_rank()
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def inference(args):
    model = build(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']
    if not os.path.exists(data['results_path']+"/directio"):
        os.makedirs(data['results_path']+"/directio")
    answers_file = data['results_path'] + f"/{str(args.ds)}.jsonl"
    annotation_file = json.load(open(annotation_file_path, "r"))
    ans_file = open(answers_file, "w")

    for item in tqdm(annotation_file):
        x = {}
        x['image'] = Image.open(item['image_path'])
        x['choices'] = item['choices']
        x['question'] = item['question']

        idx = item['question_id']
        x['data_type'] = "image"
        x['data_path'] = None

        losses = model(x)
        class_ranks = torch.argsort(losses, dim=-1).cpu()
        pred_id = all_options[class_ranks[0]]
        
        ans = {}
        ans['index'] = idx
        ans['pred'] = pred_id
        # ans['gt'] = all_options[item['choices'].index(item['gt_answers'])]
        ans['gt'] = item['gt_answers']

        ans = json.dumps(ans)
        ans_file.write(ans+"\n")

    ans_file.close()


def inference_debias(args):
    model = build(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']
    # if not os.path.exists(data['results_path']+"/debias"):
    #     os.makedirs(data['results_path']+"/debias")
    # answers_file = os.path.join(data['record_logits_path'], model_name, args.ds) + f"/{str(args.ds)}.jsonl"
    annotation_file = json.load(open(annotation_file_path, "r"))
    # ans_file = open(answers_file, "w")

    logits_recorded = {}
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []
    logits_recorded['question'] = []
    logits_recorded['image_path'] = []

    for item in tqdm(annotation_file):
        x = {}
        x['image'] = Image.open(item['image_path'])
        x['choices'] = item['choices']
        x['question'] = item['question']

        idx = item['question_id']
        x['data_type'] = "image"
        x['data_path'] = None

        losses, logits_img, logits_ques = model.forward_debias(x)
        class_ranks = torch.argsort(losses, dim=-1).cpu()
        pred_id = all_options[class_ranks[0]]

        logits_recorded['logits_img'].append(logits_img)
        logits_recorded['logits_ques'].append(logits_ques)
        logits_recorded['gt_answer'].append(item['gt_answers'])
        logits_recorded['question_id'].append(idx)
        logits_recorded['question'].append(item['question'])
        logits_recorded['image_path'].append(item['image_path'])
        
        ans = {}
        ans['index'] = idx
        ans['pred'] = pred_id
        # ans['gt'] = all_options[item['choices'].index(item['gt_answers'])]
        ans['gt'] = item['gt_answers']

        ans = json.dumps(ans)
        # ans_file.write(ans+"\n")

    # ans_file.close()
    dataset = Dataset.from_dict(logits_recorded)
    save_file_path = os.path.join(data['record_logits_path'], model_name, args.ds)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    dataset.save_to_disk(save_file_path)


def inference_debias_batched_llava(args):
    model = build(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']

    vqadataset = VQADataset(annotation_file_path)
    datacollator = DataCollator()
    dataloader = DataLoader(vqadataset, batch_size=args.batchsize, shuffle=False, collate_fn=datacollator)

    logits_recorded = {}
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []
    logits_recorded['question'] = []
    logits_recorded['image_path'] = []

    for data_batched in tqdm(dataloader):

        losses, logits_img, logits_ques = model.forward_debias_batched(args, data_batched)
        logits_recorded['logits_img'].extend(logits_img)
        logits_recorded['logits_ques'].extend(logits_ques)
        logits_recorded['question_id'].extend(data_batched['question_id'])
        logits_recorded['gt_answer'].extend(data_batched['gt_answers'])
        logits_recorded['image_path'].extend(data_batched['data_path'])
        logits_recorded['question'].extend(data_batched['question'])


    dataset = Dataset.from_dict(logits_recorded)
    if args.noise_step is not None:
        model_name = model_name + f"_noisestep{args.noise_step}"
    save_file_path = os.path.join(data['record_logits_path'], model_name, args.ds)
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    dataset.save_to_disk(save_file_path)




def inference_cog(args):
    model = build(args)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data
    data = ds_collections[args.ds]
    annotation_file_path = data['test_file_path']
    if not os.path.exists(data['results_path'] + "/causal_cog"):
        os.makedirs(data['results_path'] + "/causal_cog")
    answers_file = data['results_path'] + f"/causal_cog/{str(args.ds)}.jsonl"
    annotation_file = json.load(open(annotation_file_path, "r"))
    ans_file = open(answers_file, "w")

    logits_recorded = {}
    logits_recorded['logits_con_img'] = []
    logits_recorded['logits_img'] = []
    logits_recorded['logits_ques'] = []
    logits_recorded['description'] = []
    logits_recorded['pred_answer'] = []
    logits_recorded['gt_answer'] = []
    logits_recorded['question_id'] = []


    for item in tqdm(annotation_file):
        x = {}
        x['image'] = Image.open(item['image_path'])
        x['choices'] = item['choices']
        x['question'] = item['question']

        idx = item['question_id']
        x['data_type'] = "image"
        x['data_path'] = None

        losses, logits_con_img, logits_img, logits_ques, description = model.forward_cog_sc(x)
        class_ranks = torch.argsort(losses, dim=-1).cpu()
        pred_id = all_options[class_ranks[0]]

        logits_recorded['logits_con_img'].append(logits_con_img)
        logits_recorded['logits_img'].append(logits_img)
        logits_recorded['logits_ques'].append(logits_ques)
        logits_recorded['description'].append(description)
        logits_recorded['pred_answer'].append(pred_id)
        logits_recorded['gt_answer'].append(all_options[item['choices'].index(item['gt_answers'])])
        logits_recorded['question_id'].append(idx)


        ans = {}
        ans['question_id'] = idx
        ans['pred_answer'] = pred_id
        ans['gt_answer'] = all_options[item['choices'].index(item['gt_answers'])]
        # ans['context'] = description

        ans = json.dumps(ans)
        ans_file.write(ans+"\n")

    ans_file.close()
    dataset = Dataset.from_dict(logits_recorded)
    if not os.path.exists(data['record_logits_path']):
        os.makedirs(data['record_logits_path'])
    dataset.save_to_disk(data['record_logits_path'])

if __name__ == "__main__":
    setup_seeds(args)
    
    print(args.use_cog)

    start_time = time.time()

    if args.use_cog == "True":
        inference_cog(args)
    else:
        print(args.ds)
        inference_debias_batched_llava(args)

    end_time = time.time()
    run_time = (end_time - start_time) / 60
    print(f"time spent on evaluating {args.ds}: {run_time} minutes, use_cog is {args.use_cog}.")
