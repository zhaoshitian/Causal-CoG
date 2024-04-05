import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
import json
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from transformers.models.clip.modeling_clip import CLIPModel
from transformers.models.clip.image_processing_clip import CLIPImageProcessor
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
from datasets import load_from_disk
from torch.nn import functional as F

from evaluate import Evaluator


clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
tokenizer_llava = AutoTokenizer.from_pretrained("/home/ubuntu/stzhao/LLaVA/finetuned_llm_weight/LLaVA-Lightning-7B-vicuna-v1-1")


def debias(args, c):

    all_options = ['A', 'B', 'C', 'D', 'E', 'F']
    # all_options = ['Yes', 'No']
    # all_options = ["Answer: False", "Answer: True"]

    logits_img_list = [[] for _ in range(args.sample_nums)] 
    logits_ques_list = [[] for _ in range(args.sample_nums)] 
    preds_list = [[] for _ in range(args.sample_nums)] 
    gts_list = [[] for _ in range(args.sample_nums)] 
    question_ids_list = [[] for _ in range(args.sample_nums)] 

    logits_recorded = load_from_disk(f"{args.logits_recorded_base_path_1}")
    logits_img_list=logits_recorded['logits_img']
    logits_ques_list=logits_recorded['logits_ques']
    gts_list=logits_recorded['gt_answer']
    question_ids_list=logits_recorded['question_id']

    debias_list = [[(1+c)*a-c*b for a,b in zip(img,ques)] for img, ques in zip(logits_img_list, logits_ques_list)]
    debias_list = [torch.tensor(item) for item in debias_list]
    logits_img_list = [torch.tensor(item) for item in logits_img_list]

    pred_answer_list = []
    pred_answer_debias_list = []

    for debias_logit, logit in zip(debias_list, logits_img_list):
        
        class_ranks = torch.argsort(debias_logit, dim=-1, descending=True).cpu()
        pred_id_debias = all_options[class_ranks[0]]
        pred_answer_debias_list.append(pred_id_debias)

        class_ranks = torch.argsort(logit, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list.append(pred_id)

    # print(pred_answer_debias_list)
    # print(pred_answer_list)
    # print(gts_list)

    nums_correct = sum([a==b for a,b in zip(pred_answer_list, gts_list)])
    nums_correct_debias = sum([a==b for a,b in zip(pred_answer_debias_list, gts_list)])

    print(nums_correct/len(gts_list), nums_correct_debias/len(gts_list), c)


def contrast(args, c):

    all_options = ['A', 'B', 'C', 'D', 'E', 'F']

    logits_img_list_1 = [[] for _ in range(args.sample_nums)] 
    logits_ques_list_1 = [[] for _ in range(args.sample_nums)] 
    preds_list_1 = [[] for _ in range(args.sample_nums)] 
    gts_list_1 = [[] for _ in range(args.sample_nums)] 
    question_ids_list_1 = [[] for _ in range(args.sample_nums)] 

    logits_recorded_1 = load_from_disk(f"{args.logits_recorded_base_path_1}")
    logits_img_list_1 = logits_recorded_1['logits_img']
    logits_ques_list_1 = logits_recorded_1['logits_ques']
    gts_list_1 = logits_recorded_1['gt_answer']
    question_ids_list_1 = logits_recorded_1['question_id']

    logits_img_list_2 = [[] for _ in range(args.sample_nums)] 
    logits_ques_list_2 = [[] for _ in range(args.sample_nums)] 
    preds_list_2 = [[] for _ in range(args.sample_nums)] 
    gts_list_2 = [[] for _ in range(args.sample_nums)] 
    question_ids_list_2 = [[] for _ in range(args.sample_nums)] 

    logits_recorded_2 = load_from_disk(f"{args.logits_recorded_base_path_2}")
    logits_img_list_2 = logits_recorded_2['logits_img']
    logits_ques_list_2 = logits_recorded_2['logits_ques']
    gts_list_2 = logits_recorded_2['gt_answer']
    question_ids_list_2 = logits_recorded_2['question_id']


    contrast_list = [[(1+c)*a-c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_img_list_2)]
    # contrast_list = [[(1+c)*a-c*b for a,b in zip(img_1, img_2)] for img_1, img_2 in zip(logits_img_list_1, logits_ques_list_2)]
    contrast_list = [torch.tensor(item) for item in contrast_list]
    logits_img_list_1 = [torch.tensor(item) for item in logits_img_list_1]

    pred_answer_list = []
    pred_answer_contrast_list = []

    for contrast_logit, logit in zip(contrast_list, logits_img_list_1):
        class_ranks = torch.argsort(contrast_logit, dim=-1, descending=True).cpu()
        pred_id_contrast = all_options[class_ranks[0]]
        pred_answer_contrast_list.append(pred_id_contrast)

        class_ranks = torch.argsort(logit, dim=-1, descending=True).cpu()
        pred_id = all_options[class_ranks[0]]
        pred_answer_list.append(pred_id)

    nums_correct = sum([a==b for a,b in zip(pred_answer_list, gts_list_1)])
    nums_correct_contrast = sum([a==b for a,b in zip(pred_answer_contrast_list, gts_list_1)])

    print(nums_correct/len(gts_list_1), nums_correct_contrast/len(gts_list_1), c)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='instruct_blip')
    parser.add_argument('--anno_path', type=str, default='SEED-Bench.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--logits_recorded_base_path_1', type=str, default=None)
    parser.add_argument('--logits_recorded_base_path_2', type=str, default=None)
    parser.add_argument('--ori_ans_file_path', type=str, default=None)
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--sample_nums', type=int, default=1)
    parser.add_argument('--top_nums_sim', type=int, default=0)
    parser.add_argument('--start_sim', type=int, default=0)
    parser.add_argument('--top_nums_tie', type=int, default=1)
    parser.add_argument('--start_tie', type=int, default=0)
    parser.add_argument('--method', type=str, default="debias")
    args = parser.parse_args()
    
    args = parser.parse_args()

    print(f'evaluating.. {args.model}')

    if args.method == "debias":
        print(args.logits_recorded_base_path_1.split("/", -1)[-1])
        debias(args, c=10)
    elif args.method == "contrast":
        contrast(args, c=10)
    elif args.method == "all":
        debias(args, c=10)
        contrast(args, c=10)