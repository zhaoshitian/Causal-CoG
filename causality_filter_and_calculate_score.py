import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import json
import argparse
from PIL import Image

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

# root directory of evaluation dimension 1-9
cc3m_dir = "/data/stzhao_data/data_vl/SEEDBench/SEED-Bench-image"

clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
tokenizer_llava = AutoTokenizer.from_pretrained("/home/ubuntu/stzhao/LLaVA/finetuned_llm_weight/LLaVA-Lightning-7B-vicuna-v1-1")

seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

    
def kl_divergence(p, q):
    return torch.sum(p * torch.log(p / q))

def js_divergence(p, q, softmax=True):
    p = torch.tensor(p).float()
    q = torch.tensor(q).float()

    p = 20*p / p.norm()
    q = 20*q / q.norm()
    if softmax == True:
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
    # 计算平均分布
    m = 0.5 * (p + q)
    
    # 计算两个KL散度
    kl_p_m = kl_divergence(p, m)
    kl_q_m = kl_divergence(q, m)
    
    # 计算JS散度
    jsd = 0.5 * (kl_p_m + kl_q_m)
    
    return jsd

def get_batch(all_data, batchsize=50):
    batch_list = []
    nums_data = len(all_data)
    nums_batch = int(nums_data / batchsize)
    for b in range(nums_batch):
        if b < nums_batch - 1:
            batch_data = all_data[b*batchsize:(b+1)*batchsize]
        else:
            batch_data = all_data[b*batchsize:]
        batch_list.append(batch_data)
    return batch_list


def causality_filter_and_calculate_score(args):

    logits_con_img_list = [[] for _ in range(args.sample_nums)] 
    logits_img_list = [[] for _ in range(args.sample_nums)] 
    logits_ques_list = [[] for _ in range(args.sample_nums)] 
    descriptions_list = [[] for _ in range(args.sample_nums)] 
    preds_list = [[] for _ in range(args.sample_nums)] 
    gts_list = [[] for _ in range(args.sample_nums)] 
    question_ids_list = [[] for _ in range(args.sample_nums)] 

    effect_con_list = [[] for _ in range(args.sample_nums)] 
    effect_img_list = [[] for _ in range(args.sample_nums)] 
    description_features_list = [[] for _ in range(args.sample_nums)] 
    similarity_list = [[] for _ in range(args.sample_nums)] 

    seed_list = [s for s in range(args.sample_nums)]

    for i, seed in enumerate(tqdm(seed_list)):

        seed += 1
        # logits_recorded = load_from_disk(f"{args.logits_recorded_base_path}/{seed}")
        logits_recorded = load_from_disk(f"{args.logits_recorded_base_path}")
        logits_con_img_list[i]=logits_recorded['logits_con_img']
        logits_img_list[i]=logits_recorded['logits_img']
        logits_ques_list[i]=logits_recorded['logits_ques']
        descriptions_list[i]=logits_recorded['descriptions']
        preds_list[i]=logits_recorded['preds']
        gts_list[i]=logits_recorded['gts']
        question_ids_list[i]=logits_recorded['indexs']

        effect_con_list[i]=[js_divergence(l0, l1, softmax=True) for l0, l1 in zip(logits_recorded['logits_con_img'], logits_recorded['logits_img'])]
        effect_img_list[i]=[js_divergence(l0, l1, softmax=True) for l0, l1 in zip(logits_recorded['logits_img'], logits_recorded['logits_ques'])] 

        description_input_ids = tokenizer(logits_recorded['descriptions'],  truncation=True, padding=True, return_tensors="pt")['input_ids'].cuda()
        batch_list = torch.chunk(description_input_ids, chunks=30, dim=0)
        with torch.no_grad():
            clip.eval()
            description_feature_list = [clip.get_text_features(batch) for batch in batch_list] 
        description_features_list[i] = torch.cat(description_feature_list, dim=0)

        if i == 0:
            images_features_list = []
            img_path_list = [os.path.join(cc3m_dir, data_id) for data_id in logits_recorded['data_id']]
            img_path_batch_list = get_batch(img_path_list, batchsize=30)
            for img_path_batch in tqdm(img_path_batch_list):
                pixel_values_list = [processor(images=Image.open(open(img_path, "rb")), return_tensors="pt")['pixel_values'][0].cuda() for img_path in img_path_batch]
                pixel_values_batch = torch.stack(pixel_values_list, dim=0)
                with torch.no_grad():
                    clip.eval()
                    images_features_list.append(clip.get_image_features(pixel_values_batch))
            images_features_list = torch.cat(images_features_list, dim=0)
        
        similarity_list[i] = F.cosine_similarity(description_features_list[i], images_features_list, dim=1)

        

    logits_con_img_list_sample_first = [] 
    logits_img_list_sample_first = []
    logits_ques_list_sample_first = []
    descriptions_list_sample_first = []
    preds_list_sample_first = []
    gts_list_sample_first = []
    question_ids_list_sample_first = []

    effect_con_list_sample_first = []
    effect_img_list_sample_first = []
    similarity_list_sample_first = []
    for j in range(len(logits_con_img_list[0])):
        logits_con_img_list_one_sample = [s[j] for s in logits_con_img_list] 
        logits_img_list_one_sample = [s[j] for s in logits_img_list] 
        logits_ques_list_one_sample = [s[j] for s in logits_ques_list] 
        descriptions_list_one_sample = [s[j] for s in descriptions_list] 
        preds_list_one_sample = [s[j] for s in preds_list] 
        gts_list_one_sample = [s[j] for s in gts_list] 
        question_ids_list_one_sample = [s[j] for s in question_ids_list] 
        effect_con_list_one_sample = [s[j] for s in effect_con_list]
        effect_img_list_one_sample = [s[j] for s in effect_img_list]
        similarity_list_one_sample = [s[j] for s in similarity_list]

        logits_con_img_list_sample_first.append(logits_con_img_list_one_sample)
        logits_img_list_sample_first.append(logits_img_list_one_sample)
        logits_ques_list_sample_first.append(logits_ques_list_one_sample)
        descriptions_list_sample_first.append(descriptions_list_one_sample)
        preds_list_sample_first.append(preds_list_one_sample)
        gts_list_sample_first.append(gts_list_one_sample)
        question_ids_list_sample_first.append(question_ids_list_one_sample)
        effect_con_list_sample_first.append(effect_con_list_one_sample)
        effect_img_list_sample_first.append(effect_img_list_one_sample)
        similarity_list_sample_first.append(similarity_list_one_sample)

############################################# debias
    # effect_con_list_sample_first0 = effect_con_list_sample_first
    # effect_con_list_sample_first = [[float(1) for logits, p in zip(logits_list, p_list)] for logits_list, p_list in zip(logits_con_img_list_sample_first, preds_list_sample_first)]

    similarity_jsd_list = [[[s, j] for s, j in zip(sim, jsd)] for sim, jsd in zip(similarity_list_sample_first, effect_con_list_sample_first)]
    similarity_jsd_list_with_index = [[s+[index] for index, s in zip(range(len(score)), score)] for score in similarity_jsd_list]
    def takefirst(elem):
        return elem[0]
    def takesecond(elem):
        return elem[1]
    selected_candidate_index_jsd = [[ss[2] for ss in sorted(s, key=takesecond, reverse=True)[args.start_tie:args.top_nums_tie]] for s in similarity_jsd_list_with_index]
    selected_candidate_index_sim = [[ss[2] for ss in sorted(s, key=takefirst, reverse=True)[args.start_sim:args.top_nums_sim]] for s in similarity_jsd_list_with_index]


    w = torch.ones(len(selected_candidate_index_jsd), args.sample_nums)

    w1 = torch.zeros(len(selected_candidate_index_jsd), args.sample_nums)
    for index, sample_weight in zip(selected_candidate_index_jsd, w1):
        sample_weight[index] = 1
    w2 = torch.zeros(len(selected_candidate_index_sim), args.sample_nums)
    for index, sample_weight in zip(selected_candidate_index_sim, w2):
        sample_weight[index] = 1

    final_w = []
    count_same = 0
    for j in range(len(w)):
        sim = torch.tensor(similarity_list_sample_first[j])
        jsd = torch.tensor(effect_con_list_sample_first[j])

        sim = sim / sim.norm()
        jsd = jsd / jsd.norm()

        mean_sim = sim.mean()
        mean_jsd = jsd.mean()

        # jsd = jsd - (mean_jsd - mean_sim)

        var_sim = sim.var()
        var_jsd = jsd.var()

        # sim = (sim - mean_sim) 
        # jsd = (jsd - mean_jsd) 
        # sim = F.relu(sim)
        # jsd = F.relu(jsd)
        sim = sim * w2[j]
        jsd = jsd * w1[j]

        final_w.append(sim + jsd)
        # print(f"sim: {sim},    jsd: {jsd}")

    ans_cog_list = []
    for i in range(len(preds_list_sample_first)):
        count_a = 0
        count_b = 0
        count_c = 0
        count_d = 0
        preds_list_one_sample = preds_list_sample_first[i]
        for j, pred in enumerate(preds_list_one_sample):
            if pred == "A":
                count_a += final_w[i][j]
            elif pred == "B":
                count_b += final_w[i][j]
            elif pred == "C":
                count_c += final_w[i][j]
            elif pred == "D":
                count_d += final_w[i][j]
        ans = ["A", "B", "C", "D"][[count_a, count_b, count_c, count_d].index(max([count_a, count_b, count_c, count_d]))]
        ans_cog_list.append(ans)

    ans_ori_list = []
    ori_ans_file = open(args.ori_ans_file_path, "r")
    for line in ori_ans_file.readlines():
        ans = line.split("\n", -1)[0]
        ans = json.loads(ans)
        ans = ans['prediction']
        ans_ori_list.append(ans)


    final_answer_list = []
    for j in range(len(logits_con_img_list[0])):

        if sum(effect_con_list_sample_first[j])/len(effect_con_list_sample_first[j]) < sum(effect_img_list_sample_first[j])/len(effect_img_list_sample_first[j]):
            ans = ans_ori_list[j]
            final_answer_list.append(ans)
        else:
            ans = ans_cog_list[j]
            final_answer_list.append(ans)

    print(len(final_answer_list))

    evaluator = Evaluator(pred_list=final_answer_list, gt_list=gts_list_sample_first, question_id_list=question_ids_list_sample_first)
    evaluator.evaluate(ds="mme")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='instruct_blip')
    parser.add_argument('--anno_path', type=str, default='SEED-Bench.json')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--logits_recorded_base_path', type=str, default=None)
    parser.add_argument('--ori_ans_file_path', type=str, default=None)
    parser.add_argument('--task', type=str, default='all')
    parser.add_argument('--sample_nums', type=int, default=1)
    parser.add_argument('--top_nums_sim', type=int, default=0)
    parser.add_argument('--start_sim', type=int, default=0)
    parser.add_argument('--top_nums_tie', type=int, default=1)
    parser.add_argument('--start_tie', type=int, default=0)
    args = parser.parse_args()
    
    args = parser.parse_args()

    print(f'evaluating.. {args.model}')
    causality_filter_and_calculate_score(args)