import json

options_list = ['A', 'B', 'C', 'D', 'E', 'F']

# MME

task_list = ['existence', 'color', 'count', 'position', 'OCR', 'code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation']
id = 0
sample_list = []
for task_name in task_list:
    file_path = f'/home/ubuntu/stzhao/MME/eval_tool/Your_Results/{task_name}.txt'
    file = open(file_path, "r").readlines()
    for line in file:
        image_path = f"/data/stzhao_data/data_vl/mme/MME_Benchmark_release_version/{task_name}/" + line.split('\t', -1)[0]
        question = line.split("\t", -1)[1]
        gt_answers = line.split('\t', -1)[-1]
        question_id = str(id) + "_" + task_name
        choices_list = ['Yes', 'No']
        gt_answers = gt_answers.strip()
        gt_answers = options_list[choices_list.index(gt_answers)]
        id += 1

        sample = {}
        sample['question_id'] = question_id
        sample['image_path'] = image_path
        sample['question'] = question
        sample['choices'] = choices_list
        sample['gt_answers'] = gt_answers

        sample_list.append(sample)

with open("/home/ubuntu/stzhao/Causal-CoG/test_data/mme.json", "w") as f:
    json.dump(sample_list, f, indent=4)



# SEEDBench
'''

file_path = '/data/stzhao_data/data_vl/SEEDBench/SEED-Bench.json'
file = json.load(open(file_path, "r"))
question_list = file['questions']
question_type_mapping_list = file['question_type']
sample_list = []
for item in question_list:
    question_type = list(question_type_mapping_list.keys())[item['question_type_id']-1]
    question_id = item['question_id']+"_"+question_type
    question = item['question']
    image_path = "/data/stzhao_data/data_vl/SEEDBench/SEED-Bench-image/" + item['data_id']
    choices_list = [item['choice_a'], item['choice_b'], item['choice_c'], item['choice_d']]
    gt_answers = item['answer']

    sample = {}
    sample['question_id'] = question_id
    sample['image_path'] = image_path
    sample['question'] = question
    sample['choices'] = choices_list
    sample['gt_answers'] = gt_answers

    sample_list.append(sample)

with open("/home/ubuntu/stzhao/Causal-CoG/test_data/seedbench.json", "w") as f:
    json.dump(sample_list, f, indent=4)

'''


#POPE
import jsonlines
subclass_name_list = ['popular', 'random', 'adversarial']
for subclass_name in subclass_name_list:
    sample_list = []
    file_path = f"/home/ubuntu/stzhao/POPE/output/coco/coco_pope_{subclass_name}.jsonl"
    with jsonlines.open(file_path) as reader:
        for line in reader:
            question_id = str(line['question_id'])+"_"+subclass_name
            question = line['text']
            image_path = f"/data/stzhao_data/data_vl/COCO_Caption/images/val2014/" + line['image']
            choices_list = ['yes', 'no']
            gt_answers = line['label']
            gt_answers = options_list[choices_list.index(gt_answers)]

            sample = {}
            sample['question_id'] = question_id
            sample['image_path'] = image_path
            sample['question'] = question
            sample['choices'] = ['Yes', 'No']
            sample['gt_answers'] = gt_answers

            sample_list.append(sample)

    with open(f"/home/ubuntu/stzhao/Causal-CoG/test_data/pope_{subclass_name}.json", "w") as f:
            json.dump(sample_list, f, indent=4)



#VSR
import jsonlines
file_path = "/home/ubuntu/stzhao/visual-spatial-reasoning/data/splits/zeroshot/test.jsonl"
id = 0
sample_list = []
with jsonlines.open(file_path) as reader:
    for line in reader:
        question = "Based on the image, is this statement true or false?\n"+ line['caption']
        question_id = id
        image_path = "/data/stzhao_data/data_vl/coco2017/images/" + line['image']
        choices_list = ['Answer: False', 'Answer: True']
        gt_answers = choices_list[line['label']]
        gt_answers = options_list[choices_list.index(gt_answers)]
        id += 1

        sample = {}
        sample['question_id'] = str(question_id)
        sample['image_path'] = image_path
        sample['question'] = question
        sample['choices'] = choices_list
        sample['gt_answers'] = gt_answers
        sample_list.append(sample)

with open("/home/ubuntu/stzhao/Causal-CoG/test_data/vsr.json", "w") as f:
        json.dump(sample_list, f, indent=4)



#MMBench
'''
import base64
import io
import random
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

class MMBenchDataset(Dataset):
    def __init__(self,
                 data_file,
                 sys_prompt='There are several options:'):
        self.df = pd.read_csv(data_file, sep='\t')
        self.sys_prompt = sys_prompt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index = self.df.iloc[idx]['index']
        image = self.df.iloc[idx]['image']
        image = decode_base64_to_image(image)
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answer'] if 'answer' in self.df.iloc[0].keys() else None
        catetory = self.df.iloc[idx]['category']
        l2_catetory = self.df.iloc[idx]['l2-category']

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: self.load_from_df(idx, cand)
            for cand in option_candidate
            if self.load_from_df(idx, cand) is not None
        }
        options_prompt = f'{self.sys_prompt}\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        hint = self.load_from_df(idx, 'hint')
        data = {
            'img': image,
            'question': question,
            'answer': answer,
            'options': options_prompt,
            'category': catetory,
            'l2-category': l2_catetory,
            'options_dict': options,
            'index': index,
            'context': hint,
        }
        return data
    def load_from_df(self, idx, key):
        if key in self.df.iloc[idx] and not pd.isna(self.df.iloc[idx][key]):
            return self.df.iloc[idx][key]
        else:
            return None


file_path_list = [("dev", "mmbench_dev_20230712.tsv"), ("dev_en", "mmbench_dev_en_20231003.tsv"), ("test", "mmbench_test_20230712.tsv")]
for split, file in file_path_list:
    file_path = "/data/stzhao_data/data_vl/MMBench/" + file
    mmbench = MMBenchDataset(file_path)
    sample_list = []
    for index in range(len(mmbench)):
        data = mmbench.__getitem__(index)
        category = data['category']
        l2_category = data['l2-category']
        question_id = str(data['index']) + "_" + category + "_" + l2_category
        question = data['question']
        choices_list = list(data['options_dict'].values())
        hint = data['context']
        gt_answers = data['answer']
        image = data['img']
        if not os.path.exists(f"/data/stzhao_data/data_vl/MMBench/images/{split}"):
            os.makedirs(f"/data/stzhao_data/data_vl/MMBench/images/{split}")
        image_path = f"/data/stzhao_data/data_vl/MMBench/images/{split}/{data['index']}_{split}.jpg"
        image.save(image_path)

        sample = {}
        sample['question_id'] = question_id
        sample['image_path'] = image_path
        sample['question'] = question
        sample['hint'] = hint
        sample['choices'] = choices_list
        sample['gt_answers'] = gt_answers

        sample_list.append(sample)

    with open(f"/home/ubuntu/stzhao/Causal-CoG/test_data/mmbench_{split}.json", "w") as f:
        json.dump(sample_list, f, indent=4)

'''

#VQA2 reformeval

import json
import base64
import io
import random
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

file_path = "/home/ubuntu/stzhao/VQAv2/vqa_val_10p.json"
file = json.load(open(file_path, "r"))
data = file['data']
sample_list = []
for i, item in enumerate(data):
    image = decode_base64_to_image(item['image_id'])
    if not os.path.exists(f"/data/stzhao_data/data_vl/VQAv2_reformeval/images"):
        os.makedirs(f"/data/stzhao_data/data_vl/VQAv2_reformeval/images")
    image_path = f"/data/stzhao_data/data_vl/VQAv2_reformeval/images/{item['question_id']}.jpg"
    image.save(image_path)
    question_id = str(item['question_id'])
    question = item['question']
    gt_answer = item['answer']
    choices_list = item['answer_options']
    gt_answer = options_list[choices_list.index(gt_answer)]

    sample = {}
    sample['question_id'] = question_id
    sample['image_path'] = image_path
    sample['question'] = question
    sample['choices'] = choices_list
    sample['gt_answers'] = gt_answer
    sample_list.append(sample)

with open(f"/home/ubuntu/stzhao/Causal-CoG/test_data/vqav2_reformeval.json", "w") as f:
    json.dump(sample_list, f, indent=4)



#GQA reformeval

import json
import base64
import io
import random
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

file_path = "/home/ubuntu/stzhao/GQA/gqa_testdev_10p.json"
file = json.load(open(file_path, "r"))
data = file['data']
sample_list = []
for i, item in enumerate(data):
    image = decode_base64_to_image(item['image_id'])
    if not os.path.exists(f"/data/stzhao_data/data_vl/GQA_reformeval/images"):
        os.makedirs(f"/data/stzhao_data/data_vl/GQA_reformeval/images")
    image_path = f"/data/stzhao_data/data_vl/GQA_reformeval/images/{item['question_id']}.jpg"
    image.save(image_path)
    question_id = str(item['question_id'])
    question = item['question']
    gt_answer = item['answer']
    choices_list = item['answer_options']
    gt_answer = options_list[choices_list.index(gt_answer)]

    sample = {}
    sample['question_id'] = question_id
    sample['image_path'] = image_path
    sample['question'] = question
    sample['choices'] = choices_list
    sample['gt_answers'] = gt_answer
    sample_list.append(sample)

with open(f"/home/ubuntu/stzhao/Causal-CoG/test_data/gqa_reformeval.json", "w") as f:
    json.dump(sample_list, f, indent=4)


#Vizwiz reformeval
import json
import base64
import io
import random
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

file_path = "/home/ubuntu/stzhao/Vizwiz/vizwiz_val_10p.json"
file = json.load(open(file_path, "r"))
data = file['data']
sample_list = []
for i, item in enumerate(data):
    image = decode_base64_to_image(item['image_id'])
    if not os.path.exists(f"/data/stzhao_data/data_vl/Vizwiz_reformeval/images"):
        os.makedirs(f"/data/stzhao_data/data_vl/Vizwiz_reformeval/images")
    image_path = f"/data/stzhao_data/data_vl/Vizwiz_reformeval/images/{item['question_id']}.jpg"
    image.save(image_path)
    question_id = str(item['question_id'])
    question = item['question']
    gt_answer = item['answer']
    choices_list = item['answer_options']
    gt_answer = options_list[choices_list.index(gt_answer)]

    sample = {}
    sample['question_id'] = question_id
    sample['image_path'] = image_path
    sample['question'] = question
    sample['choices'] = choices_list
    sample['gt_answers'] = gt_answer
    sample_list.append(sample)

with open(f"/home/ubuntu/stzhao/Causal-CoG/test_data/vizwiz_reformeval.json", "w") as f:
    json.dump(sample_list, f, indent=4)

#Winoground reformeval
import json
import base64
import io
import random
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image

file_path = "/home/ubuntu/stzhao/Winoground/winoground_all_10p.json"
file = json.load(open(file_path, "r"))
data = file['data']
sample_list = []
for i, item in enumerate(data):
    image = decode_base64_to_image(item['image_id'])
    if not os.path.exists(f"/data/stzhao_data/data_vl/Winoground_reformeval/images"):
        os.makedirs(f"/data/stzhao_data/data_vl/Winoground_reformeval/images")
    image_path = f"/data/stzhao_data/data_vl/Winoground_reformeval/images/{item['question_id']}.jpg"
    image.save(image_path)
    question_id = str(item['question_id'])
    question = item['question']
    gt_answer = item['answer']
    choices_list = item['answer_options']
    gt_answer = options_list[choices_list.index(gt_answer)]

    sample = {}
    sample['question_id'] = question_id
    sample['image_path'] = image_path
    sample['question'] = question
    sample['choices'] = choices_list
    sample['gt_answers'] = gt_answer
    sample_list.append(sample)

with open(f"/home/ubuntu/stzhao/Causal-CoG/test_data/winoground_reformeval.json", "w") as f:
    json.dump(sample_list, f, indent=4)

#OKVQA reformeval
import json
import base64
import io
import random
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

file_path = "/home/ubuntu/stzhao/OKVQA/okvqa_val_10p.json"
file = json.load(open(file_path, "r"))
data = file['data']
sample_list = []
for i, item in enumerate(data):
    image = decode_base64_to_image(item['image_id'])
    if not os.path.exists(f"/data/stzhao_data/data_vl/OKVQA_reformeval/images"):
        os.makedirs(f"/data/stzhao_data/data_vl/OKVQA_reformeval/images")
    image_path = f"/data/stzhao_data/data_vl/OKVQA_reformeval/images/{item['question_id']}.jpg"
    image.save(image_path)
    question_id = str(item['question_id'])
    question = item['question']
    gt_answer = item['answer']
    choices_list = item['answer_options']
    gt_answer = options_list[choices_list.index(gt_answer)]

    sample = {}
    sample['question_id'] = question_id
    sample['image_path'] = image_path
    sample['question'] = question
    sample['choices'] = choices_list
    sample['gt_answers'] = gt_answer
    sample_list.append(sample)

with open(f"/home/ubuntu/stzhao/Causal-CoG/test_data/okvqa_reformeval.json", "w") as f:
    json.dump(sample_list, f, indent=4)

    
  


