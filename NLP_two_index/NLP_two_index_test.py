import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import transformers
from transformers import RobertaTokenizer, RobertaConfig
from sklearn.model_selection import StratifiedKFold

import csv

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# device
# device = torch.device('cpu')
device = torch.device('cuda')

# csv path
csv_path = f'Batch_answers - test_data(no_label).csv'  # clean_csv path (load)

name = 'best_jaccard' ################  num or best_jaccard
model_path = f'model/{name}.pth'

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
def read_csv_split_extraction(csv_path):

    # read, select [id, q, r, s]
    df = (pd
           .read_csv(csv_path)
           .iloc[:, 0:4])

    ids = []
    q_texts = []
    r_texts = []

    for row_idx in range(len(df)):
        # id, q, r, s
        id, q, r, s = df.iloc[row_idx]

        # remove --> start " and "
        q, r, = q[1:-1], r[1:-1]
        ids.append(id)
        q_texts.append(q)
        r_texts.append(r)

    return ids, q_texts, r_texts
class CSVDataset_extraction(Dataset):
    def __init__(self,
                 ids,
                 q_texts,
                 r_texts,

                 ):
        self.ids = ids
        self.q_texts = q_texts
        self.r_texts = r_texts


    def __getitem__(self, index):
        id = self.ids[index]
        q_text = {key: torch.tensor(val[index]) for key, val in self.q_texts.items()}
        r_text = {key: torch.tensor(val[index]) for key, val in self.r_texts.items()}

        return {
                'id': id,
                'q_text': q_text,
                'r_text': r_text,
        }

    def __len__(self):
        return len(self.ids)

ids, q_texts, r_texts = read_csv_split_extraction(csv_path)

# tokenize all text
q_texts = tokenizer(q_texts, padding=True, truncation=True, return_tensors='pt').to(device)
r_texts = tokenizer(r_texts, padding=True, truncation=True, return_tensors='pt').to(device)

test_dataset = CSVDataset_extraction(ids, q_texts, r_texts)

test_loader = DataLoader(test_dataset,
                          batch_size=8,
                          shuffle=False,
                          drop_last=False,
                          num_workers=0)

class NLP_model(nn.Module):
    def __init__(self):
        super().__init__()
        # 'deepset/roberta-base-squad2'
        # "roberta-base"

        model_name = "roberta-base"
        model_config = RobertaConfig.from_pretrained(model_name, output_hidden_states=True, )
        self.num_layers = model_config.num_hidden_layers
        self.roberta = transformers.RobertaModel.from_pretrained(model_name, config=model_config)

        self.high_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(model_config.hidden_size * 2, 2)


    def forward(self, x):

        ids, mask = x['input_ids'], x['attention_mask']

        # last_hidden_state ->  bs * sl * 768
        # pooler_output -> bs * 768
        # hidden_state -> (12) * bs * sl * 768

        _, po, hs = self.roberta(ids, attention_mask=mask,).to_tuple()

        stack_hs = torch.stack([hs[i] for i in range(self.num_layers)], dim=0)

        hs_mean = torch.mean(stack_hs, dim=0)
        hs_max = torch.max(stack_hs, dim=0)[0]

        # bs * sl * (768 * 2)
        out = torch.cat((hs_mean, hs_max), dim=-1)

        """"EXTRACTION"""

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        # logits
        logits = torch.mean(torch.stack([self.fc(self.high_dropout(out)) for _ in range(5)], dim=0), dim=0)
        # print(f'{logits.shape=}')

        # start_logits, end_logits
        # (batch_size, num_tokens, 2) -> (batch_size, num_tokens, 1)
        start_logits, end_logits = logits.split(1, dim=-1)

        # -> (batch_size, num_tokens)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
        }

# model
model = NLP_model().to(device)
model.load_state_dict(torch.load(model_path)) ##

ids_ans = []
q_prime_ans = []
r_prime_ans = []

model.eval()
with torch.no_grad():
    for batch_idx, inputs in enumerate(tqdm(test_loader, total=len(test_loader))):

        # inputs
        # 'text', 'prime', 'start_index', 'end_index'
        id = inputs['id'].numpy()
        q_text, r_text = inputs['q_text'], inputs['r_text']

        # model
        q_outputs = model(q_text)
        r_outputs = model(r_text)

        # outputs
        # 'start_logits', 'end_logits'
        q_start_logits, q_end_logits = q_outputs['start_logits'].to(device), q_outputs['end_logits'].to(device)
        r_start_logits, r_end_logits = r_outputs['start_logits'].to(device), r_outputs['end_logits'].to(device)

        # q_pred
        q_start_index = torch.argmax(q_start_logits, dim=-1).to(device)
        q_end_index = torch.argmax(q_end_logits, dim=-1).to(device)

        # r_pred
        r_start_index = torch.argmax(r_start_logits, dim=-1).to(device)
        r_end_index = torch.argmax(r_end_logits, dim=-1).to(device)

        # print(f'{q_start_index=}, {q_end_index=}')
        # print(f'{r_start_index=}, {r_end_index=}')

        q_ids_to_tokens = [tokenizer.convert_ids_to_tokens(tensor, skip_special_tokens=True) for tensor in
                           q_text['input_ids']]
        # print(f'{q_ids_to_tokens=}')
        r_ids_to_tokens = [tokenizer.convert_ids_to_tokens(tensor, skip_special_tokens=True) for tensor in
                           r_text['input_ids']]
        # print(f'{r_ids_to_tokens=}')
        # print(ids_to_tokens)

        q_tokens_to_string = [tokenizer.convert_tokens_to_string(tokens[start: end]) for (tokens, start, end) in
                              zip(q_ids_to_tokens, q_start_index, q_end_index)]
        # print(f'{q_tokens_to_string=}')
        r_tokens_to_string = [tokenizer.convert_tokens_to_string(tokens[start: end]) for (tokens, start, end) in
                          zip(r_ids_to_tokens, r_start_index, r_end_index)]
        # print(f'{r_tokens_to_string=}')

        # print(len(r_tokens_to_string))

        ids_ans.extend(id)
        q_prime_ans.extend(q_tokens_to_string)
        r_prime_ans.extend(r_tokens_to_string)


with open(f'./result.csv', 'w', newline='', encoding="utf-8") as csv_file:

    spamwriter = csv.writer(csv_file)

    # first row
    spamwriter.writerow(('id', 'q', 'r'))

    for index, (id, q_prime, r_prime) in enumerate(zip(ids_ans, q_prime_ans, r_prime_ans)):

        # add " " back
        spamwriter.writerow((id, '"' + q_prime + '"', '"' + r_prime + '"'))
