import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
from nltk import word_tokenize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from rouge_score import rouge_scorer
from timm.optim.optim_factory import create_optimizer_v2
from timm.scheduler.cosine_lr import CosineLRScheduler

MAX_LENGTH = 512
THRESHOLD = 0.3
BATCH_SIZE = 16
# PATH = r'D:\AICUP_nlp\models\microsoftdeberta-v3-base\NLP_label\model\4.pth'  # epoch_num
PATH = f'./model/4.pth'  # epoch_num
MODEL = 'microsoft/deberta-base'


tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=MAX_LENGTH, add_prefix_space=True)

# csv path
# csv_path = f'Batch_answers - test_data(no_label).csv'
csv_path = f'xlm-roberta-large_index_3.csv'

# output list
ids_ans = []

q_prime_index_ans = []
r_prime_index_ans = []

q_prime_label_ans = []
r_prime_label_ans = []

def read_csv_split_extraction(csv_path):
    # read, select [id, q, r, s]
    df = (pd
          .read_csv(csv_path)
          .iloc[:, 0:4])

    ids = []
    q_texts = []
    r_texts = []
    q_texts_wt = []  # word_tokenize
    r_texts_wt = []  # word_tokenize
    for row_idx in range(len(df)):
        # id, q, r, s
        try:
            id, q, r, s = df.iloc[row_idx]
        except ValueError: # no s
            id, q, r = df.iloc[row_idx]
        # remove --> start " and "
        q, r, = q[1:-1], r[1:-1]
        # wt
        q_wt = word_tokenize(q)
        r_wt = word_tokenize(r)
        ids.append(id)
        q_texts.append(q)
        r_texts.append(r)
        q_texts_wt.append(q_wt)
        r_texts_wt.append(r_wt)
    return ids, q_texts, r_texts, q_texts_wt, r_texts_wt,
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
class NLP_model_label(nn.Module):
    def __init__(self):
        super().__init__()

        model_config = AutoConfig.from_pretrained(MODEL, output_hidden_states=True)
        self.num_layers = model_config.num_hidden_layers
        self.model = AutoModel.from_pretrained(MODEL, config=model_config)

        self.high_dropout = nn.Dropout(0.5)
        self.label_fc = nn.Linear(model_config.hidden_size * 2, 1)

    def forward(self, x):
        ids, mask = x['input_ids'], x['attention_mask']

        # last_hidden_state ->  bs * sl * hs
        # pooler_output -> bs * hs
        # hidden_state -> (12) * bs * sl * hs

        # _, po, hs = self.model(ids, attention_mask=mask).to_tuple() # ONLY FOR ROBERTA
        lhs, hs = self.model(ids, attention_mask=mask).to_tuple()

        stack_hs = torch.stack([hs[i] for i in range(self.num_layers)], dim=0)

        hs_mean = torch.mean(stack_hs, dim=0)
        hs_max = torch.max(stack_hs, dim=0)[0]

        # bs * sl * (hs * 2)
        out = torch.cat((hs_mean, hs_max), dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        label_logits = torch.mean(torch.stack([self.label_fc(self.high_dropout(out)) for _ in range(5)], dim=0), dim=0)
        label_logits = label_logits.squeeze(-1)

        return {'label_logits': label_logits, 'output_mask': mask}


if __name__ == '__main__':

    # device
    device = torch.device('cuda')
    # device = torch.device('cpu')

    # dataset
    ids, q_texts, r_texts, q_texts_wt, r_texts_wt = read_csv_split_extraction(csv_path)

    # wt
    q_texts = tokenizer(q_texts_wt, is_split_into_words=True, padding='max_length', truncation=True, return_tensors='pt').to(device)
    r_texts = tokenizer(r_texts_wt, is_split_into_words=True, padding='max_length', truncation=True, return_tensors='pt').to(device)
    # not wt
    # q_texts = tokenizer(q_texts, padding='max_length', truncation=True, return_tensors='pt').to(device)
    # r_texts = tokenizer(r_texts, padding='max_length', truncation=True, return_tensors='pt').to(device)


    test_dataset = CSVDataset_extraction(ids, q_texts, r_texts)

    # dataloader
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             drop_last=False,
                             num_workers=0)

    # model
    model = NLP_model_label().to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()

    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(test_loader, total=len(test_loader))):
            # inputs : 'id', 'text'
            id = inputs['id'].numpy()
            q_text, r_text = inputs['q_text'], inputs['r_text']

            # model
            q_outputs = model(q_text)
            r_outputs = model(r_text)

            # outputs
            # q_start_logits = q_outputs['start_logits'].to(device)
            # q_end_logits = q_outputs['end_logits'].to(device)
            q_label_logits = q_outputs['label_logits'].to(device)

            # index
            # q_pred_start_indices = torch.argmax(q_start_logits, dim=-1).cpu()
            # q_pred_end_indices = torch.argmax(q_end_logits, dim=-1).cpu()

            # label
            q_label_logits = F.logsigmoid(q_label_logits).exp()  # sigmoid
            q_label_logits = (q_label_logits > THRESHOLD).float()  # binarize # > threshold -> 1 ; <= threshold ->  0
            q_pred_label_indices = [np.where(logit == 1) for logit in q_label_logits.cpu()]

            # outputs
            # r_start_logits = r_outputs['start_logits'].to(device)
            # r_end_logits = r_outputs['end_logits'].to(device)
            r_label_logits = r_outputs['label_logits'].to(device)

            # index
            # r_pred_start_indices = torch.argmax(r_start_logits, dim=-1).cpu()
            # r_pred_end_indices = torch.argmax(r_end_logits, dim=-1).cpu()

            # label
            r_label_logits = F.logsigmoid(r_label_logits).exp()  # sigmoid
            r_label_logits = (r_label_logits > THRESHOLD).float()  # binarize # > threshold -> 1 ; <= threshold ->  0
            r_pred_label_indices = [np.where(logit == 1) for logit in r_label_logits.cpu()]

            # [INDEX]
            # q_pred_index_tokens_to_string = \
            #     [tokenizer.decode(tensors[start_index: end_index], skip_special_tokens=True) for
            #      (tensors, start_index, end_index) in
            #      zip(q_text['input_ids'],  q_pred_start_indices,  q_pred_end_indices)]
            #
            # r_pred_index_tokens_to_string = \
            #     [tokenizer.decode(tensors[start_index: end_index], skip_special_tokens=True) for
            #      (tensors, start_index, end_index) in
            #      zip(r_text['input_ids'], r_pred_start_indices, r_pred_end_indices)]

            # [LABEL]
            q_pred_label_tokens_to_string = \
                [tokenizer.decode(tensors[index], skip_special_tokens=True) for
                 (tensors, index) in
                 zip(q_text['input_ids'], q_pred_label_indices)]
            r_pred_label_tokens_to_string = \
                [tokenizer.decode(tensors[index], skip_special_tokens=True) for
                 (tensors, index) in
                 zip(r_text['input_ids'], r_pred_label_indices)]

            ids_ans.extend(id)
            # q_prime_index_ans.extend(q_pred_index_tokens_to_string)
            # r_prime_index_ans.extend(r_pred_index_tokens_to_string)

            q_prime_label_ans.extend(q_pred_label_tokens_to_string)
            r_prime_label_ans.extend(r_pred_label_tokens_to_string)

    with open(f'./index_N_label_result_{Path(PATH).stem}.csv', 'w', newline='', encoding="utf-8") as csv_file:

        spamwriter = csv.writer(csv_file)

        # first row
        spamwriter.writerow(('id', 'q', 'r'))

        for index, (id, q_prime, r_prime) in enumerate(zip(ids_ans, q_prime_label_ans, r_prime_label_ans)):
            # add " " back
            spamwriter.writerow((id, '"' + q_prime + '"', '"' + r_prime + '"'))