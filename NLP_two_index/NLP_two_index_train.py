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
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# device
device = torch.device('cuda')

# csv path
csv_path = f'./Batch_answers - train_data (no-blank).csv'  # clean_csv path (load)

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

    # read, select [id, q, r, s, q', r'], remove duplicates
    df = (pd
           .read_csv(csv_path)
           .iloc[:, 0:6]
           .drop_duplicates())

    texts = []
    prime_texts = []
    start_labels = []
    end_labels = []

    for row_idx in range(len(df)):
        # id, q, r, s, q', r'
        id, q, r, s, q_prime, r_prime = df.iloc[row_idx]

        # remove --> start " and "
        q, r, q_prime, r_prime = q[1:-1], r[1:-1], q_prime[1:-1], r_prime[1:-1]

        # s
        # classes = {'AGREE': 1, 'DISAGREE': 0}
        # s = classes[s]

        # tokenize_text `add_prefix_space=True` for consistent tokens
        q_text, r_text = tokenizer.tokenize(q, add_prefix_space=True), tokenizer.tokenize(r, add_prefix_space=True)

        # prime_text
        q_prime_text, r_prime_text = tokenizer.tokenize(q_prime, add_prefix_space=True), tokenizer.tokenize(r_prime, add_prefix_space=True)

        # len
        len_q_prime_text, len_r_prime_text = len(q_prime_text), len(r_prime_text)

        # print(f't_ {q_text}')
        # print(f'p_ {q_prime_text}\n')

        # start end idx
        q_idx_0, q_idx_1, r_idx_0, r_idx_1 = None, None, None, None

        # q_inx
        for ind in (i for i, e in enumerate(q_text) if e == q_prime_text[0]):
            if q_text[ind:ind + len_q_prime_text] == q_prime_text:
                # groundtruth
                q_idx_0 = ind
                q_idx_1 = ind + len_q_prime_text
                # print(q_idx_0,  q_idx_1)
                break
        # r_inx
        for ind in (i for i, e in enumerate(r_text) if e == r_prime_text[0]):
            if r_text[ind:ind + len_r_prime_text] == r_prime_text:
                # groundtruth
                r_idx_0 = ind
                r_idx_1 = ind + len_r_prime_text
                # print(r_idx_0,  r_idx_1)
                break

        # None - not continuous ; (< 512) - sequence larger than `512`
        if q_idx_0 is not None and q_idx_1 is not None and q_idx_0 < 512 and q_idx_1 < 512:
            texts.append(q)
            prime_texts.append(q_prime)
            start_labels.append(q_idx_0)
            end_labels.append(q_idx_1)

        if r_idx_0 is not None and r_idx_1 is not None and r_idx_0 < 512 and r_idx_1 < 512:
            texts.append(r)
            prime_texts.append(r_prime)
            start_labels.append(r_idx_0)
            end_labels.append(r_idx_1)

        # texts.append(q)
        # prime_texts.append(q_prime)
        # start_labels.append(q_idx_0)
        # end_labels.append(q_idx_1)
        #
        # texts.append(r)
        # prime_texts.append(r_prime)
        # start_labels.append(r_idx_0)
        # end_labels.append(r_idx_1)

    return texts, prime_texts, start_labels, end_labels

class CSVDataset_extraction(Dataset):
    def __init__(self,
                 texts,
                 prime_texts,
                 start_labels,
                 end_labels,
                 ):

        self.texts = texts

        self.prime_texts = prime_texts
        self.start_labels = start_labels
        self.end_labels = end_labels


    def __getitem__(self, index):

        text = {key: torch.tensor(val[index]) for key, val in self.texts.items()}
        prime = self.prime_texts[index]

        start_index = self.start_labels[index]
        end_index = self.end_labels[index]

        return {
                # extraction
                'text': text,
                'prime': prime,
                'start_index': start_index,
                'end_index': end_index,
        }

    def __len__(self):
        return len(self.start_labels)

texts, prime_texts, start_labels, end_labels = read_csv_split_extraction(csv_path)

# print(f' start {list(set(start_labels))}')
# print(f' end {list(set(end_labels))}')

# tokenize all text
texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)

csv_dataset = CSVDataset_extraction(texts, prime_texts, start_labels, end_labels)

train_dataset, val_dataset = train_test_split(csv_dataset, test_size=0.1)

skf = StratifiedKFold(n_splits=5)

train_loader = DataLoader(train_dataset,
                          batch_size=16,
                          shuffle=True,
                          drop_last=True,
                          num_workers=0)

val_dataset = DataLoader(val_dataset,
                         batch_size=16,
                         shuffle=False,
                         num_workers=0,)

# input : ids, mask, output : start_logits, ens_logits
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

epochs = 100

# optimizer
from timm.optim.optim_factory import create_optimizer_v2

optimizer = create_optimizer_v2(model,
                                opt='adam',
                                lr=1e-4,
                                weight_decay=0)

# scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

scheduler = CosineLRScheduler(optimizer,
                              t_initial=100,
                              lr_min=1e-6,
                              warmup_t=5,
                              warmup_lr_init=1e-4,
                              cycle_limit=1)
# loss
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# save
best_jaccard_value = 0
best_jaccard_epoch = 0





for epoch in range(epochs):
    # model
    train_losses = AverageMeter()
    model.train()
    for batch_idx, inputs in enumerate(tqdm(train_loader, total=len(train_loader))):
        # print(f'{batch_idx=}')
        # if batch_idx < 1692:
        #     continue

        num_batches_per_epoch = len(train_loader)
        num_updates = epoch * num_batches_per_epoch

        # inputs
        # 'text', 'prime', 'start_index', 'end_index'
        text, start_index, end_index = inputs['text'], \
                                       inputs['start_index'].to(device), \
                                       inputs['end_index'].to(device)

        # does not need to transform into tensor
        prime = inputs['prime']

        # print(prime)

        # model
        outputs = model(text)

        # outputs
        # 'start_logits', 'end_logits'
        start_logits, end_logits = outputs['start_logits'].to(device), outputs['end_logits'].to(device)

        # extraction
        loss = loss_fn(start_logits, start_index) + loss_fn(end_logits, end_index)

        # losses
        train_losses.update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step_update(num_updates=num_updates, metric=train_losses.avg)

    print(f'[TRAIN] epoch{epoch + 1} total_loss:{train_losses.avg}')

    pred_jaccard = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, inputs in enumerate(tqdm(val_dataset, total=len(val_dataset))):

            # inputs
            # 'text', 'prime', 'start_index', 'end_index'
            text, start_index, end_index = inputs['text'], \
                                           inputs['start_index'].to(device), \
                                           inputs['end_index'].to(device)

            # does not need to transform into tensor
            prime = inputs['prime']

            # model
            outputs = model(text)

            # outputs
            # 'start_logits', 'end_logits'
            start_logits, end_logits = outputs['start_logits'].to(device), outputs['end_logits'].to(device)

            # pred
            pred_start_index = torch.argmax(start_logits, dim=-1).to(device)
            pred_end_index = torch.argmax(end_logits, dim=-1).to(device)

            ids_to_tokens = [tokenizer.convert_ids_to_tokens(tensor, skip_special_tokens=True) for tensor in
                               text['input_ids']]
            # print(ids_to_tokens)

            for (start, end) in zip(pred_start_index, pred_end_index):
                # print(start, end)
                tokens_to_string = [tokenizer.convert_tokens_to_string(tokens[start: end]) for tokens in
                                      ids_to_tokens]


            for pred, true in zip(tokens_to_string, prime):
                # print(jaccard(pred, true))
                pred_jaccard.update(jaccard(pred, true))

        if pred_jaccard.avg > best_jaccard_value:
            best_jaccard_value = pred_jaccard.avg
            best_jaccard_epoch = epoch + 1
            torch.save(model.state_dict(), f'model/best_jaccard.pth')

        torch.save(model.state_dict(), f'model/{epoch + 1}.pth')

        print(f'[VAL] epoch : {epoch + 1} jaccard_mean : {pred_jaccard.avg}\n')
        print(f'[VAL] best epoch : {best_jaccard_epoch} best_jaccard_mean : {best_jaccard_value}\n')