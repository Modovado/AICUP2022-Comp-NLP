import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from nltk import word_tokenize
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from rouge_score import rouge_scorer
from timm.optim.optim_factory import create_optimizer_v2
from timm.scheduler.cosine_lr import CosineLRScheduler
from pprint import pprint
MAX_LENGTH = 512
EPOCHS = 100
BATCH_SIZE = 8
THRESHOLD = 0.5
MODEL = 'xlm-roberta-large'


# tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=MAX_LENGTH)
tokenizer = AutoTokenizer.from_pretrained(MODEL, add_prefix_space=True)

# dir & csv path
csv_path = f'./Batch_answers - train_data (no-blank).csv'
save_dir = f'./model/'

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
class EpochMonitor:
    """Stores the best (performance) loss / metric value and epoch"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"epoch": 0, "value": 0.})

    def update(self, metric_name: str, epoch: int, value: float, format: str, n=1):

        assert format in ['loss', 'metric']
        metric = self.metrics[metric_name]

        # if first epoch
        # or if metric value is greater than previous
        # or if loss   value is fewer   than previous

        if epoch == 1 or \
                value > metric["value"] and format == 'metric' or value < metric["value"] and format == 'loss':
            metric["epoch"] = epoch
            metric["value"] = value

    def __str__(self):
        return f''.join(
            f'{metric_name:10} Best Epoch: {metric_value["epoch"]:4}, Best Value: {metric_value["value"]}\n'
            for (metric_name, metric_value) in self.metrics.items())
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / ((len(a) + len(b) - len(c)) + 1e-6)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)  # ROUGE
def extractive_summarization_label(text, prime_text):
    # OUTPUT
    labels = []
    idxs = []

    # drop item if it not in both list
    XOR = set(text) ^ set(prime_text)
    prime_text = [i for i in prime_text if i not in XOR]

    for idx, token in enumerate(text):
        # first check prime_text[0] is ever exists in text

        if prime_text and token == prime_text[0]:  # if prime_text is not []
            # label
            labels.append(1)
            # `new` prime_text
            prime_text = prime_text[1:]
            # index
            idxs.append(idx)

        else:
            labels.append(0)

    return labels
def extractive_summarization_index(text, prime_text):

    # drop item if it not in both list
    XOR = set(text) ^ set(prime_text)
    prime_text = [i for i in prime_text if i not in XOR]

    idx_0 = text.index(prime_text[0])
    idx_1 = len(text) - text[::-1].index(prime_text[-1])

    if idx_0 == idx_1:  # only 1 item in idxs list will cause idx_0 == idx_1, manually assign idx_1 by '+ 1'
        idx_1 = idx_1 + 1

    indices = (idx_0, idx_1)

    return indices
# def read_csv_split(csv_path):
#     # read, select [id, q, r, s, q', r'], remove duplicates
#     df = (pd
#           .read_csv(csv_path)
#           .iloc[:, 0:6]
#           .drop_duplicates()
#           )
#
#     index_texts = []  # q r
#     label_texts = []  # new_q new_r
#
#     index_texts_wt = []  # word_tokenize
#     label_texts_wt = []  # word_tokenize
#
#     index_indices = []
#     label_labels = []
#
#     for row_idx in range(len(df)):
#         # id, q, r, s, q', r'
#         id, q, r, _, q_prime, r_prime = df.iloc[row_idx]
#
#         # remove --> start " and "
#         q, r, q_prime, r_prime = q[1:-1], r[1:-1], q_prime[1:-1], r_prime[1:-1]
#
#         # nltk word tokenize
#         q_wt = word_tokenize(q)
#         r_wt = word_tokenize(r)
#         q_prime_wt = word_tokenize(q_prime)
#         r_prime_wt = word_tokenize(r_prime)
#
#         # tokenize_text `add_prefix_space=True` for consistent tokens
#         q_text = tokenizer(q_wt, is_split_into_words=True, padding='max_length', truncation=True,
#                            # add_prefix_space=True,
#                            )['input_ids']
#
#         q_prime_text = tokenizer(q_prime_wt, is_split_into_words=True, truncation=True, add_special_tokens=False,
#                                  # add_prefix_space=True,
#                                  )['input_ids']
#
#         r_text = tokenizer(r_wt, is_split_into_words=True, padding='max_length', truncation=True,
#                            # add_prefix_space=True,
#                            )['input_ids']
#
#         r_prime_text = tokenizer(r_prime_wt, is_split_into_words=True, truncation=True, add_special_tokens=False,
#                                  # add_prefix_space=True,
#                                  )['input_ids']
#         # get indices
#         q_indices = extractive_summarization_index(q_text, q_prime_text)
#         r_indices = extractive_summarization_index(r_text, r_prime_text)
#
#         new_q = tokenizer.decode(q_text[q_indices[0]:q_indices[1]], skip_special_tokens=True)
#         new_r = tokenizer.decode(r_text[r_indices[0]:r_indices[1]], skip_special_tokens=True)
#
#         new_q_wt = word_tokenize(new_q)
#         new_r_wt = word_tokenize(new_r)
#
#         new_q_text = tokenizer(new_q_wt, is_split_into_words=True, padding='max_length', truncation=True,
#                                # add_prefix_space=True,
#                                )['input_ids']
#
#         new_r_text = tokenizer(new_r_wt, is_split_into_words=True, padding='max_length', truncation=True,
#                                # add_prefix_space=True,
#                                )['input_ids']
#
#         q_labels = extractive_summarization_label(new_q_text, q_prime_text)
#         r_labels = extractive_summarization_label(new_r_text, r_prime_text)
#
#         # index task
#         index_texts.append(q)
#         index_texts.append(r)
#
#         index_indices.append(q_indices)
#         index_indices.append(r_indices)
#
#         # label task
#         label_texts.append(new_q)
#         label_texts.append(new_r)
#
#         label_labels.append(q_labels)
#         label_labels.append(r_labels)
#
#         # wt
#         index_texts_wt.append(q_wt)
#         index_texts_wt.append(r_wt)
#
#         label_texts_wt.append(new_q_wt)
#         label_texts_wt.append(new_r_wt)
#
#     return {'index_texts': index_texts,
#             'label_texts': label_texts,
#             'index_texts_wt': index_texts_wt,
#             'label_texts_wt': label_texts_wt,
#             'index_indices': index_indices,
#             'label_labels': label_labels}
def read_csv_split(csv_path):
    # read, select [id, q, r, s, q', r'], remove duplicates
    df = (pd
          .read_csv(csv_path)
          .iloc[:, 0:6]
          .drop_duplicates()
          )

    index_texts = []  # q r
    label_texts = []  # new_q new_r

    index_texts_wt = []  # word_tokenize
    label_texts_wt = []  # word_tokenize

    index_indices = []
    label_labels = []

    for row_idx in range(len(df)):
        # id, q, r, s, q', r'
        id, q, r, _, q_prime, r_prime = df.iloc[row_idx]

        # remove --> start " and "
        q, r, q_prime, r_prime = q[1:-1], r[1:-1], q_prime[1:-1], r_prime[1:-1]

        # nltk word tokenize
        q_wt = word_tokenize(q)
        r_wt = word_tokenize(r)
        q_prime_wt = word_tokenize(q_prime)
        r_prime_wt = word_tokenize(r_prime)

        # tokenize_text `add_prefix_space=True` for consistent tokens
        q_text = tokenizer(q_wt, is_split_into_words=True, padding='max_length', truncation=True,
                           # add_prefix_space=True,
                           )['input_ids']

        q_prime_text = tokenizer(q_prime_wt, is_split_into_words=True, truncation=True, add_special_tokens=False,
                                 # add_prefix_space=True,
                                 )['input_ids']

        r_text = tokenizer(r_wt, is_split_into_words=True, padding='max_length', truncation=True,
                           # add_prefix_space=True,
                           )['input_ids']

        r_prime_text = tokenizer(r_prime_wt, is_split_into_words=True, truncation=True, add_special_tokens=False,
                                 # add_prefix_space=True,
                                 )['input_ids']
        # get indices
        q_indices = extractive_summarization_index(q_text, q_prime_text)
        r_indices = extractive_summarization_index(r_text, r_prime_text)

        # index task
        index_texts.append(q)
        index_texts.append(r)

        index_indices.append(q_indices)
        index_indices.append(r_indices)

        # wt
        index_texts_wt.append(q_wt)
        index_texts_wt.append(r_wt)


    return {
        'index_texts_wt': index_texts_wt,
        'index_indices': index_indices,
    }

class CSVDataset_extraction_label(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        text = {key: torch.tensor(val[index]) for key, val in self.texts.items()}
        label = torch.tensor(self.labels[index]).to(device)
        return {'text': text, 'label': label}

    def __len__(self):
        return len(self.labels)
class CSVDataset_extraction_index(Dataset):
    def __init__(self, texts, indices):
        self.texts = texts
        self.indices = indices

    def __getitem__(self, index):
        text = {key: torch.tensor(val[index]) for key, val in self.texts.items()}
        index = torch.tensor(self.indices[index]).to(device)
        return {'text': text, 'index': index}

    def __len__(self):
        return len(self.indices)

class NLP_model_index(nn.Module):
    def __init__(self):
        super().__init__()

        model_config = AutoConfig.from_pretrained(MODEL, output_hidden_states=True)
        self.num_layers = model_config.num_hidden_layers
        self.model = AutoModel.from_pretrained(MODEL, config=model_config)

        self.high_dropout = nn.Dropout(0.5)
        self.index_fc = nn.Linear(model_config.hidden_size * 2, 2)

    def forward(self, x):

        ids, mask = x['input_ids'], x['attention_mask']

        # last_hidden_state ->  bs * sl * hs
        # pooler_output -> bs * 768
        # hidden_state -> (12) * bs * sl * hs

        _, po, hs = self.model(ids, attention_mask=mask).to_tuple() # ONLY FOR ROBERTA
        # lhs, hs = self.model(ids, attention_mask=mask).to_tuple()

        stack_hs = torch.stack([hs[i] for i in range(self.num_layers)], dim=0)

        hs_mean = torch.mean(stack_hs, dim=0)
        hs_max = torch.max(stack_hs, dim=0)[0]

        # bs * sl * (hs * 2)
        out = torch.cat((hs_mean, hs_max), dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        index_logits = torch.mean(torch.stack([self.index_fc(self.high_dropout(out)) for _ in range(5)], dim=0), dim=0)

        # start_logits, end_logits
        # (batch_size, num_tokens, 2) -> (batch_size, num_tokens, 1) -> (batch_size, num_tokens)
        start_logits, end_logits = index_logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)

        return {'start_logits': start_logits, 'end_logits': end_logits, 'output_mask': mask}
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

        _, po, hs = self.model(ids, attention_mask=mask).to_tuple() # ONLY FOR ROBERTA
        # lhs, hs = self.model(ids, attention_mask=mask).to_tuple()

        stack_hs = torch.stack([hs[i] for i in range(self.num_layers)], dim=0)

        hs_mean = torch.mean(stack_hs, dim=0)
        hs_max = torch.max(stack_hs, dim=0)[0]

        # bs * sl * (hs * 2)
        out = torch.cat((hs_mean, hs_max), dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        label_logits = torch.mean(torch.stack([self.label_fc(self.high_dropout(out)) for _ in range(5)], dim=0), dim=0)
        label_logits = label_logits.squeeze(-1)

        return {'label_logits': label_logits, 'output_mask': mask}

ValMonitor = EpochMonitor()
if __name__ == '__main__':

    # device
    device = torch.device('cuda')
    # 'index_texts', 'label_texts', 'index_texts_wt, 'label_texts_wt', 'index_indices', 'label_labels'
    # index_texts, label_texts, index_texts_wt, label_texts_wt, index_indices, label_labels = \
    #     read_csv_split(csv_path)['index_texts'], \
    #     read_csv_split(csv_path)['label_texts'], \
    #     read_csv_split(csv_path)['index_texts_wt'], \
    #     read_csv_split(csv_path)['label_texts_wt'], \
    #     read_csv_split(csv_path)['index_indices'], \
    #     read_csv_split(csv_path)['label_labels'],
    csv_split_outputs = read_csv_split(csv_path)
    index_texts_wt, index_indices = csv_split_outputs['index_texts_wt'], csv_split_outputs['index_indices']
    # print(index_texts, index_indices, label_texts, label_labels)
    # print(len(index_texts), len(index_indices), len(label_texts), len(label_labels))
    # dataset
    # wt
    index_texts = tokenizer(index_texts_wt, is_split_into_words=True, padding='max_length', truncation=True,
                            return_tensors='pt').to(device)
    # label_texts = tokenizer(label_texts_wt, is_split_into_words=True, padding='max_length', truncation=True,
    #                         return_tensors='pt').to(device)
    # not wt
    # index_texts = tokenizer(index_texts, padding='max_length', truncation=True, return_tensors='pt').to(device)
    # label_texts = tokenizer(label_texts, padding='max_length', truncation=True, return_tensors='pt').to(device)

    # label
    # csv_dataset = CSVDataset_extraction_label(label_texts, label_labels)
    # index
    csv_dataset = CSVDataset_extraction_index(index_texts, index_indices)

    train_dataset, val_dataset = train_test_split(csv_dataset, test_size=0.1)

    # dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=0)

    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=0)

    # model
    # label
    # model = NLP_model_label().to(device)
    # index
    model = NLP_model_index().to(device)

    # optimizer
    optimizer = create_optimizer_v2(model,
                                    opt='adam',
                                    lr=1e-4,
                                    weight_decay=0)

    # scheduler
    scheduler = CosineLRScheduler(optimizer,
                                  t_initial=EPOCHS,
                                  lr_min=1e-6,
                                  warmup_t=5,
                                  warmup_lr_init=1e-5,
                                  cycle_limit=1)

    # loss
    index_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    label_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    for epoch in range(EPOCHS):
        # model
        train_losses = AverageMeter()
        model.train()
        for batch_idx, inputs in enumerate(tqdm(train_loader, total=len(train_loader))):
            num_batches_per_epoch = len(train_loader)
            num_updates = epoch * num_batches_per_epoch

            # label
            # 'text', 'label'
            # text, label = inputs['text'], inputs['label'].float().to(device)

            # index
            # 'text', 'index'
            text, index = inputs['text'], inputs['index'].to(device).T
            start_indices = index[0]
            end_indices = index[1]

            # model
            outputs = model(text)

            # label
            # 'label_logits','output_mask'
            # label_logits = outputs['label_logits'].to(device)
            # mask = outputs['output_mask'].to(device)

            # index
            # 'start_logits', 'end_logits', 'output_mask'
            start_logits = outputs['start_logits'].to(device)
            end_logits = outputs['end_logits'].to(device)
            mask = outputs['output_mask'].to(device)

            # label
            # label_loss = ((label_loss_fn(label_logits, label) * mask).sum(dim=-1) / mask.sum(dim=-1)).mean()
            # loss = label_loss

            # index
            index_loss = index_loss_fn(start_logits, start_indices) + index_loss_fn(end_logits, end_indices)
            loss = index_loss
            train_losses.update(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step_update(num_updates=num_updates, metric=train_losses.avg)

        print(f'[TRAIN] epoch{epoch + 1} total_loss:{train_losses.avg}')

        val_losses = AverageMeter()
        pred_jaccard = AverageMeter()
        pred_rouge1 = AverageMeter()
        pred_rouge2 = AverageMeter()

        model.eval()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(tqdm(val_loader, total=len(val_loader))):

                # label
                # 'text', 'label'
                # text, label = inputs['text'], inputs['label'].float().to(device)

                # index
                # 'text', 'index'
                text, index = inputs['text'], inputs['index'].to(device).T
                start_indices = index[0]
                end_indices = index[1]

                # model
                outputs = model(text)

                # label
                # 'label_logits','output_mask'
                # label_logits = outputs['label_logits'].to(device)
                # mask = outputs['output_mask'].to(device)

                # index
                # 'start_logits', 'end_logits', 'output_mask'
                start_logits = outputs['start_logits'].to(device)
                end_logits = outputs['end_logits'].to(device)
                mask = outputs['output_mask'].to(device)

                # label
                # label_loss = ((label_loss_fn(label_logits, label) * mask).sum(dim=-1) / mask.sum(dim=-1)).mean()
                # loss = label_loss

                # index
                index_loss = index_loss_fn(start_logits, start_indices) + index_loss_fn(end_logits, end_indices)
                loss = index_loss
                val_losses.update(loss)

                # index
                pred_start_indices = torch.argmax(start_logits, dim=-1).cpu()
                pred_end_indices = torch.argmax(end_logits, dim=-1).cpu()

                pred_index_tokens_to_string = \
                    [tokenizer.decode(tensors[start_index: end_index], skip_special_tokens=True) for
                     (tensors, start_index, end_index) in zip(text['input_ids'], pred_start_indices, pred_end_indices)]

                true_index_tokens_to_string = \
                    [tokenizer.decode(tensors[start_index: end_index], skip_special_tokens=True) for
                     (tensors, start_index, end_index) in zip(text['input_ids'], start_indices, end_indices)]

                # label
                # pred_logits = F.logsigmoid(label_logits).exp()  # sigmoid
                # pred_logits = (pred_logits > THRESHOLD).float()  # binarize # > threshold -> 1 ; <= threshold ->  0
                # pred_label_indices = [np.where(pred_logit == 1) for pred_logit in pred_logits.cpu()]
                # label_indices = [np.where(logit == 1) for logit in label.cpu()]

                # label
                # pred_label_tokens_to_string = \
                #     [tokenizer.decode(tensors[pred_index], skip_special_tokens=True) for
                #      (tensors, pred_index) in zip(text['input_ids'], pred_label_indices)]
                #
                # true_label_tokens_to_string = \
                #     [tokenizer.decode(tensors[true_index], skip_special_tokens=True) for
                #      (tensors, true_index) in zip(text['input_ids'], label_indices)]

                # index
                for pred, true in zip(pred_index_tokens_to_string, true_index_tokens_to_string):
                    # JACCARD
                    pred_jaccard.update(jaccard(pred, true))

                    # ROUGE
                    # reference, output
                    scores = scorer.score(true, pred)
                    pred_rouge1.update(scores['rouge1'].fmeasure)
                    pred_rouge2.update(scores['rouge2'].fmeasure)

                # label
                # for pred, true in zip(pred_label_tokens_to_string, true_label_tokens_to_string):
                #     # JACCARD
                #     pred_jaccard.update(jaccard(pred, true))
                #
                #     # ROUGE
                #     # reference, output
                #     scores = scorer.score(true, pred)
                #     pred_rouge1.update(scores['rouge1'].fmeasure)
                #     pred_rouge2.update(scores['rouge2'].fmeasure)

            ValMonitor.update('loss', epoch=epoch + 1, value=val_losses.avg, format='loss')
            ValMonitor.update('jaccard', epoch=epoch + 1, value=pred_jaccard.avg, format='metric')
            ValMonitor.update('rouge-1', epoch=epoch + 1, value=pred_rouge1.avg, format='metric')
            ValMonitor.update('rouge-2', epoch=epoch + 1, value=pred_rouge2.avg, format='metric')

            torch.save(model.state_dict(), f'{save_dir}/{epoch + 1}.pth')

            print(f'[VAL] epoch : {epoch + 1}\n'
                  f'loss : {val_losses.avg}\n'
                  f'jaccard : {pred_jaccard.avg}\n'
                  f'rouge-1 : {pred_rouge1.avg}\n'
                  f'rouge-2 : {pred_rouge2.avg}\n')

            print(ValMonitor)