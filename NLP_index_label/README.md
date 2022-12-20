# 自然語言理解的解釋性資訊標記計畫競賽

## 指導教授: 劉宗榮  

## 隊名: TEAM_2245

## 隊長 : 蘇郁宸  組員: [黃裕芳](https://github.com/Andrewhsin), 林峻安, [陳柏瑋](https://github.com/bobo0303), 賴庭旭  

```
deberta-base              contains NLP_index NLP_label
microsoftdeberta-v3-base  contains NLP_index NLP_label
xlm-roberta-large         contains NLP_index

Choose one directory

NLP_index
| 
├── Batch_answers - train_data (no-blank).csv   競賽所提供訓練資料
└── Batch_answers - test_data(no_label).csv     競賽所提供測試資料
程式碼
├── NLP_index_train.py                          訓練與驗證程式碼
└── NLP_index_test.py                           測試程式碼
存放路徑
└── model                                       存放權重檔路徑資料夾

NLP_label
| 
├── Batch_answers - train_data (no-blank).csv   競賽所提供訓練資料
└── Batch_answers - test_data(no_label).csv     競賽所提供測試資料
程式碼
├── NLP_label_train.py                          訓練與驗證程式碼
└── NLP_label_test.py                           測試程式碼
存放路徑
└── model                                       存放權重檔路徑資料夾


```

## Installation

install latest Pytorch
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```
and other libraries
```
pip install -r requirements.txt
```


## Training & Valdiation

Run NLP_index_train.py 

```
python NLP_index_train.py 
```
Run NLP_label_train.py 

```
python NLP_label_train.py 
```

## Inference

###index ONLY  
Run NLPindex_test.py 

```
python NLP_index_test.py 
```
###label ONLY  
Run NLP_label_test.py 

```
python NLP_label_test.py 
```
###index + label  
first RUN NLP_index_test.py to get index output 
```
python NLP_index_test.py 
```
set index output as input, then  RUN NLP_label_test.py
```
python NLP_label_test.py 
```

## Checkpoints
Default (*.pth) saving path: `./model/`

## Acknowledgements

* [Kaggle Tweet Sentiment Extraction Competition: 1st place solution](https://github.com/heartkilla/kaggle_tweet)
* [RoBERTa - Hugging Face](https://huggingface.co/docs/transformers/model_doc/roberta)
* [pytorch](https://pytorch.org/)
* [pandas](https://github.com/pandas-dev/pandas/)
* [scikit-learn](https://github.com/scikit-learn/scikit-learn)
* [tqdm](https://github.com/tqdm/tqdm)
* [huggingface/transformers](https://github.com/huggingface/transformers)
* [timm](https://github.com/rwightman/pytorch-image-models)

