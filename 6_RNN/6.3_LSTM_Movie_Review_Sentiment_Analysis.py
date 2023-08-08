# reference : https://github.com/lih0905/korean-pytorch-sentiment-analysis/tree/master



import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from konlpy.tag import Komoran
import pandas as pd
import random
import time

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##### txt 데이터셋을 csv로 변환 ( 변환 완료 )
# columns = ['id', 'text', 'label']
# train_data = pd.read_csv('../txt_datasets/data_naver_movie_ratings_train.txt', sep='\t', names=columns, skiprows=1).dropna() # null 데이터 삭제
# test_data = pd.read_csv('../txt_datasets/data_naver_movie_ratings_test.txt', sep='\t', names=columns, skiprows=1).dropna()

# train_data.to_csv('../csv_datasets/data_naver_movie_ratings_train.csv', index=False)
# test_data.to_csv('../csv_datasets/data_naver_movie_ratings_test.csv', index=False)



#### 전처리


# Field 지정. 한글 데이터를 다루므로 토크나이저 또한 별도로 지정해야 한다.
komoran = Komoran()

TEXT = data.Field(tokenize=komoran.morphs, include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)


# torchtext에 내장된 데이터셋을 이용하는 게 아니므로, 각 컬럼별로 해당하는 Field를 지정해줘야 한다.
fields = {'text': ('text',TEXT), 'label': ('label', LABEL)}
# 딕셔너리 형식은 {csv 컬럼명 : (데이터 컬럼명, Field 이름)}

train_data, test_data = data.TabularDataset.splits(
    path = '../csv_datasets',
    train = 'data_naver_movie_ratings_train.csv',
    test = 'data_naver_movie_ratings_train.csv',
    format = 'csv',
    fields = fields
)


# 데이터에 검증 데이터가 따로 주어져 있지 않으므로 생성해준다
train_data, valid_data = train_data.split(random_state=random.seed(SEED))


# 단어 벡터는 전처리된 단어 벡터를 받는다. 한글을 지원하는 fasttext.simple.300d 사용
# 사전 훈련된 단어집에는 없는 단어는 0으로 처리하는걸 방지하기 위해 unk_init = torch.Tensor.normal 옵션을 준다.



MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data,
                max_size = MAX_VOCAB_SIZE,
                vectors = 'fasttext.simple.300d',
                unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)



# BucketIterator를 이용하여 데이터 생성자를 만든다 
BATCH_SIZE = 64
batch_sizes = (BATCH_SIZE, BATCH_SIZE, BATCH_SIZE)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = batch_sizes,
    #sort_within_batch = True,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True,
    device = device)



#### 모델 생성 - Multi-layered bi-directional LSTM
