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

komoran = Komoran()

TEXT = data.Field(tokenize=komoran.morphs)
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


# 총 단어가 몇 개인지 확인
TEXT.build_vocab(train_data)
print(len(TEXT.vocab))  # 45963

# 이 중 최소 두 번 이상 등장하는 단어의 갯수
TEXT.build_vocab(train_data, min_freq=2)
print(len(TEXT.vocab))  # 25210


# 단어를 25,000개로 끊는다
MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

print("TEXT 단어장의 갯수 : {}".format(len(TEXT.vocab)))  # 25002
print("LABEL 단어장의 갯수 : {}".format(len(LABEL.vocab)))  # 2
# <unk> 와 <pad> 토큰이 추가되어 있으므로 단어의 갯수가 25,000개이다.



# 최빈 단어들이 어떤 것인지 확인
print(TEXT.vocab.freqs.most_common(20))


# BucketIterator를 이용하여 데이터 생성자를 만든다
BATCH_SIZE = 64
batch_sizes = (BATCH_SIZE, BATCH_SIZE, BATCH_SIZE)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_sizes=batch_sizes,
    device = device,
    sort_key = lambda x: len(x.text),
    sort_within_batch = False,
)

# 생성된 데이터의 크기 확인
print(next(iter(train_iterator)).text.shape)  # torch.Size([93, 64]) -> 문장의 길이, 배치 사이즈


#### 모델 생성 - RNN

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        
        # text : [sent_Len, batch_size]
        embedded = self.embedding(text)
        # embedded : [sent_Len, batch_size, emb_dim]
        output, hidden = self.rnn(embedded)
        # output : [sent_len, batch_size, hidden_dim]
        # hidden : [1, batch_size, hidden_dim]
        
        assert torch.equal(output[-1, :, :], hidden.squeeze(0))
        
        return self.fc(hidden.squeeze(0))  # [batch_size, output_dim]


# hyper parameters
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1


# model
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# 모델의 파라미터 수 추출
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("이 모델은 {}개의 파라미터를 가지고 있다.".format(count_parameters(model)))  # 2592105


#### 모델 훈련

# optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.001)

# cost function - BCE with logits
cost_function = nn.BCEWithLogitsLoss()

# 모델과 손실함수를 GPU에 올리기
model = model.to(device)
cost_function = cost_function.to(device)

# 평가를 위해 임의의 실수를 0과 1 두 정수 중 하나로 변환하는 함수 만들기
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# 모델의 훈련과 평가를 위함 함수 만들기
def train(model, iterator, optimizer, cost_function):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        cost = cost_function(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        
        cost.backward()
        optimizer.step()
        
        
        epoch_loss += cost.item()  # tensor에 item()을 취하면 value를 반환
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 평가를 위한 함수는 gradient 업데이트를 하지 않아야 하므로 with torch.no_grad() 구문으로 감싼다.
def evaluate(model, iterator, cost_function):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            cost = cost_function(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            
            epoch_loss += cost.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 에폭마다 걸린 훈련시간을 측정하는 함수를 만든다
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Train. 검증 셋의 손실이 개선되면 모델을 저장하도록 구현
N_EPOCHS = 5
best_valid_loss = float('inf')


    
for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, cost_function)
    valid_loss, valid_acc = evaluate(model, valid_iterator, cost_function)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'checkpoint/6.2_RNN_Movie_Review.pt')
        
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:0.2f}%')
    print(f'\t  Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:0.2f}%')
            
            



