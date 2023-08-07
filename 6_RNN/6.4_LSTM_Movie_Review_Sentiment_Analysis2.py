

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request

from konlpy.tag import Okt
from tqdm import tqdm
from konlpy.tag import Mecab


train_data = pd.read_table('../txt_datasets/data_naver_movie_ratings_train.txt')
test_data = pd.read_table('../txt_datasets/data_naver_movie_ratings_test.txt')


print("훈련용 리뷰 개수:", len(train_data))  # 훈련용 리뷰 개수: 150000
print(train_data[:5])  

print('테스트용 리뷰 개수:', len(test_data))  # 테스트용 리뷰 개수: 50000
print(test_data[:5])  


# document 열과 label 열의 중복을 제외한 값의 개수
print(train_data['document'].nunique(), train_data['label'].nunique())  # 146182 2
# 중복을 제거한 샘플의 개수가 146,182개 라는 것 -> 약 4000개의 중복 샘플 존재

# document 열의 중복 제거
train_data.drop_duplicates(subset=['document'], inplace=True) # inplace가 True면 원본 데이터셋에서 중복 데이터 제거

# 중복이 제거되었는지 전체 샘플 수 확인
print("총 샘플의 수 :", len(train_data))  # 총 샘플의 수 : 146183

# train_data에서 해당 리뷰의 긍, 부정 유무가 기재되어 있는 label 값 분포 확인

train_data['label'].value_counts().plot(kind='bar')
# plt.show()

print(train_data.groupby('label').size().reset_index(name='count'))
#    label  count
# 0      0  73342
# 1      1  72841


# 리뷰 중에서 Null 값을 가진 샘플이 있는지 확인
print(train_data.isnull().values.any())  # True -> 데이터 중에 Null 값을 가진 샘플 존재


# 어떤 열에 존재하는지 확인
print(train_data.isnull().sum())
# id          0
# document    1
# label       0
# dtype: int64


# Null 값을 가진 샘플이 어느 인덱스의 위치에 존재하는지 출력
print(train_data.loc[train_data.document.isnull()])
#             id document  label
# 25857  2172111      NaN      1

# Null 값을 가진 샘플 제거
train_data = train_data.dropna(how='any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인
# False

# 다시 샘플의 갯수 출력
print(len(train_data))  # 146182




# 데이터 전처리

# 예제 - 알파벳과 공백을 제외하고 모두 제거
eng_text = 'do!!!  you expect... people~ to~ read~ the FAQ, etc. and actually accept hard~! atheism?@@'
print(re.sub(r'[^a-zA-Z]', '', eng_text))
# doyouexpectpeopletoreadtheFAQetcandactuallyaccepthardatheism


# 한글과 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
# regex를 사용할 경우 정규표현식으로 원하는 값을 지정해서 변경 할 수 있다

print(train_data[:5])



# 한글이 없는 리뷰였다면 더이상 아무런 값도 없는 빈 값이 되었을 것.(한글과 띄어쓰기만 유지하고 나머지는 제거했으므로)
# 따라서 train_data에 공백만 있다면 Null 값으로 변경하도록 하고, Null 값이 존재하는지 확인
train_data['document'] = train_data['document'].str.replace('^ +',"",regex=True) # white space(공백) 데이터를 empty value로 변경

train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())
# id            0
# document    789
# label         0
# dtype: int64

# Null 값이 789개나 새로 생겼다. 출력해보자
print(train_data.loc[train_data.document.isnull()][:5])
#            id document  label
# 404   4221289      NaN      0
# 412   9509970      NaN      1
# 470  10147571      NaN      1
# 584   7117896      NaN      0
# 593   6478189      NaN      0


# 아무런 의미도 없는 데이터이므로 제거해준다
train_data = train_data.dropna(how='any')
print(len(train_data))  # 145393


# test data에 대해서도 동일한 전처리 과정을 진행한다.

# document 열에서 중복인 내용이 있다면 중복 제거
test_data.drop_duplicates(subset=['document'], inplace=True)

# 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)

# 공백은 empty 값으로 변경
test_data['document'] = test_data['document'].str.replace('^ +', "", regex=True)

# empty 값  Null 값으로 변경
test_data['document'].replace('', np.nan, inplace=True)

# Null 값 제거
test_data = test_data.dropna(how='any')

print('전처리 후 테스트용 샘플의 개수 :', len(test_data)) # 전처리 후 테스트용 샘플의 개수 : 48852



# 토큰화 - 불 용어 제거

# 불 용어 정의
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


# Okt 복습
okt = Okt()
x = okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔', stem = True)
print(x)
# ['오다', '이렇다', '것', '도', '영화', '라고', '차라리', '뮤직비디오', '를', '만들다', '게', '나다', '뻔']


# Okt는 위와 같이 konlpy에서 제공하는 형태소 분석기이다.
# 한국어를 토큰화할 때는 영어처럼 띄어쓰기 기준으로 토큰화를 하는 것이 아니라, 주로 형태소 분석기 사용
# stem = True를 사용하면 일정 수준의 정규화를 수행해준다. ex) 이런 -> 이렇다, 만드는 -> 만들다


# # train_data에 형태소 분석기를 사용하여 토큰화를 하면서 불 용어를 제거 후, X_train에 저장
# X_train = []
# for sentence in tqdm(train_data['document']):
#     tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
#     stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    
#     X_train.append(stopwords_removed_sentence)
    
# # 상위 3개 샘플 출력하여 결과 확인
# print(X_train[:3])