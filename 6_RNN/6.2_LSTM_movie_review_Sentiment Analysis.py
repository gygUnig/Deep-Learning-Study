# reference : https://wikidocs.net/44249




import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request


from konlpy.tag import Okt

from tqdm import tqdm


# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# data load
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('../text_datasets/data_naver_movie_ratings_train.txt')
test_data = pd.read_table('../text_datasets/data_naver_movie_ratings_test.txt')



print("훈련용 리뷰 개수:", len(train_data))  # 훈련용 리뷰 개수: 150000
print(train_data[:5])

print('테스트용 리뷰 개수:', len(test_data))  # 테스트용 리뷰 개수: 50000
print(test_data[:5])


# document 열과 label 열의 중복을 제외한 값의 개수
print(train_data['document'].nunique(), train_data['label'].nunique()) # 146182 2
# 중복을 제거한 샘플의 개수가 146,182개 라는 것 -> 약 4000개의 중복 샘플이 존재


# document 열의 중복 제거
train_data.drop_duplicates(subset=['document'], inplace=True)

# 중복이 제거되었는지 전체 샘플 수 확인
print("총 샘플의 수:", len(train_data))  # 총 샘플의 수: 146183


# train_data에서 해당 리뷰의 긍, 부정 유무가 기재되어 있는 label 값 분포 확인
train_data['label'].value_counts().plot(kind = 'bar')
# plt.show()

print(train_data.groupby('label').size().reset_index(name = 'count'))
#    label  count
# 0      0  73342
# 1      1  72841


# 리뷰 중에 Null 값을 가진 샘플이 있는지 확인
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
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인
# False


# 다시 샘플의 개수 출력
print(len(train_data)) # 146182




# 데이터 전처리

# 예제 - 알파벳과 공백을 제외하고 모두 제거
eng_text = 'do!!!  you expect... people~ to~ read~ the FAQ, etc. and actually accept hard~! atheism?@@'
print(re.sub(r'[^a-zA-Z]', '', eng_text))
# doyouexpectpeopletoreadtheFAQetcandactuallyaccepthardatheism


# 한글과 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

print(train_data[:5])


train_data['document'] = train_data['document'].str.replace('^ +',"")
train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

# id            0
# document    789
# label         0
# dtype: int64


print(train_data.loc[train_data.document.isnull()][:5])

#            id document  label
# 404   4221289      NaN      0
# 412   9509970      NaN      1
# 470  10147571      NaN      1
# 584   7117896      NaN      0
# 593   6478189      NaN      0


train_data = train_data.dropna(how = 'any')
print(len(train_data))  # 145393


# test 데이터 전처리
test_data.drop_duplicates(subset = ['document'], inplace = True) # document 열에서 중복인 내용이 있다면 중복 제거

test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "") # 정규 표현식 수행

test_data['document'] = test_data['document'].str.replace('^ +', "") # 공백은 empty 값으로 변경

test_data['document'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경

test_data = test_data.dropna(how='any') # Null 값 제거

print('전처리 후 테스트용 샘플의 개수: ', len(test_data)) # 전처리 후 테스트용 샘플의 개수:  48852



# 토큰화
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']



okt = Okt()
print(okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔', stem = True))
# ['오다', '이렇다', '것', '도', '영화', '라고', '차라리', '뮤직비디오', '를', '만들다', '게', '나다', '뻔']










