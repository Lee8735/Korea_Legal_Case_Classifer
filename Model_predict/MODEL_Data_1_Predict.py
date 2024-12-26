#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras._tf_keras.keras.models import *
from keras._tf_keras.keras.layers import *
from keras._tf_keras.keras.utils import to_categorical

from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

from keras._tf_keras.keras.models import load_model

import pickle
import re
from konlpy.tag import Okt
from nltk.tokenize import word_tokenize
import glob


Remove_Words = [
    '제', 
    '점', 
    '항', 
    '조', 
    '이',
    '제', 
    '승소',
    '사건', 
    '이에',
    '판결',
    '판시', 
    '사례', 
    '사안', 
    '선고', 
    '내지', 
    '가합', 
    '구합', 
    '고단', 
    '고합',
    '가단',
    '가소',  
    '개요', 
    '요지',                  
    '카합', 
    '가소', 
    '표장',
    '제목',
    '형사',
    '병합',
    '선고한',
    '사례임',
    '기각한', 
    '사안의',               
    '민사부', 
    '대상판결', 
    '사건번호', 
    '판결요지', 
    '판시사항', 
    '주요판결',
    '서울고등법원', 
    '대전고등법원', 
    '대구고등법원', 
    '부산고등법원',
    '광주고등법원',
    '수원고등법원',
    '특허법원',
    '서울중앙지방법원',
    '서울가정법원',
    '서울행정법원',
    '서울회생법원',
    '서울동부지방법원',
    '서울남부지방법원',
    '서울북부지방법원',
    '서울서부지방법원',
    '의정부지방법원',
    '인천지방법원',
    '인천가정법원',
    '춘천지방법원',
    '대전지방법원',
    '대전가정법원',
    '청주지방법원',
    '대구지방법원',
    '대구가정법원',
    '부산지방법원',
    '부산가정법원',
    '부산회생법원',
    '울산지방법원',
    '울산가정법원',
    '창원지방법원',
    '광주지방법원',
    '광주가정법원',
    '전주지방법원',
    '제주지방법원',
    '수원지방법원',
    '수원가정법원',
    '수원회생법원'                
]


# 텍스트에서 한글만 남겨서 반환하는 함수
def Remain_Korean(text):
    if pd.isna(text):
        return pd.NA
    else:
        return re.compile('[^가-힣 ]').sub(' ', text)


# 텍스트에서 연속된 공백을 줄인 텍스트를 반환하는 함수.
def Reduce_Space(text):
    if pd.isna(text):
        return pd.NA
    else:
        return re.sub(r'\s+', ' ', text).strip()
    
    
# 단순히 단어로 토큰화 한 뒤 불용어를 제거하고 다시 문장으로 반환하는 함수.
def Tokenize_and_Remove(text, remove_words):
    tokens = word_tokenize(text)
    removed_tokens = [token for token in tokens if token not in remove_words]
    result = ' '.join(removed_tokens) 
    return result




#===================================================================================================
# 텍스트 입력받기
target_text = input('text 입력: \n')

#===================================================================================================
# 입력 데이터 전처리

# 한글만 남기고 연속공백 제거.
target_text = Remain_Korean(target_text)
target_text = Reduce_Space(target_text)


# 단어단위로 토큰화 한 후 불용어 제거.
target_text = Tokenize_and_Remove(target_text, Remove_Words)


# 형태소 분리.
okt = Okt()
target_text = okt.morphs(target_text)
print(target_text)

# 텍스트를 토큰화 했던 정보를 불러옴.
Data_Token_dir = './Make_Learning_Data/Data_1/'
with open(Data_Token_dir + '/Data_token_max_3974_21124.pickle', 'rb') as f:
    token = pickle.load(f)

# 토큰의 형태소들이 ''로 쌓여 있으므로 똑같이 맞춰줌.
    target_text = [f"'{word}'" for word in target_text]

tokened_X = token.texts_to_sequences([target_text])

# 만약 새로 크롤링한 데이터의 문장길이가 학습할 때 보다 많을경우 자른다.
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 3974:
        tokened_X[i] = tokened_X[i][:3974]

X_pad = pad_sequences(tokened_X, 3974)



# 모델을 불러와서 예측시켜 본다.
model_dir = './Model_Learning/MODEL_Data_1'
model_name = '/Korea_Legal_Case_classfier_model_0.8769305944442749.h5'
model = load_model(model_dir + model_name)
model.summary()

# 라벨 인코더 불러옴.
label_encoder_dir = './Make_Learning_Data/Data_1'
with open(label_encoder_dir + '/Category_encoder_label.pickle', 'rb') as f:
    encoder = pickle.load(f)


# 모델 예측
pred = model.predict(X_pad)
max_index = np.argmax(pred, axis=1)[0]

result = encoder.inverse_transform([max_index])


# 결과 출력
probability = pred[0][max_index] * 100
str = f'{result[0]} 사건  (확률: {probability:.2f}%)'
print(str)