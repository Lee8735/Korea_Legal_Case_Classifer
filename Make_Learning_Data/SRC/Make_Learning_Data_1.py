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

import pickle
from konlpy.tag import Okt



#===================================================================================================
#%% 전처리 된 데이터를 가져와서 디자인, 상표 카데고리를 특허와 병합
df = pd.read_csv('./Preprocessing/Pre_End/Pre_End_1.csv')
df.info()
df.head(20)

df.loc[df['Category'] == '디자인', 'Category'] = '특허'
df.loc[df['Category'] == '상표', 'Category'] = '특허'

df['Category'].value_counts()



#===================================================================================================
#%% 카테고리 더미화.

encoder = LabelEncoder()
# fit_transform은 처음 한번만 해야한다.
df['Labeled Category'] = encoder.fit_transform(df['Category'])


label = encoder.classes_
print(label)

# Y = pd.get_dummies(Y)
# print(Y.head())


# 더미화 할 때 인코더의 라벨 정보를 파일로 저장.
save_dir = './Make_Learning_Data/Data_1/'
with open(save_dir + '/Category_encoder_label.pickle', 'wb') as f:
    pickle.dump(encoder, f)



#===================================================================================================
#%% Data 더미화후 토큰 정보만 저장.

token = Tokenizer()
token.fit_on_texts(df['Data'])
tokened_X = token.texts_to_sequences(df['Data'])

# 단어 개수
# 패딩을 하기 위해서 +1로 토큰화된 개수를 늘려줌.
wordsize = len(token.word_index) + 1

print(tokened_X[:5])


# 제일 긴 문장을 찾음
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])

save_dir = './Make_Learning_Data/Data_1'
with open(save_dir + f'/Data_token_max_{max}_{wordsize}.pickle', 'wb') as f:
    pickle.dump(token, f)

# 길이가 max가 되도록 0으로 채워준다.
paded_Data = pad_sequences(tokened_X, max)



#===================================================================================================
#%% 렌덤으로 1000개를 추출해서 학습데이터와 테스트 데이터로 나눔.
# 테스트 데이터의 가사 부분이 79로 적음.

# 샘플 크기 설정
sample_size = 1000


# 첫 번째 샘플링 (원래 인덱스 유지)
sampled_df = df.groupby("Category").apply(
    lambda x: x.sample(n=min(len(x), sample_size), random_state=42)
).reset_index(level=0, drop=True)  # 원래 인덱스를 유지


# 첫 번째 샘플링 데이터의 원래 인덱스를 추출
sampled_indices = sampled_df.index


# 원래 데이터에서 첫 번째 샘플링 데이터를 제거
remaining_df = df.drop(sampled_indices)


# 두 번째 샘플링 (남은 데이터에서 추출)
sampled_df2 = remaining_df.groupby("Category").apply(
    lambda x: x.sample(n=min(len(x), sample_size), random_state=42)
).reset_index(level=0, drop=True)


# 결과 확인
print("First sampled data counts:")
print(sampled_df['Category'].value_counts())

print("\nSecond sampled data counts:")
print(sampled_df2['Category'].value_counts())

save_dir = './Make_Learning_Data/Data_1/'
sampled_df.to_csv(save_dir + 'Learn_Data.csv', index=False)
sampled_df2.to_csv(save_dir + 'Test_Data.csv', index=False)



#===================================================================================================
#%%
load_dir = './Make_Learning_Data/Data_1/'

# Category 더미화
encoder = LabelEncoder()

with open(load_dir + 'Category_encoder_label.pickle', 'rb') as f:
    encoder = pickle.load(f)

label = encoder.classes_
labeled_Y = encoder.transform(sampled_df['Category'])
onehot_Y = to_categorical(labeled_Y)
print(onehot_Y)




# Data 더미화
with open(load_dir + 'Data_token_max_3974_21124.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_X = token.texts_to_sequences(sampled_df['Data'])

# 만약 새로 크롤링한 데이터의 문장길이가 학습할 때 보다 많을경우 자른다.
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 3974:
        tokened_X[i] = tokened_X[i][:3974]

X_pad = pad_sequences(tokened_X, 3974)

print(tokened_X[:5])




#학습데이터 저장.
Xtrain = X_pad
Ytrain = onehot_Y

print(Xtrain.shape, Ytrain.shape)

save_dir = './Make_Learning_Data/Data_1/npData/'
np.save(save_dir + f'/X_train_wordsize_{wordsize}', Xtrain)
np.save(save_dir + f'/Y_train_wordsize_{wordsize}', Ytrain)




#===================================================================================================
#%%

load_dir2 = './Make_Learning_Data/Data_1/'

# Category 더미화
encoder2 = LabelEncoder()

with open(load_dir2 + 'Category_encoder_label.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

label2 = encoder2.classes_
labeled_Y2 = encoder2.transform(sampled_df2['Category'])
onehot_Y2 = to_categorical(labeled_Y2)
print(onehot_Y2)




# Data 더미화
with open(load_dir2 + 'Data_token_max_3974_21124.pickle', 'rb') as f:
    token2 = pickle.load(f)

tokened_X2 = token2.texts_to_sequences(sampled_df2['Data'])

# 만약 새로 크롤링한 데이터의 문장길이가 학습할 때 보다 많을경우 자른다.
for i in range(len(tokened_X2)):
    if len(tokened_X2[i]) > 3974:
        tokened_X2[i] = tokened_X2[i][:3974]

X_pad2 = pad_sequences(tokened_X2, 3974)

print(tokened_X2[:5])

Xtest = X_pad2
Ytest = onehot_Y2



print(Xtest.shape, Ytest.shape)

save_dir2 = './Make_Learning_Data/Data_1/npData/'
np.save(save_dir2 + f'/X_test_wordsize_{wordsize}', Xtest)
np.save(save_dir2 + f'/Y_test_wordsize_{wordsize}', Ytest)