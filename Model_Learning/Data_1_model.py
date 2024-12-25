#===================================================================================================
#%% 필요 모듈 임포트
import numpy as np
import matplotlib.pyplot as plt
import glob

from keras._tf_keras.keras.models import *
from keras._tf_keras.keras.layers import *
from keras._tf_keras.keras.callbacks import *
from keras._tf_keras.keras.optimizers import Adam


#===================================================================================================
#%% 학습데이터 불러옴

# 데이터 경로
npdata_dir = './Make_Learning_Data/Data_1/npData/'

# 와일드카드를 사용해 파일 경로 찾기
X_train_path = glob.glob(npdata_dir + '*X_train*.npy')[0]
X_test_path = glob.glob(npdata_dir + '*X_test*.npy')[0]
Y_train_path = glob.glob(npdata_dir + '*Y_train*.npy')[0]
Y_test_path = glob.glob(npdata_dir + '*Y_test*.npy')[0]

# 파일 로드
X_train = np.load(X_train_path)
X_test = np.load(X_test_path)
Y_train = np.load(Y_train_path)
Y_test = np.load(Y_test_path)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)



#===================================================================================================
#%% 모델 생성
model = Sequential()


# 자연어 할 때 맨 앞에 위치되는 레이어
# 전체 입력 데이터의 형태소들의 개수를 입력.
# 형태소들의 의미를 학습하는 레이어.
# 사람이 언어를 학습하는 방식과 같은 방식으로 학습됨.
# 단어 하나하나가 좌표를 가지고 있다고 하면,
# 하나의 단어를 좌표의 원점으로 놓고 축을 그려서 문장에서 같은 위치에 위치한
# 단어들을 축에서 가깝게 배치한다.
# 전혀 관계없는 단어들은 축에서 멀리 배치한다.
# 단어들 마다 축(차원)을 늘리면서 같은 일을 반복한다. (입력 형태소 개수 차원)
# 입력 형태소 개수 차원에서 좌표가 정해짐 -> 벡터화 한다.
# 비슷한 좌표의 단어들이 연상어가 된다.
# 관계가 있는 단어들끼리 가까워지고, 관계가 없을수록 멀어진다.
# 의미공간 상에 벡터화가 된다.
# 차원이 늘어나면 늘어날수록 거리가 멀어진다.-> 차원이 늘어날 수록 희소해진다. -> 학습이 안됨
# 차원이 늘어날수록 데이터가 많아져야 한다.(2의 제곱수로 데이터 수가 늘어나야 함.)
# 위를 차원의 저주라고 한다.
# input_dim= : 입력 문장들의 총 형태소 개수.(차원)
# output_dim= : 데이터 손실을 최대한 줄여서 300차원으로 차원을 축소한다.
# 어느정도로 차원을 축소할건지는 경험이다.
model.add(Embedding(21124, 300))


# conv레이어는 위치관계를 학습한다.
# 문장도 위치관계가 필요하다.
# 앞, 뒤 관계이기 때문에 1D이다. 
# 순서관계는 알 수 없다.
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
# 보통 conv가 가면 maxpooling이 따라간다.
# 여기서는 달라지는게 없다.(없어도 됨)
model.add(MaxPooling1D(pool_size=1))


# RNN은 입력이 두 개이다.
# 입력 하나, 나머지 하나의 입력은 출력이 다시 되돌아 간다.
# 출력이 입력으로 되돌아 갈 때 마다 1보다 작은수가 곱해진다.
# weight가 곱해지는데 반복할 수록 입력값이 작아진다.
# 즉 apple가 입력되었다면,
# a:5, p:4, p:3, l:2, e:1 씩 곱해진다.
# 따라서 입력의 길이가 짧아야 한다.
# 그래서 잘 않쓴다.

# 순서 데이터를 학습한다.
# LSTM은 원래의 값을 계속해서 다시 준다. -> 장기기억
# 데이터가 길어지면 앞의 입력이 사라지는 문제를 개선함.
# 활성 함수는 'tanh'를 사용한다. 시그모이드와 비슷하지만 범위가 -1 ~ 1이다.
# 언어는 정규분포를 따르지 않는다. -> 좌우 대칭이 아니기 때문에 음수를 버리면 안된다.
# 값을 예측하는 경우 마지막 레이어에서 활성함수를 사용하지 않는다.
# 다음 레이어도 LSTM이므로,
# 매번 출력이 되먹임 될 때 마다의 출력을 각각 저장해서 다음 레이어로 넘긴다.
# 위의 설정이 return_sequences=True이다.
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.35))


model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.35))

# 이 다음 레이어는 되먹임이 모두 끝난후의 하나의 값만 전달해 줘야 하므로,
# return_sequences=False가 된다.(기본값임)
# 마지막 레이어에 return_sequences=True를 줘도 되긴 함.
# 여기서는 그럴 필요없지만 필요할 때도 있다.
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.35))


model.add(Flatten())


model.add(Dense(128,activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(5, activation='softmax'))



#===================================================================================================
#%% 모델 학습
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

Adam(learning_rate=0.0001)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

fit_hist = model.fit(X_train, 
                     Y_train, 
                     batch_size=128, 
                     epochs=60, 
                     validation_data=(X_test, Y_test),
                     callbacks=early_stopping)

print(model.summary())

score = model.evaluate(X_test, Y_test, verbose=0)

print('Final test set accuracy', score[1])

plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()


#===================================================================================================
#%% 모델 저장.
model_dir = './Model_Learning/MODEL_Data_1/'
model_accuracy = score[1]
model_name = f'Korea_Legal_Case_classfier_model_{model_accuracy}.h5'

model.save(model_dir + model_name)