#===================================================================================================
#%% 필요 모듈 임포트
import numpy as np
import matplotlib.pyplot as plt
import glob

from keras._tf_keras.keras.models import *
from keras._tf_keras.keras.layers import *
from keras._tf_keras.keras.callbacks import *
from keras._tf_keras.keras.optimizers import Adam
from keras import saving

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

model.add(Embedding(21124, 300))

model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))

model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.35))

model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.35))

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

# 모델 저장.
model_dir = './Model_Learning/MODEL_Data_1/'
model_accuracy = score[1]
model_name = f'rea_Legal_Case_classfier_model_{model_accuracy}.h5'

model.save(model_dir + model_name)

# 모델 저장.
model_name = f'rea_Legal_Case_classfier_model_{model_accuracy}.keras'

saving.save_model(model ,model_dir + model_name)