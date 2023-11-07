import tensorflow as tf
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# CSV 파일에서 데이터 로드

data = pd.read_csv('')

# -----ex) 온도와 배터리 소모량 데이터 추출 -----

# 'Temperature' 열에서 온도 데이터 추출
temperature = data['Temperature'].values

battery_consumption = data['BatteryConsumption'].values # 'BatteryConsumption' 열에서 배터리 소모량 데이터 추출

# -----시퀀스 길이와 특성 수를 정의-----

sequence_length = 10
feature_dim = 1

X_sequence, y_sequence = [], []

# -----입력 시퀀스(X_sequence)와 출력 레이블(y_sequence) 생성-----

for i in range(len(temperature) - sequence_length):
    X_sequence.append(temperature[i:i + sequence_length])
    # X_sequence는 온도 데이터의 시퀀스로, 현재 시점에서 이전 시점까지의 데이터 포함
    y_sequence.append(battery_consumption[i + sequence_length])
    # y_sequence는 배터리 소모량의 레이블로, 현재 시점에서의 배터리 소모량 포함

X_sequence = np.array(X_sequence)
# X_sequence를 NumPy 배열로 변환 -> 모델 학습에 사용
y_sequence = np.array(y_sequence)
# y_sequence도 NumPy 배열로 변환 -> 모델 학습에 사용

# -----데이터 분할 (학습 및 테스트 데이터)-----

# 데이터를 학습 및 테스트 세트로 분할할 비율 설정
split_ratio = 0.8
split_index = int(split_ratio * len(X_sequence))

# 데이터를 학습용과 테스트용으로 분할 (X_train과 y_train은 학습데이터, X_test와 y_test는 테스트 데이터)
X_train, X_test = X_sequence[:split_index], X_sequence[split_index:]
y_train, y_test = y_sequence[:split_index], y_sequence[split_index:]

#-----LSTM 모델 만들기-----

model = Sequential()
# LSTM 레이어 추가(DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업))
model.add(LSTM(10, activation = 'relu', input_shape=(sequence_length, feature_dim)))
 # 출력 레이어 추가(회귀 위해 사용)
model.add(Dense(1, activation='sigmoid'))
#모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# -----모델 구조 확인-----

model.summary()

# -----모델 학습-----

model.fit(X_train, y_train, epochs=50, batch_size=32)

# -----테스트 데이터로 예측-----
y_pred = model.predict(X_test)

# -----예측 결과와 실제 데이터 그래프로 그리기-----
plt.plot(y_test, label='실제 데이터')
plt.plot(y_pred, label='모델 예측')
plt.legend()
plt.show()