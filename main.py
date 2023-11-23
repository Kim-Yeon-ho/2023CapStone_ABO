import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.initializers import Constant
import matplotlib.pyplot as plt

danigoApp = pd.read_csv("", encoding = 'UTF-8')

danigoIot1 = pd.read_csv("", encoding = 'UTF-8')
danigoIot2 = pd.read_csv("", encoding = 'UTF-8')
danigoIot3 = pd.read_csv("", encoding = 'UTF-8')
danigoIot4 = pd.read_csv("", encoding = 'UTF-8')
danigoIot5 = pd.read_csv("", encoding = 'UTF-8')

dataNow = 20230309
#tempAvg = []
n = []
for date in range(5):
    temp = danigoApp[danigoApp['시작일'] == dataNow]

    temp['시작 시간'] = temp['시작 시간'].str.replace(':', '')
    temp['시작 시간'] = temp['시작 시간'].astype('int64')
    temp['시작 시간'] = temp['시작 시간'] * 100

    temp['종료시간'] = temp['종료시간'].str.replace(':', '')
    temp['종료시간'] = temp['종료시간'].astype('int64')
    temp['종료시간'] = (temp['종료시간'] * 100) + 59

    start = list(temp['시작 시간'])
    stop = list(temp['종료시간'])

    for i in range(len(start)):
        n.append(danigoIot1[(danigoIot1['등록시간'] <= stop[i]) & (danigoIot1['등록시간'] >= start[i])])
        #tempAvg.append(sum(n['온도(°C)'])/len(n['온도(°C)']))
    dataNow += 1

#n[0] = 첫번째 구간 전체 데이터, 인덱스로 접근

# danigoApp['Avg_Temperature'] = tempAvg
    
#엑셀 저장
#danigoApp.to_excel('danigoAppTem.xlsx', index=False)

# -----ex) 온도와 배터리 소모량 데이터 추출 -----
# 'temperature' 열에서 온도 데이터 추출
# X = danigoApp['Avg_Temperature'].values
# y = danigoApp['연료 사용량 (km)'].values # 'BatteryConsumption' 열에서 배터리 소모량 데이터 추출

# 데이터 정규화
# scaler = MinMaxScaler()  # Min-Max 스케일러를 사용하여 데이터를 [0, 1] 범위로 정규화
# X_scaled = scaler.fit_transform(X.reshape(-1, 1))  # 1차원 배열을 열벡터 형태로 변환하여 정규화

# # -----데이터 분할 (학습 및 테스트 데이터)-----

# # 데이터를 학습 및 테스트 세트로 분할할 비율 설정
# # 학습 및 테스트 데이터로 분할
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)  # 데이터를 학습용과 테스트용으로 나눔
# #(test_size : 데이터를 테스트세트로 사용할 비율)

# #-----LSTM 모델 만들기-----
# # 데이터를 LSTM 입력 형태로 변환(데이터 전처리)
# X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # LSTM 입력 형태로 데이터 변환
# #2차원인 X_train을 3차원배열로 바꿔줌
# X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# #ex) 가중치 상수 설정
# temperature_weight = 7.0 #2.0

# # LSTM 모델 생성
# model = Sequential()  # Sequential 모델 생성
# model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), kernel_initializer=Constant(value=temperature_weight)))  # LSTM 레이어 추가
# model.add(Dense(units=1))  # 출력 레이어 추가

# # 모델 컴파일
# model.compile(optimizer='adam', loss='mean_squared_error')

# # 모델 학습
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
# #(50번 반복, 한번의 반복동안 32개만큼 사용됨)

# # 모델 평가
# loss = model.evaluate(X_test, y_test)  # 테스트 데이터로 모델을 평가하고 손실 값을 출력(손실값 작을수록 성능 좋음)
# print(f'Mean Squared Error on Test Data: {loss}')

# # 예측
# y_pred = model.predict(X_test)  # 학습된 모델을 사용하여 테스트 데이터에 대한 예측 수행

# # 시각화
# plt.figure(figsize=(10, 6))

# # 실제 데이터와 모델 예측 결과 선 그래프로 표시
# plt.plot(y_test, label='Real Data', marker='o')
# plt.plot(y_pred, label='Prediction', marker='o', linestyle='dashed', color='orange')

# # 그래프 제목 및 레이블 설정
# plt.title('Prediction')
# plt.xlabel('temperture')
# plt.ylabel('Consumption')
# plt.legend()

# # 추세선 추가
# plt.plot(np.arange(len(y_test)), np.poly1d(np.polyfit(np.arange(len(y_test)), y_test, 1))(np.arange(len(y_test))), label='실제 데이터 추세', linestyle='dashed', color='green')
# plt.plot(np.arange(len(y_test)), np.poly1d(np.polyfit(np.arange(len(y_test)), y_pred.flatten(), 1))(np.arange(len(y_test))), label='모델 예측 추세', linestyle='dashed', color='red')

# # 그래프 표시
# plt.show()