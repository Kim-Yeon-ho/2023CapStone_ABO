# -*- coding: utf-8 -*-
"""선형회귀

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Htuj81fET7jO-62QPluNyqMROe8rYG3X
"""

from google.colab import drive
drive.mount('/content/gdrive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.initializers import Constant
import matplotlib.pyplot as plt

danigoApp = pd.read_csv("/content/gdrive/MyDrive/다니고밴.csv", encoding = 'UTF-8')

danigoIot1 = pd.read_csv("/content/gdrive/MyDrive/D20230309.csv", encoding = 'cp949')
danigoIot2 = pd.read_csv("/content/gdrive/MyDrive/D20230310.csv", encoding = 'cp949')
danigoIot3 = pd.read_csv("/content/gdrive/MyDrive/D20230311.csv", encoding = 'cp949')
danigoIot4 = pd.read_csv("/content/gdrive/MyDrive/D20230312.csv", encoding = 'cp949')
danigoIot5 = pd.read_csv("/content/gdrive/MyDrive/D20230313.csv", encoding = 'cp949')

danigoIotDatas = []
danigoIotDatas.append(danigoIot1)
danigoIotDatas.append(danigoIot2)
danigoIotDatas.append(danigoIot3)
danigoIotDatas.append(danigoIot4)
danigoIotDatas.append(danigoIot5)

#시작일 기준 행 추출하기
dataNow = 20230309
tempAvg = []
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
        n.append(danigoIotDatas[date][(danigoIotDatas[date]['등록시간'] <= stop[i]) & (danigoIotDatas[date]['등록시간'] >= start[i])])

    dataNow += 1

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt


# CSV 파일 읽기
data = pd.read_csv("/content/gdrive/MyDrive/다니고밴 - 복사본.csv", encoding = 'UTF-8')

# 데이터 확인
print(data.head())

# 데이터 전처리
X = data['In_Temp'].values.reshape(-1, 1)  # 온도 변수
y = data['연료 사용량 (km)'].values  # 배터리 소모량 변수

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 학습 결과 확인
print(f'회귀 계수 (기울기): {model.coef_[0]}')
print(f'절편: {model.intercept_}')

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 모델 평가
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# 시각화
plt.scatter(X_test, y_test, color='black', label='Actual Consumption')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Temperature')
plt.ylabel('Consumption')
plt.title('Linear Regression Model')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt


# CSV 파일 읽기
data = pd.read_csv("/content/gdrive/MyDrive/다니고밴 - 복사본.csv", encoding = 'UTF-8')

# 데이터 확인
print(data.head())

# 거리가 4.2인 데이터만 선택
distance_data = data[data['거리'] == 6.6]


# 데이터 전처리
X = distance_data['신호등 정지'].values.reshape(-1, 1)  # 온도 변수
y = distance_data['연료 사용량 (km)'].values  # 배터리 소모량 변수

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 학습 결과 확인
print(f'회귀 계수 (기울기): {model.coef_[0]}')
print(f'절편: {model.intercept_}')

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 모델 평가
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# 시각화
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('')
plt.ylabel('Consumption')
plt.title('Linear Regression Model')
plt.show()