from google.colab import drive
drive.mount('/content/gdrive')

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from re import X
#경고문 삭제
pd.set_option('mode.chained_assignment',  None)

#다니고 데이터
danigoApp = pd.read_csv("/content/gdrive/MyDrive/다니고밴_전주_역방향_앱데이터.csv", encoding = 'UTF-8')

danigoIot1 = pd.read_csv("/content/gdrive/MyDrive/다니고밴-[전주]Type12-역방향-20230309.csv", encoding = 'UTF-8')
danigoIot2 = pd.read_csv("/content/gdrive/MyDrive/다니고밴-[전주]Type12-역방향-20230310.csv", encoding = 'UTF-8')
danigoIot3 = pd.read_csv("/content/gdrive/MyDrive/다니고밴-[전주]Type12-역방향-20230311.csv", encoding = 'UTF-8')
danigoIot4 = pd.read_csv("/content/gdrive/MyDrive/다니고밴-[전주]Type12-역방향-20230312.csv", encoding = 'UTF-8')
danigoIot5 = pd.read_csv("/content/gdrive/MyDrive/다니고밴-[전주]Type12-역방향-20230313.csv", encoding = 'UTF-8')

danigoIotDatas = []
danigoIotDatas.append(danigoIot1)
danigoIotDatas.append(danigoIot2)
danigoIotDatas.append(danigoIot3)
danigoIotDatas.append(danigoIot4)
danigoIotDatas.append(danigoIot5)

#외부온도 파일
JeonjuTemp1 = pd.read_csv("/content/gdrive/MyDrive/전주 기온 0309.csv", encoding = 'cp949')
JeonjuTemp2 = pd.read_csv("/content/gdrive/MyDrive/전주 기온 0310.csv", encoding = 'cp949')
JeonjuTemp3 = pd.read_csv("/content/gdrive/MyDrive/전주 기온 0311.csv", encoding = 'cp949')
JeonjuTemp4 = pd.read_csv("/content/gdrive/MyDrive/전주 기온 0312.csv", encoding = 'cp949')
JeonjuTemp5 = pd.read_csv("/content/gdrive/MyDrive/전주 기온 0313.csv", encoding = 'cp949')

outSideTemps = []
outSideTemps.append(JeonjuTemp1)
outSideTemps.append(JeonjuTemp2)
outSideTemps.append(JeonjuTemp3)
outSideTemps.append(JeonjuTemp4)
outSideTemps.append(JeonjuTemp5)

#외부온도 시간 계산
for i in range(len(outSideTemps)):
    temp_col = []
    n = outSideTemps[i]['일시']
    for j in range(len(outSideTemps[i])):
        sss = str(n[j]).split(' ')
        ttt = sss[1].replace(':', '')
        ddd = int(ttt) * 100
        temp_col.append(ddd)
    outSideTemps[i]['시간'] = temp_col


#시작일 기준 행 추출하기
dataNow = 20230309
tempAvg = []
outTempAvg = []
n = []
m = []
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
        m.append(outSideTemps[date][(outSideTemps[date]['시간'] <= stop[i]) & (outSideTemps[date]['시간'] >= start[i])])

    dataNow += 1

for i in range(len(n)):
    tempAvg.append(sum(n[i]['온도(°C)'])/len(n[i]['온도(°C)']))
    outTempAvg.append(sum(m[i]['기온(°C)'])/len(m[i]['기온(°C)']))

danigoApp['In_Temp'] = tempAvg
danigoApp['Out_Temp'] = outTempAvg

danigoApp = danigoApp[danigoApp['연료 사용량 (km)'] != 0]
distances = [4.2, 5.8, 6.6]

#이상치 제거
def outlier_remove(data, threshold=2):
    z_scores = np.abs(data - np.mean(data)) / np.std(data) # Z-score 계산

    filtered_data = data[z_scores < threshold]
    outlier = data[z_scores>threshold]

    return filtered_data, outlier
    #return filtered_data, outlier, q1, q3, iqr, lower_bound, upper_bound

#X 이상값 추출
filtered_time,outlier_t=outlier_remove(danigoApp['소요 시간(분)'])
filtered_tl,outlier_tl=outlier_remove(danigoApp['신호등 정지'])
filtered_intemp,outlier_in=outlier_remove(danigoApp['In_Temp'])
filtered_outtemp,outlier_out=outlier_remove(danigoApp['Out_Temp'])
filtered_dist,outlier_dist=outlier_remove(danigoApp['거리'])

#Y 이상값 추출
filtered_fuel,outlier_f=outlier_remove(danigoApp['연료 사용량 (km)'])

#X 이상값 제거
for i in outlier_t:
  danigoApp.drop(danigoApp[danigoApp['소요 시간(분)'] == i].index)
for i in outlier_tl:
  danigoApp.drop(danigoApp[danigoApp['신호등 정지'] == i].index)
for i in outlier_in:
  danigoApp.drop(danigoApp[danigoApp['In_Temp'] == i].index)
for i in outlier_out:
  danigoApp.drop(danigoApp[danigoApp['Out_Temp'] == i].index)
for i in outlier_dist:
  danigoApp.drop(danigoApp[danigoApp['거리'] == i].index)
#Y 이상값 제거
for i in outlier_f:
  danigoApp=danigoApp[danigoApp['연료 사용량 (km)'] != i]


danigoApp_X=danigoApp[['거리','Out_Temp','소요 시간(분)','신호등 정지']]
danigoApp_Y=danigoApp['연료 사용량 (km)']


x_train, x_test, y_train, y_test = train_test_split(danigoApp_X, danigoApp_Y, train_size=0.8, test_size=0.2)


model = Sequential()
model.add(Dense(1, input_dim=4, activation='linear'))


sgd = optimizers.SGD(learning_rate=0.0001)

# loss : 평가에 사용된 손실 함수입니다. 이진 분류에서는 binary_crossentropy를 사용합니다
# optimizer : 최적 파라미터를 찾는 알고리즘으로 경사 하강법의 하나인 adam을 사용합니다.
# metrics : 평가척도, 분류문제에서는 보통 accuracy를 씁니다.


model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

model.fit(x_train, y_train, epochs=2000)

print(model.predict(x_test))


score = model.evaluate(x_test, y_test)
print("정확도는 {}% 입니다.".format(score[1] * 100))

# model = LinearRegression()
# model.fit(x_train,y_train)

my_danigo = [[4.8,-5,20,7]]

my_predict = model.predict(my_danigo)
print('-------')
print(my_predict)
print('-----------')




# w1 = tf.Variable(tf.random.uniform([1]))
# w2 = tf.Variable(tf.random.uniform([1]))
# w3 = tf.Variable(tf.random.uniform([1]))
# w4 = tf.Variable(tf.random.uniform([1]))
# w5 = tf.Variable(tf.random.uniform([1]))
# b = tf.Variable(tf.random.uniform([1]))

# def loss_function():
# # w*x는 행렬 곱셈 후, bias를 더하여 pred_y를 계산
#   pred_y = w1*danigoApp[['In_Temp']].values.tolist()+w2*danigoApp[['Out_Temp']].values.tolist()+w3*danigoApp[['거리']].values.tolist()+w4*danigoApp[['소요 시간(분)']].values.tolist()+w5*danigoApp[['신호등 정지']].values.tolist()+b
#     #평균 제곱근 오차(Mean squared error)를 손실함수(loss function)으로 활용
#   cost = tf.reduce_mean(tf.square(pred_y - danigoApp_Y.values.tolist()))
#   return cost

# #보편적으로 가장 많이 사용하는 adam optimizer 활용
# optimizer = tf.optimizers.Adam(learning_rate=0.01)

# #훈련시키기
# for step in range(3000):
#     cost_val=optimizer.minimize(loss_function, var_list=[w1,w2,w3,w4,w5,b])
#     if step % 100 == 0:
# #훈련값 출력
#         print(step,"loss_value:", loss_function().numpy(), 'weight:', w1.numpy(),w2.numpy(),w3.numpy(),w4.numpy(),w5.numpy(), 'bias:', b.numpy()[0])

# pred_y_list= w1*danigoApp[['In_Temp']].values.tolist()+w2*danigoApp[['Out_Temp']].values.tolist()+w3*danigoApp[['거리']].values.tolist()+w4*danigoApp[['소요 시간(분)']].values.tolist()+w5*danigoApp[['신호등 정지']].values.tolist() +b
# print(pred_y_list) #실제 y값과 비교

# # 실제 데이터 시각화
# plt.title("Orginal data")
# plt.xlabel("data number")
# plt.ylabel("Label Y value")
# plt.plot(danigoApp_X, danigoApp_Y, 'ro', label='Original data')
# plt.show()


# # 예측 데이터 시각화
# plt.title("Predicted value")
# plt.xlabel("data number")
# plt.ylabel("Predicted Y value")
# plt.plot(danigoApp_Y, pred_y_list,'bo', label='Predicted data')
# plt.legend()
# plt.show()


# outlierdata = []
# Iotfiltered_datas = []
# for i in range(len(n)):
#     #filtered_data, outlier, q1, q3, IQR, lower_bound, upper_bound = outlier_remove(n[i]['온도(°C)'])
#     filtered_data, outlier = outlier_remove(n[i]['온도(°C)'])
#     outlierdata.append(outlier)
#     Iotfiltered_datas.append(filtered_data)


# print(Iotfiltered_datas)

# print(danigoIotDatas[0]['위도'])
# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111, projection='3d')

# longtermdanigoIot = []

# i = 0

# while (i < len(danigoIotDatas[0])):
#   longtermdanigoIot.append(danigoIotDatas[0].iloc[i])
#   i += 12


# x = []
# y = []
# z = []
# print("----------6.6km ,첫번째 반복, 구간 1에서 구간 3로 진행함-----------")
# print(longtermdanigoIot[0:18])


# # 구간 3-2 ( 0:18 ) 2-1 ( 21:41 ) 1-3 ( 50:66 ) 첫번째 반복
# for i in range(len(longtermdanigoIot[50:66])):
#   x.append(longtermdanigoIot[50:66][i][5])
#   y.append(longtermdanigoIot[50:66][i][4])
#   z.append(longtermdanigoIot[50:66][i][3])

# ax.plot(x, y, z)

# x1 = []
# y1 = []
# z1 = []
# print("----------6.6km ,두번째 반복, 구간 1에서 구간 3로 진행함-----------")
# print(longtermdanigoIot[129:153])
# # 구간 3-2 ( 73:90 ) 2-1 (97:118), 1-3 (129:153)
# for i in range(len(longtermdanigoIot[129:153])):
#   x1.append(longtermdanigoIot[129:153][i][5])
#   y1.append(longtermdanigoIot[129:153][i][4])
#   z1.append(longtermdanigoIot[129:153][i][3])

# ax.plot(x1, y1, z1)

