# 다중회귀 모델을 짜보자
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso

Appdata = pd.read_csv(
    "/content/gdrive/MyDrive/도로노선정보-20231114/다니고밴-[전주]Type12-역방향/APP/다니고밴.csv"
)


def outlier_remove(data, threshold=2.1):
    z_scores = np.abs(data - np.mean(data)) / np.std(data)  # Z-score 계산

    filtered_data = data[z_scores < threshold]
    outlier = data[z_scores > threshold]

    return filtered_data, outlier


Appdata = pd.merge(Appdata, danigoApp)

# X 이상값 추출
filtered_time, outlier_t = outlier_remove(Appdata["소요 시간(분)"])
filtered_tl, outlier_tl = outlier_remove(Appdata["신호등 정지"])
filtered_intemp, outlier_in = outlier_remove(Appdata["In_Temp"])
filtered_outtemp, outlier_out = outlier_remove(Appdata["Out_Temp"])
filtered_dist, outlier_dist = outlier_remove(Appdata["거리"])

# Y 이상값 추출
filtered_fuel, outlier_f = outlier_remove(Appdata["연료 사용량 (km)"])

# X 이상값 제거
for i in outlier_t:
    Appdata.drop(Appdata[Appdata["소요 시간(분)"] == i].index)
for i in outlier_tl:
    Appdata.drop(Appdata[Appdata["신호등 정지"] == i].index)
for i in outlier_in:
    Appdata.drop(Appdata[Appdata["In_Temp"] == i].index)
for i in outlier_out:
    Appdata.drop(Appdata[Appdata["Out_Temp"] == i].index)
for i in outlier_dist:
    Appdata.drop(Appdata[Appdata["거리"] == i].index)
# Y 이상값 제거
for i in outlier_f:
    Appdata = Appdata[Appdata["연료 사용량 (km)"] != i]

# 값 제거 안하면 0.13441536283220157 [-0.09425034 -0.06246091  0.00426062  0.116113   -0.01573865]

# 값 제거 하면 0.12278575303028283 [-0.08173695 -0.04908362 -0.00548274  0.12570456 -0.01296417]
Appdata_X = Appdata[["거리", "In_Temp", "Out_Temp", "신호등 정지", "소요 시간(분)"]]
Appdata_Y = Appdata["연료 사용량 (km)"]

x_train, x_test, y_train, y_test = train_test_split(
    Appdata_X, Appdata_Y, train_size=0.8, test_size=0.2
)

w1 = tf.Variable(tf.random.uniform([1]))
w2 = tf.Variable(tf.random.uniform([1]))
w3 = tf.Variable(tf.random.uniform([1]))
w4 = tf.Variable(tf.random.uniform([1]))
w5 = tf.Variable(tf.random.uniform([1]))
b = tf.Variable(tf.random.uniform([1]))


def loss_function():
    # w*x는 행렬 곱셈 후, bias를 더하여 pred_y를 계산
    pred_y = (
        w1 * Appdata[["In_Temp"]].values.tolist()
        + w2 * Appdata[["Out_Temp"]].values.tolist()
        + w3 * Appdata[["거리"]].values.tolist()
        + w4 * Appdata[["소요 시간(분)"]].values.tolist()
        + b
    )
    # 평균 제곱근 오차(Mean squared error)를 손실함수(loss function)으로 활용
    cost = tf.reduce_mean(tf.square(pred_y - Appdata_Y.values.tolist()))
    return cost


# 보편적으로 가장 많이 사용하는 adam optimizer 활용
optimizer = tf.optimizers.Adam(learning_rate=0.01)
# 훈련시키기
for step in range(3000):
    cost_val = optimizer.minimize(loss_function, var_list=[w1, w2, w3, b])
    if step % 100 == 0:
        # 훈련값 출력
        print(
            step,
            "loss_value:",
            loss_function().numpy(),
            "weight:",
            w1.numpy(),
            w2.numpy(),
            w3.numpy(),
            w4.numpy(),
            w5.numpy(),
            "bias:",
            b.numpy()[0],
        )

pred_y_list = (
    w1 * Appdata[["In_Temp"]].values.tolist()
    + w2 * Appdata[["Out_Temp"]].values.tolist()
    + w3 * Appdata[["거리"]].values.tolist()
    + w4 * Appdata[["소요 시간(분)"]].values.tolist()
    + w5 * Appdata[["신호등 정지"]]
    + b
)
print(pred_y_list)  # 실제 y값과 비교

# 실제 데이터 시각화
plt.title("Orginal data")
plt.xlabel("data number")
plt.ylabel("Label Y value")
plt.plot(Appdata_X, Appdata_Y, "ro", label="Original data")
plt.show()

# 예측 데이터 시각화
plt.title("Predicted value")
plt.xlabel("data number")
plt.ylabel("Predicted Y value")
plt.plot(Appdata_Y, pred_y_list, "bo", label="Predicted data")
plt.legend()
plt.show()
"""
#General한 예측
model=LinearRegression()
model.fit(Appdata_X,Appdata_Y)
y_predict = model.predict(x_test)
print(model.score(Appdata_X,Appdata_Y))
print(model.coef_)
plt.scatter(y_test, y_predict.astype(int), alpha=0.4)
plt.xlabel("real Fuel")
plt.ylabel("predict_fuel_demand")
plt.title("MULTIPLE LINEAR REGRESSION")
plt.show() """
