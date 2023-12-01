import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#경고문 삭제
pd.set_option('mode.chained_assignment',  None)

danigoApp = pd.read_csv("/kaggle/input/adddistance/adddistanceDanigot.csv", encoding = 'cp949')

danigoIot1 = pd.read_csv("/kaggle/input/jeonju/도로노선정보-20231114 - 복사본/다니고밴-Type12-역방향/LTE Device/20230309.csv", encoding = 'UTF-8')
danigoIot2 = pd.read_csv("/kaggle/input/jeonju/도로노선정보-20231114 - 복사본/다니고밴-Type12-역방향/LTE Device/20230310.csv", encoding = 'UTF-8')
danigoIot3 = pd.read_csv("/kaggle/input/jeonju/도로노선정보-20231114 - 복사본/다니고밴-Type12-역방향/LTE Device/20230311.csv", encoding = 'UTF-8')
danigoIot4 = pd.read_csv("/kaggle/input/jeonju/도로노선정보-20231114 - 복사본/다니고밴-Type12-역방향/LTE Device/20230312.csv", encoding = 'UTF-8')
danigoIot5 = pd.read_csv("/kaggle/input/jeonju/도로노선정보-20231114 - 복사본/다니고밴-Type12-역방향/LTE Device/20230313.csv", encoding = 'UTF-8')

danigoIotDatas = []
danigoIotDatas.append(danigoIot1)
danigoIotDatas.append(danigoIot2)
danigoIotDatas.append(danigoIot3)
danigoIotDatas.append(danigoIot4)
danigoIotDatas.append(danigoIot5)

#외부온도 파일
JeonjuTemp1 = pd.read_csv("/kaggle/input/jeonjutemp/0309.csv", encoding = 'cp949')
JeonjuTemp2 = pd.read_csv("/kaggle/input/jeonjutemp/0310.csv", encoding = 'cp949')
JeonjuTemp3 = pd.read_csv("/kaggle/input/jeonjutemp/0311.csv", encoding = 'cp949')
JeonjuTemp4 = pd.read_csv("/kaggle/input/jeonjutemp/0312.csv", encoding = 'cp949')
JeonjuTemp5 = pd.read_csv("/kaggle/input/jeonjutemp/0313.csv", encoding = 'cp949')

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
newdatasForTest = danigoApp[danigoApp['거리'] == distances[2]]
        
X = newdatasForTest["Out_Temp"]
y = newdatasForTest["연료 사용량 (km)"]

#newdatasForTest.to_excel('DistancesixOnly.xlsx', index=False)
#danigoApp.to_excel('AddTempertureD.xlsx', index=False)

X = danigoApp["Out_Temp"]
y = danigoApp["연료 사용량 (km)"]

line_fitter = LinearRegression()
line_fitter.fit(X.values.reshape(-1,1), y)

print('기울기: ', line_fitter.coef_)
print('절편: ', line_fitter.intercept_)

plt.plot(X, y, 'o')
plt.plot(X,line_fitter.predict(X.values.reshape(-1,1)))
plt.show()
                