import pandas as pd
import time
import matplotlib .pyplot as plt
def init(QoSAttribute):
    path = 'qws2resetIndex.csv'
    data = pd.read_csv(path)
    data = data.reset_index(drop=True)
    for i in QoSAttribute:
        if i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Compliance' or i == 'Best Practices':
            data[[i]] = data[[i]]/100
    data = data[QoSAttribute]
    # 候选集服务个数
    candidate_number=40
    dfA = data[0:candidate_number]
    dfB = data[candidate_number:candidate_number*2]
    dfC = data[candidate_number*2:candidate_number*3]
    dfD = data[candidate_number*3:candidate_number*4]
    dfE = data[candidate_number*4:candidate_number*5]
    dfF = data[candidate_number*5:candidate_number*6]
    dfG = data[candidate_number*6:candidate_number*7]
    dfH = data[candidate_number*7:candidate_number*8]
    dfI = data[candidate_number*8:candidate_number*9]
    return dfA,dfB,dfC,dfD,dfE,dfF,dfG,dfH,dfI

def skyline(dfRaw, QoSAttribute):
    # 选出dfRaw中的Skyline服务
    df=dfRaw.copy(deep=True)
    for i in QoSAttribute:
        if i=='Response Time' or i=='Latency' :
            df[[i]] = df[[i]]
        # 正属性加上负号,便于正负属性同时处理
        elif i=='Availability' or i=='Successability' or i=='Reliability' or i=='Throughput' or i=='Compliance' or i=='Best Practices':
            df[[i]] = -df[[i]]

    Candidate=df.values
    skyline_nos=list()
    # 判断是否被支配,若不被任何服务支配,则是Skyline服务
    for i in Candidate:
        dominated=True
        for j in Candidate:
            if all(i==j):
                continue
            if A_dominates_B(j,i):
                dominated=False
        if dominated:
            skyline_nos.append(i)
    dfNew = pd.DataFrame(skyline_nos, columns=QoSAttribute)
    # 选出Skyline服务之后要将正属性的负号去掉
    for i in QoSAttribute:
        if i=='Response Time' or i=='Latency' :
            dfNew[[i]] = dfNew[[i]]
        elif i=='Availability' or i=='Successability' or i=='Reliability' or i=='Throughput' or i=='Compliance' or i=='Best Practices':
            dfNew[[i]] = -dfNew[[i]]
    return dfNew

def A_dominates_B(A,B):
    if all(A<=B) and any(A<B):
        isDominated=True
    else:
        isDominated=False
    return isDominated

def Xor1(dfB,dfCD,dfE, QoSAttribute):
    # 异或关系,且子任务有三个(B,CD,E),对其遍历并进行组合,计算复合服务的QoS属性值
    dfBCDE = pd.DataFrame(columns=QoSAttribute)
    for index, row in dfB.iterrows():
        for indexs1, rows1 in dfCD.iterrows():
            for indexs2, rows2 in dfE.iterrows():
                rowList = list()
                for i in QoSAttribute:
                    if i=='Response Time' or i=='Latency' :
                        rowList.append(max(row[i],rows1[i],rows2[i]))
                    elif i=='Availability' or i=='Successability' or i=='Reliability' or i=='Compliance' or i=='Best Practices':
                        rowList.append(min(row[i],rows1[i],rows2[i]))
                    elif i=='Throughput':
                        rowList.append(min(row[i],rows1[i],rows2[i]))

                dfBCDE.loc[len(dfBCDE)] = rowList
    return dfBCDE

def And1(dfX,dfY, QoSAttribute):
    # 与关系，且子任务有两个（X，Y），对其遍历并进行组合,计算复合服务的QoS属性值
    dfXY = pd.DataFrame(columns=QoSAttribute)
    for index, row in dfX.iterrows():
        for indexs, rows in dfY.iterrows():
            rowList = list()
            for i in QoSAttribute:
                if i == 'Response Time' or i == 'Latency':
                    rowList.append(max(row[i], rows[i]))
                elif i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Compliance' or i == 'Best Practices':
                    rowList.append(row[i]*rows[i])
                elif i == 'Throughput':
                    rowList.append(row[i]+rows[i])
            dfXY.loc[len(dfXY)] = rowList
    return dfXY

def Seq(dfX, dfY, QoSAttribute):
    # 顺序关系，且子任务有两个（X，Y），对其遍历并进行组合,计算复合服务的QoS属性值
    dfXY = pd.DataFrame(columns=QoSAttribute)
    for index, row in dfX.iterrows():
        for indexs, rows in dfY.iterrows():
            rowList = list()
            for i in QoSAttribute:
                if i == 'Response Time' or i == 'Latency':
                    rowList.append(row[i]+rows[i])
                elif i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Compliance' or i == 'Best Practices':
                    rowList.append(row[i]*rows[i])
                elif i == 'Throughput':
                    rowList.append(min(row[i], rows[i]))
            dfXY.loc[len(dfXY)] = rowList
    return dfXY

def Normalization(dfRaw, df_Max_Min, QoSAttribute):
    # 对QoS属性归一化，均是值越大越好
    df = dfRaw.copy(deep=True)
    for i in QoSAttribute:
        # 如果最大值等于最小值，那么归一化的值为1
        if df_Max_Min.loc['max', i]==df_Max_Min.loc['min', i]:
            df[[i]]=1
        else:
            if i == 'Response Time' or i == 'Latency':
                df[[i]]=abs((df[[i]]-df_Max_Min.loc['max', i])/(df_Max_Min.loc['min', i]-df_Max_Min.loc['max', i]))
            elif i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Throughput' or i == 'Compliance' or i == 'Best Practices':
                df[[i]] = abs((df[[i]] - df_Max_Min.loc['min', i]) / (df_Max_Min.loc['max', i] - df_Max_Min.loc['min', i]))
    return df

def get_Min_Max(dfA, dfB, dfC, dfD, dfE, dfF, dfG, dfH, dfI, QoSAttribute) :
    df_Max_Min = pd.DataFrame(index = ['max','min'],columns = QoSAttribute)
    # 根据任务选出理想的最好与最差组合，为归一化提供数据
    for i in QoSAttribute:
        if i == 'Response Time' or i == 'Latency':
            df_Max_Min.loc['max', i] = dfA[i].max() + max(dfB[i].max(), dfC[i].max() + dfD[i].max(), dfE[i].max()) + dfF[i].max() + max(dfG[i].max(), dfH[i].max()) + dfI[i].max()
            df_Max_Min.loc['min', i] = dfA[i].min() + max(dfB[i].min(), dfC[i].min() + dfD[i].min(), dfE[i].min()) + dfF[i].min() + min(dfG[i].min(), dfH[i].min()) + dfI[i].min()
        elif i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Compliance' or i == 'Best Practices':
            df_Max_Min.loc['max', i] = dfA[i].max() * min(dfB[i].max(), dfC[i].max() * dfD[i].max(), dfE[i].max()) * dfF[i].max() * dfG[i].max() * dfH[i].max() * dfI[i].max()
            df_Max_Min.loc['min', i] = dfA[i].min() * min(dfB[i].min(), dfC[i].min() * dfD[i].min(), dfE[i].min()) * dfF[i].min() * dfG[i].min() * dfH[i].min() * dfI[i].min()
        elif i == 'Throughput':
            df_Max_Min.loc['max', i] = min(dfA[i].max(), min(dfB[i].max(), min(dfC[i].max(), dfD[i].max()), dfE[i].max()), dfF[i].max(), dfG[i].max()+dfH[i].max(), dfI[i].max())
            df_Max_Min.loc['min', i] = min(dfA[i].min(), min(dfB[i].min(), min(dfC[i].min(), dfD[i].min()), dfE[i].min()), dfF[i].min(), dfG[i].min()+dfH[i].min(), dfI[i].min())
    return df_Max_Min

def process_data( tplAll ,QoSAttribute):
    #  tplAll[0] : dfA  tplAll[1] : dfB tplAll[2] : dfC tplAll[3] : dfD tplAll[4] : dfE
    '''
    in our tree structure,we need to process C and D at first,and return CD
    next,processing B,CD and E is what we need to do and returning BCDE
    at the same time, we can process G and H,return GH
    last,processing ABCDEFGHI
    :param tplAll:
    :return:
    '''
    # 获取每个子任务的候选集
    dfA = tplAll[0]
    dfB = tplAll[1]
    dfC = tplAll[2]
    dfD = tplAll[3]
    dfE = tplAll[4]
    dfF = tplAll[5]
    dfG = tplAll[6]
    dfH = tplAll[7]
    dfI = tplAll[8]
    df_min_max = get_Min_Max(dfA, dfB, dfC, dfD, dfE, dfF, dfG, dfH, dfI, QoSAttribute)

    # 获得C子任务的Skyline服务集
    dfCSky = skyline(dfC, QoSAttribute)
    # 获得D子任务的Skyline服务集
    dfDSky = skyline(dfD, QoSAttribute)
    # 获得C和D的复合服务组合
    dfCD = Seq(dfCSky, dfDSky, QoSAttribute)
    # 获得B子任务的Skyline服务集
    dfBSky = skyline(dfB, QoSAttribute)
    # 获得CD的复合服务的Skyline服务集
    dfCDSky = skyline(dfCD, QoSAttribute)
    # 获得E子任务的Skyline服务集
    dfESky = skyline(dfE, QoSAttribute)
    # 获得B，CD，E的复合服务组合
    dfBCDE = Xor1(dfBSky, dfCDSky, dfESky, QoSAttribute)
    # 获得BCDE复合服务的Skyline服务集
    dfBCDESky = skyline(dfBCDE, QoSAttribute)
    # 获得A子任务的Skyline服务集
    dfASky = skyline(dfA, QoSAttribute)
    # 获得A，BCDE的复合服务组合
    dfABCDE = Seq(dfASky, dfBCDESky, QoSAttribute)
    # 获得ABCDE复合服务的Skyline服务集
    dfABCDESky = skyline(dfABCDE, QoSAttribute)
    # 获得G子任务的Skyline服务集
    dfGSky = skyline(dfG, QoSAttribute)
    # 获得H子任务的Skyline服务集
    dfHSky = skyline(dfH, QoSAttribute)
    # 获得F子任务的Skyline服务集
    dfFSky = skyline(dfF, QoSAttribute)
    # 获得ABCDE和F的复合服务组合
    dfABCDEF = Seq(dfABCDESky, dfFSky, QoSAttribute)
    # 获得ABCDEF复合服务的Skyline服务集
    dfABCDEFSky = skyline(dfABCDEF, QoSAttribute)
    # 获得G和H的复合服务组合
    dfGH = And1(dfGSky, dfHSky, QoSAttribute)
    # 获得GH复合服务的Skyline服务集
    dfGHSky = skyline(dfGH, QoSAttribute)

    #由于当QoS个数大于4个或者候选服务集个数过多时，dfABCDEFSky的个数非常多，导致计算量巨大，所以选择dfABCDEFSky中Fitness前100个作为选择的数据进行下一轮计算
    #当QoS属性个数为3时，我们取前100个，属性个数为4时，取前150
    dfTemp=dfABCDEFSky.copy(deep=True)
    dfTempNor=Normalization(dfTemp,df_min_max,QoSAttribute)
    dfABCDEFSky['sum']=dfTempNor.apply(lambda x: x.sum(), axis=1)
    dfABCDEFSkySelect=dfABCDEFSky.sort_values(by='sum', ascending=False)
    dfABCDEFSkySelect= dfABCDEFSkySelect.head(100)
    dfABCDEFSkySelect = dfABCDEFSkySelect.reset_index(drop=True)
    dfABCDEFSkySelect.drop('sum', axis=1, inplace=True)

    # 获得I子任务的Skyline服务集
    dfISky = skyline(dfI, QoSAttribute)
    # 获得GH和I的复合服务组合
    dfGHI = Seq(dfGHSky, dfISky, QoSAttribute)
    # 获得GHI复合服务的Skyline服务集
    dfGHISky = skyline(dfGHI, QoSAttribute)

    # 获得总任务的服务组合
    dfAll = Seq(dfABCDEFSkySelect, dfGHISky, QoSAttribute)
    # 获得总任务的Skyline服务组合
    dfAllSky = skyline(dfAll, QoSAttribute)


    # 归一化
    dfAllSkyNor = Normalization(dfAllSky, df_min_max, QoSAttribute)
    # 归一化后求和
    dfAllSkyNor['sum'] = dfAllSkyNor.apply(lambda x: x.sum(), axis=1)
    dfSelect = dfAllSkyNor.sort_values(by='sum', ascending=False)
    dfSelect = dfSelect.head(100)
    dfSelect = dfSelect.sort_values(by='sum',ascending=True)
    list = dfSelect['sum'].tolist()
    # 计算最后总的优化性能
    ser = (pd.Series(list) * 100) / len(QoSAttribute)
    list = ser.tolist()
    print(list)
    plt.plot(list, color='red')
    plt.ylabel(u'Overall Optimality')
    plt.xlabel(u'Iterations')
    plt.show()

if  __name__ == "__main__" :
    n=eval(input("please input the number of the dimension of the QoS"))
    QoSAttribute = list()
    dic = {1: 'Response Time', 2: 'Availability', 3: 'Throughput', 4: 'Successability', 5: 'Reliability', 6: 'Compliance', 7: 'Best Practices', 8: 'Latency'}
    print("1:Response Time\n"
          "2:Availability\n"
          "3:Throughput\n"
          "4:Successability\n"
          "5:Reliability\n"
          "6:Compliance\n"
          "7:Best Practices\n"
          "8:Latency")
    for i in range(n):
        print('please input the number of the ', end="")
        print(i+1, end="")
        print('th QoSAttribute:', end="")
        qos=eval(input())
        QoSAttribute.append(dic[qos])

    tplAll = init(QoSAttribute)  # tplAll[0]:dfA  tplAll[1]:dfB tplAll[2]:dfC tplAll[3]:dfD tplAll[4]:dfE
    process_data(tplAll,QoSAttribute)
