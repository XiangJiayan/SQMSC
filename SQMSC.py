import numpy as np
import pandas as pd
import random
import time
import copy
import matplotlib .pyplot as plt
def init(QoSAttribute):
    path = 'qws2resetIndex.csv'
    data = pd.read_csv(path)
    data = data.reset_index(drop=True)
    for i in QoSAttribute:
        if i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Compliance' or i == 'Best Practices':
            data[[i]] = data[[i]]/100
    data = data[QoSAttribute]
    # candidate_number为起始的候选集服务个数
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
        # 正属性加上负号,便于正负属性同时处理
        if i=='Response Time' or i=='Latency' :
            df[[i]] = df[[i]]
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
    # 根据任务选出理想的最好与最差组合，为归一化提供数据
    df_Max_Min = pd.DataFrame(index = ['max','min'],columns = QoSAttribute)
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

def SkylinePart( tplAll ,QoSAttribute):
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
    # 获得A子任务的Skyline服务集
    dfASky = skyline(dfA, QoSAttribute)
    # 获得B子任务的Skyline服务集
    dfBSky = skyline(dfB, QoSAttribute)
    # 获得C子任务的Skyline服务集
    dfCSky = skyline(dfC, QoSAttribute)
    # 获得D子任务的Skyline服务集
    dfDSky = skyline(dfD, QoSAttribute)
    # 获得E子任务的Skyline服务集
    dfESky = skyline(dfE, QoSAttribute)
    # 获得F子任务的Skyline服务集
    dfFSky = skyline(dfF, QoSAttribute)
    # 获得G子任务的Skyline服务集
    dfGSky = skyline(dfG, QoSAttribute)
    # 获得H子任务的Skyline服务集
    dfHSky = skyline(dfH, QoSAttribute)
    # 获得I子任务的Skyline服务集
    dfISky = skyline(dfI, QoSAttribute)
    # 获得C和D复合服务组合
    dfCD = Seq(dfCSky, dfDSky, QoSAttribute)
    # 获得CD复合服务的Skyline服务集
    dfCDSky = skyline(dfCD, QoSAttribute)
    # 获得B，CD和E的复合服务组合
    dfBCDE = Xor1(dfBSky, dfCDSky, dfESky, QoSAttribute)
    # 获得BCDE复合服务的Skyline服务集
    dfBCDESky = skyline(dfBCDE, QoSAttribute)
    # 获得G和H复合服务组合
    dfGH = And1(dfGSky, dfHSky, QoSAttribute)
    # 获得GH复合服务的Skyline服务集
    dfGHSky = skyline(dfGH, QoSAttribute)

    # 注意，返回的除了Skyline服务集还有最大值和最小值
    return dfASky,dfBCDESky,dfFSky,dfGHSky,dfISky,df_min_max



def initGA(tplAll,parameter):
    # 初始化种群
    population = []
    for i in range(parameter['population_size']):
        temp = []
        for j in range(parameter['task_number']):
            # 为每个种群个体的每个子任务随机选择候选集中的数据
            k = random.randint(0, len(tplAll[j].values)-1)
            temp.append(tplAll[j].values[k])
        population.append(temp)
    return population

def GA_QoS_Computing(QoS,QoSAttribute):
    # 计算复合服务的QoS值，其中传进来的QoS正属性均是负数
    a=[]
    for i in range(len(QoSAttribute)):
        if QoSAttribute[i] == 'Response Time' or QoSAttribute[i] == 'Latency':
            a.append(max(QoS[0][i], QoS[1][i], QoS[2][i] , QoS[3][i], QoS[4][i]))
        elif QoSAttribute[i] == 'Availability' or QoSAttribute[i] == 'Successability' or QoSAttribute[i] == 'Reliability' or QoSAttribute[i] == 'Compliance' or QoSAttribute[i] == 'Best Practices':
            a.append(-abs(max(QoS[0][i], QoS[1][i], QoS[2][i], QoS[3][i], QoS[4][i])))
        elif QoSAttribute[i] == 'Throughput':
            a.append(-abs(max(QoS[0][i], QoS[1][i], QoS[2][i], QoS[3][i], QoS[4][i])))
    mat = np.array(a)
    return mat

def GA_Selection(populationRaw,parameter,QoSAttribute):
    population=copy.deepcopy(populationRaw)
    #将正属性的值前加符号，方便Skyline service的评判
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k]=='Availability' or QoSAttribute[k]=='Successability' or QoSAttribute[k]=='Reliability' or QoSAttribute[k]=='Throughput' or QoSAttribute[k]=='Compliance' or QoSAttribute[k]=='Best Practices':
                    population[i][j][k]=-population[i][j][k]

    new_population=[]
    i = 0
    # 选择种群中的Skyline个体组成新的种群
    while i < 100:
        dominated = True
        A_po = random.choice(population)
        A = GA_QoS_Computing(A_po,QoSAttribute)
        for j in range(parameter['population_size']):
            B = GA_QoS_Computing(population[j],QoSAttribute)
            if all(A == B):
                continue
            if A_dominates_B(B, A):
                dominated = False
        if dominated:
            i += 1
            new_population.append(A_po)
    new_population = np.array(new_population)

    # 将加上的负号去除
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k]=='Availability' or QoSAttribute[k]=='Successability' or QoSAttribute[k]=='Reliability' or QoSAttribute[k]=='Throughput' or QoSAttribute[k]=='Compliance' or QoSAttribute[k]=='Best Practices':
                    new_population[i][j][k]=-new_population[i][j][k]
    return new_population

def GA_Crossover(populationRaw,parameter):
    # 交叉概率
    cp = parameter['crossover_probability']
    population = copy.deepcopy(populationRaw).tolist()
    new_population = []
    crossover_population = []

    for i in population:
        # 随机生成一个0到1的数，如果小于交叉概率则要进行交叉操作，反之不进行交叉操作
        r = random.random()
        if r <= cp:
            crossover_population.append(i)
        else:
            new_population.append(i)

    if len(crossover_population) % 2 != 0:
        new_population.append(crossover_population[len(crossover_population) - 1])
        del crossover_population[len(crossover_population) - 1]

    for i in range(0, len(crossover_population), 2):
        i_crossover = crossover_population[i]
        j_crossover = crossover_population[i + 1]
        crossover_position = random.randint(1, parameter['task_number']-2)
        left_i = copy.deepcopy(i_crossover[0:crossover_position])
        right_i = copy.deepcopy(i_crossover[crossover_position:parameter['task_number']])
        left_j = copy.deepcopy(j_crossover[0:crossover_position])
        right_j = copy.deepcopy(j_crossover[crossover_position:parameter['task_number']])

        # 生成新个体
        new_i = copy.deepcopy(left_i + right_j)
        new_j = copy.deepcopy(left_j + right_i)
        new_population.append(new_i)
        new_population.append(new_j)
        if (i + 1) == (len(crossover_population) - 1):
            break
    new_population=np.array(new_population)
    return new_population

def GA_Mutation(populationRaw, parameter,tplAll, QoSAttribute):
    # 突变概率
    mp = parameter['mutation_probability']
    population = copy.deepcopy(populationRaw).tolist()
    new_population = []
    for i in population:
        # 随机生成一个0到1的数，如果小于突变概率则要进行突变操作，反之不进行突变操作
        r = random.random()
        if r <= mp:
            mutation_position = random.randint(0, parameter['task_number']-1)
            tpl=tplAll[mutation_position].values
            qos = []
            for j in range(len(QoSAttribute)):
                qos.append(tpl[random.randint(0, len(tpl)-1)][j])
            i[mutation_position] = qos
            new_population.append(i)
        else:
            new_population.append(i)
    new_population=np.array(new_population)
    return new_population

def GA_FuzzingMatch(population, tplAll,parameter):
    """
        用最小欧氏距离，在候选服务集中寻找种群中与每个个体的单个服务值最接近的真实值
        参数：种群
    """
    new_population = []  # 初始化新种群
    # 对于种群中的每个个体
    for i in range(parameter['population_size']):
        temp_list =[]
        # 对于每个个体的每个任务
        for j in range(parameter['task_number']):
            map_service = []  # 初始化匹配的服务
            # 对于每个任务的候选服务集
            difference = 1000000
            E_distance = 0  # 初始化欧氏距离列表
            # 第j个任务的候选服务集
            candidates_j = tplAll[j].values
            # 欧氏距离中被替换个体的参数
            s1=population[i][j][0]
            s2=population[i][j][1]
            s3=population[i][j][2]
            s4=population[i][j][3]
            # s5=population[i][j][4]

            for k in range(len(tplAll[j].values)):
                # 欧氏距离中替换个体的参数
                sj1 = candidates_j[k][0]
                sj2 = candidates_j[k][1]
                sj3 = candidates_j[k][2]
                sj4 = candidates_j[k][3]
                # sj5 = candidates_j[k][4]
                E_distance = abs(s1 - sj1) + abs(s2 - sj2) + abs(s3 - sj3) + abs(s4 - sj4) #+ abs(s5 - sj5)
                # 为种群第i个个体任务j的服务 匹配欧氏距离最小的真实服务
                if E_distance < difference:
                    min=candidates_j[k]
                    difference = E_distance
            map_service=min
            # 将第i个个体第j个任务的真实服务添加进来
            temp_list.append(map_service)
        # 将第i个个体所有任务的真实服务添加进来
        new_population.append(temp_list)
    return new_population

def GA_Computing(QoS, QoSAttribute):
    # 计算复合服务的QoS值
    a=[]
    for i in range(len(QoSAttribute)):
        if QoSAttribute[i] == 'Response Time' or QoSAttribute[i] == 'Latency':
            a.append(max(QoS[0][i], QoS[1][i], QoS[2][i], QoS[3][i], QoS[4][i]))
        elif QoSAttribute[i] == 'Availability' or QoSAttribute[i] == 'Successability' or QoSAttribute[i] == 'Reliability' or QoSAttribute[i] == 'Compliance' or QoSAttribute[i] == 'Best Practices':
            a.append(min(QoS[0][i], QoS[1][i], QoS[2][i], QoS[3][i], QoS[4][i]))
        elif QoSAttribute[i] == 'Throughput':
            a.append(min(QoS[0][i], QoS[1][i], QoS[2][i], QoS[3][i], QoS[4][i]))
    mat = np.array(a)
    return mat

def GA_Normalization(qos, df_Max_Min, QoSAttribute):
    # 对QoS属性归一化，均是值越大越好
    df=pd.DataFrame(qos,columns=QoSAttribute)
    for i in QoSAttribute:
        # 如果最大值等于最小值，那么归一化的值为1
        if df_Max_Min.loc['max', i] == df_Max_Min.loc['min', i]:
            df[[i]] = 1
        else:
            if i == 'Response Time' or i == 'Latency':
                df[[i]] = abs(
                    (df[[i]] - df_Max_Min.loc['max', i]) / (df_Max_Min.loc['min', i] - df_Max_Min.loc['max', i]))
            elif i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Throughput' or i == 'Compliance' or i == 'Best Practices':
                df[[i]] = abs(
                    (df[[i]] - df_Max_Min.loc['min', i]) / (df_Max_Min.loc['max', i] - df_Max_Min.loc['min', i]))
    return df

def GA_get_Min_Max(tplAll,QoSAttribute):
    dfA=tplAll[0]
    dfBCDE=tplAll[1]
    dfF=tplAll[2]
    dfGH=tplAll[3]
    dfI=tplAll[4]
    # 得到每个候选服务集里每个QoS属性的最大值和最小值，根据任务选出理想的最好与最差组合，为归一化提供数据
    df_Max_Min = pd.DataFrame(index=['max', 'min'], columns=QoSAttribute)
    for i in QoSAttribute:
        if i == 'Response Time' or i == 'Latency':
            df_Max_Min.loc['max', i] = dfA[i].max() + dfBCDE[i].max() + dfF[i].max() + dfGH[i].max() + dfI[i].max()
            df_Max_Min.loc['min', i] = dfA[i].min() + dfBCDE[i].min() + dfF[i].min() + dfGH[i].min() + dfI[i].min()
        elif i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Compliance' or i == 'Best Practices':
            df_Max_Min.loc['max', i] = dfA[i].max() * dfBCDE[i].max() * dfF[i].max() * dfGH[i].max() * dfI[i].max()
            df_Max_Min.loc['min', i] = dfA[i].min() * dfBCDE[i].min() * dfF[i].min() * dfGH[i].min() * dfI[i].min()
        elif i == 'Throughput':
            df_Max_Min.loc['max', i] = min(dfA[i].max(), dfBCDE[i].max(), dfF[i].max(), dfGH[i].max(), dfI[i].max())
            df_Max_Min.loc['min', i] = min(dfA[i].min(), dfBCDE[i].min(), dfF[i].min(), dfGH[i].min(), dfI[i].min())
    return df_Max_Min

def GA_Fitness(population, QoSAttribute, df_max_min):
    # 计算种群的适应度
    qos=[]
    for i in population:
        qos.append(GA_Computing(i, QoSAttribute))
    qos=np.array(qos)
    dfMaxMin= df_max_min
    dfNor=GA_Normalization(qos,dfMaxMin,QoSAttribute)
    return dfNor.values.sum()

def depart(tplAll):
    return tplAll[0], tplAll[1], tplAll[2], tplAll[3], tplAll[4]

def GAPart(tplAll, QoSAttribute):
    # 定义参数
    parameter = {'crossover_probability': 0.88, 'mutation_probability': 0.1, 'task_number': 5, 'population_size': 100,'iteration_number': 100}
    # tplAll最后一个是最大值和最小值，要注意分离
    df_max_min=tplAll[5]
    tpl=depart(tplAll)
    # 初始化种群
    population = initGA(tpl,parameter)
    list = []

    for i in range(100):
        # 选择
        population_Select = GA_Selection(population, parameter, QoSAttribute)
        # 交叉
        population_Crossover = GA_Crossover(population_Select, parameter)
        # 突变
        population_Mutation = GA_Mutation(population_Crossover, parameter, tplAll, QoSAttribute)
        # 从真实数据集中选择与种群中数据最相似的来代替这个种群
        population_Real = GA_FuzzingMatch(population_Mutation, tplAll, parameter)

        if GA_Fitness(population_Real, QoSAttribute, df_max_min) > GA_Fitness(population, QoSAttribute, df_max_min):
            population = population_Real
        list.append(GA_Fitness(population, QoSAttribute, df_max_min) / len(QoSAttribute))

    plt.plot(list, color='red')
    plt.ylabel(u'Overall Optimality')
    plt.xlabel(u'Iterations')
    plt.show()

if  __name__ == "__main__" :
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
    # QoS属性个数
    for i in range(3):
        print('please input the number of the ', end="")
        print(i+1, end="")
        print('th QoSAttribute:', end="")
        qos=eval(input())
        QoSAttribute.append(dic[qos])

    tplAll = init(QoSAttribute)  # tplAll[0]:dfA  tplAll[1]:dfB tplAll[2]:dfC tplAll[3]:dfD tplAll[4]:dfE
    # 获得每个子任务的Skyline服务集
    tpl=SkylinePart(tplAll,QoSAttribute)
    # 使用遗传算法来找寻最好的优化性能组合
    GAPart(tpl, QoSAttribute)
