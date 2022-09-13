import random
import time

import matplotlib .pyplot as plt
import copy
import numpy as np
import pandas as pd

def data(QoSAttribute):
    path = 'qws2resetIndex.csv'
    # 读取文件
    data = pd.read_csv(path)
    data = data.reset_index(drop=True)

    for i in QoSAttribute:
        if i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Compliance' or i == 'Best Practices':
            data[[i]] = data[[i]]/100
    # 选择对应的QoS属性列
    data = data[QoSAttribute]
    # 每个候选集的服务个数
    candidate_number =100
    dfA = data[0:candidate_number]
    dfB = data[candidate_number:candidate_number * 2]
    dfC = data[candidate_number * 2:candidate_number * 3]
    dfD = data[candidate_number * 3:candidate_number * 4]
    dfE = data[candidate_number * 4:candidate_number * 5]
    dfF = data[candidate_number * 5:candidate_number * 6]
    dfG = data[candidate_number * 6:candidate_number * 7]
    dfH = data[candidate_number * 7:candidate_number * 8]
    dfI = data[candidate_number * 8:candidate_number * 9]
    dfB = dfB.reset_index(drop=True)
    dfC = dfC.reset_index(drop=True)
    dfD = dfD.reset_index(drop=True)
    dfE = dfE.reset_index(drop=True)
    dfF = dfF.reset_index(drop=True)
    dfG = dfG.reset_index(drop=True)
    dfH = dfH.reset_index(drop=True)
    dfI = dfI.reset_index(drop=True)
    return dfA,dfB,dfC,dfD,dfE,dfF,dfG,dfH,dfI

def init(tplAll,parameter):
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

def QoS_Computing(QoS,QoSAttribute):
    # 计算复合服务的QoS值，其中传进来的QoS正属性均是负数
    a=[]
    for i in range(len(QoSAttribute)):
        if QoSAttribute[i] == 'Response Time' or QoSAttribute[i] == 'Latency':
            a.append(QoS[0][i]+max(QoS[1][i], QoS[2][i]+QoS[3][i], QoS[4][i])+QoS[5][i]+max(QoS[6][i], QoS[7][i]) + QoS[8][i])
        elif QoSAttribute[i] == 'Availability' or QoSAttribute[i] == 'Successability' or QoSAttribute[i] == 'Reliability' or QoSAttribute[i] == 'Compliance' or QoSAttribute[i] == 'Best Practices':
            a.append(-abs(QoS[0][i]*max(QoS[1][i],QoS[2][i]*QoS[3][i],QoS[4][i])*QoS[5][i]*QoS[6][i]*QoS[7][i]*QoS[8][i]))
        elif QoSAttribute[i] == 'Throughput':
            a.append(-abs(max(QoS[0][i],max(QoS[1][i],max(QoS[2][i],QoS[3][i]),QoS[4][i]),QoS[5][i],QoS[6][i]+QoS[7][i],QoS[8][i])))
    mat = np.array(a)
    return mat

def A_dominates_B (A, B):

    if all(A <= B) and any(A < B):
        isDominated = True
    else:
        isDominated = False
    return isDominated

def Selection(populationRaw,parameter,QoSAttribute):
    population=copy.deepcopy(populationRaw)
    #将正属性的值前加负号，方便Skyline service的评判
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
        A = QoS_Computing(A_po,QoSAttribute)
        for j in range(parameter['population_size']):
            B = QoS_Computing(population[j],QoSAttribute)
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

def Crossover(populationRaw,parameter):
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

def Mutation(populationRaw, parameter,tplAll, QoSAttribute):
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

def Computing(QoS, QoSAttribute):
    # 计算复合服务的QoS值
    a=[]
    for i in range(len(QoSAttribute)):
        if QoSAttribute[i] == 'Response Time' or QoSAttribute[i] == 'Latency':
            a.append(QoS[0][i]+max(QoS[1][i], QoS[2][i]+QoS[3][i], QoS[4][i])+QoS[5][i]+max(QoS[6][i], QoS[7][i]) +QoS[8][i])
        elif QoSAttribute[i] == 'Availability' or QoSAttribute[i] == 'Successability' or QoSAttribute[i] == 'Reliability' or QoSAttribute[i] == 'Compliance' or QoSAttribute[i] == 'Best Practices':
            a.append(QoS[0][i]*min(QoS[1][i],QoS[2][i]*QoS[3][i],QoS[4][i])*QoS[5][i]*QoS[6][i]*QoS[7][i]*QoS[8][i])
        elif QoSAttribute[i] == 'Throughput':
            a.append(min(QoS[0][i],min(QoS[1][i],min(QoS[2][i],QoS[3][i]),QoS[4][i]),QoS[5][i],QoS[6][i]+QoS[7][i],QoS[8][i]))
    mat = np.array(a)
    return mat

def get_Min_Max(tplAll,QoSAttribute):
    # 得到每个候选服务集里每个QoS属性的最大值和最小值
    dfA=tplAll[0]
    dfB=tplAll[1]
    dfC=tplAll[2]
    dfD=tplAll[3]
    dfE=tplAll[4]
    dfF=tplAll[5]
    dfG=tplAll[6]
    dfH=tplAll[7]
    dfI=tplAll[8]
    df_Max_Min = pd.DataFrame(index=['max', 'min'], columns=QoSAttribute)
    for i in QoSAttribute:
        if i == 'Response Time' or i == 'Latency':
            df_Max_Min.loc['max', i] = dfA[i].max() + max(dfB[i].max(), dfC[i].max() + dfD[i].max(), dfE[i].max()) + \
                                       dfF[i].max() + max(dfG[i].max(), dfH[i].max()) + dfI[i].max()
            df_Max_Min.loc['min', i] = dfA[i].min() + max(dfB[i].min(), dfC[i].min() + dfD[i].min(), dfE[i].min()) + \
                                       dfF[i].min() + min(dfG[i].min(), dfH[i].min()) + dfI[i].min()
        elif i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Compliance' or i == 'Best Practices':
            df_Max_Min.loc['max', i] = dfA[i].max() * min(dfB[i].max(), dfC[i].max() * dfD[i].max(), dfE[i].max()) * \
                                       dfF[i].max() * dfG[i].max() * dfH[i].max() * dfI[i].max()
            df_Max_Min.loc['min', i] = dfA[i].min() * min(dfB[i].min(), dfC[i].min() * dfD[i].min(), dfE[i].min()) * \
                                       dfF[i].min() * dfG[i].min() * dfH[i].min() * dfI[i].min()
        elif i == 'Throughput':
            df_Max_Min.loc['max', i] = min(dfA[i].max(),
                                           min(dfB[i].max(), min(dfC[i].max(), dfD[i].max()), dfE[i].max()),
                                           dfF[i].max(), dfG[i].max() + dfH[i].max(), dfI[i].max())
            df_Max_Min.loc['min', i] = min(dfA[i].min(),
                                           min(dfB[i].min(), min(dfC[i].min(), dfD[i].min()), dfE[i].min()),
                                           dfF[i].min(), dfG[i].min() + dfH[i].min(), dfI[i].min())
    return df_Max_Min

def Normalization(qos, df_Max_Min, QoSAttribute):
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

def Fitness(population, tplAll, QoSAttribute):
    # 计算种群的适应度
    qos=[]
    for i in population:
        qos.append(Computing(i, QoSAttribute))
    qos=np.array(qos)
    dfMaxMin=get_Min_Max(tplAll, QoSAttribute)
    dfNor=Normalization(qos,dfMaxMin,QoSAttribute)
    return dfNor.values.sum()

def FuzzingMatch(population, tplAll,parameter):
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
            # s4 = population[i][j][3]
            # s5 = population[i][j][4]

            for k in range(len(tplAll[j].values)):
                # 欧氏距离中替换个体的参数
                sj1 = candidates_j[k][0]
                sj2 = candidates_j[k][1]
                sj3 = candidates_j[k][2]
                # sj4 = candidates_j[k][3]
                # sj5 = candidates_j[k][4]
                E_distance = abs(s1 - sj1) + abs(s2 - sj2) + abs(s3 - sj3) #+ abs(s4 - sj4) #+ abs(s5-sj5)
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

def GA( tplAll, QoSAttribute):
    # 定义参数
    parameter = {'crossover_probability': 0.88, 'mutation_probability': 0.1, 'task_number': 9, 'population_size': 100, 'iteration_number': 100}
    # 初始化种群
    population = init(tplAll,parameter)
    list=[]
    # 添加时间戳
    s1=time.perf_counter()
    count=0
    for i in range(100): # 迭代次数
        # 记录刚开始种群的优化性能
        b = Fitness(population, tplAll, QoSAttribute) / len(QoSAttribute)
        # 选择
        population_Select = Selection(population, parameter, QoSAttribute)
        # 交叉
        population_Crossover = Crossover(population_Select, parameter)
        # 突变
        population_Mutation = Mutation(population_Crossover, parameter, tplAll, QoSAttribute)
        # 从真实数据集中选择与种群中数据最相似的来代替这个种群
        population_Real = FuzzingMatch(population_Mutation, tplAll, parameter)
        list.append(Fitness(population_Real, tplAll, QoSAttribute) / len(QoSAttribute))
        # 记录迭代一次后种群的优化性能
        a=Fitness(population_Real, tplAll, QoSAttribute)/ len(QoSAttribute)

        if a>b:
            population=population_Real

    max=Fitness(population_Real, tplAll, QoSAttribute) / len(QoSAttribute)
    s2=time.perf_counter()
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
        qos = eval(input())
        QoSAttribute.append(dic[qos])
    tplAll = data(QoSAttribute)
    GA(tplAll, QoSAttribute)