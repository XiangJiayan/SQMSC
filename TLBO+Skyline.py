import random
import numpy as np
import pandas as pd
import copy
import time
import matplotlib.pyplot as plt

def data(QoSAttribute):
    path = 'qws2resetIndex.csv'
    data = pd.read_csv(path)
    data = data.reset_index(drop=True)
    for i in QoSAttribute:
        if i == 'Availability' or i == 'Successability' or i == 'Reliability' or i == 'Compliance' or i == 'Best Practices':
            data[[i]] = data[[i]]/100
    data = data[QoSAttribute]
    # 每个候选集的服务个数
    candidate_number = 40
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
    return dfA, dfB, dfC, dfD, dfE, dfF, dfG, dfH, dfI

def Pareto(pop_XRaw,parameter, QoSAttribute):
    # 找寻帕累托服务
    pop_X=copy.deepcopy(pop_XRaw)
    for i in range(len(pop_X)):
        for j in range(len(QoSAttribute)):
            if QoSAttribute[j]=='Availability' or QoSAttribute[j]=='Successability' or QoSAttribute[j]=='Reliability' or QoSAttribute[j]=='Throughput' or QoSAttribute[j]=='Compliance' or QoSAttribute[j]=='Best Practices':
                pop_X[i][j]=-pop_X[i][j]
    pareto = []
    for i in pop_X:
        dominated = True
        for j in pop_X:
            if all(i == j):
                continue
            if A_dominates_B(j, i):
                dominated = False
        if dominated:
            pareto.append(i)

    for par in pareto:
        for i in range(len(QoSAttribute)):
            if QoSAttribute[i]=='Availability' or QoSAttribute[i]=='Successability' or QoSAttribute[i]=='Reliability' or QoSAttribute[i]=='Throughput' or QoSAttribute[i]=='Compliance' or QoSAttribute[i]=='Best Practices':
                par[i]=-par[i]
    return pareto

def init(tplAll, parameter, QoSAttribute):
    # 初始化学生群体
    population = []
    tpl = []
    for i in range(parameter['task_number']):
        # 对每个子任务的候选集进行操作，选出Skyline服务
        tpl.append(Pareto(tplAll[i].values, parameter, QoSAttribute))
    for i in range(parameter['population_size']):
        temp = []
        for j in range(parameter['task_number']):
            # 为每个粒子群个体的每个子任务随机选择候选集中的数据
            k = random.randint(0, len(tpl[j]) - 1)
            temp.append(tpl[j][k])
        population.append(temp)
    return population,tpl

def QoS_Computing(QoS, QoSAttribute):
    # 计算复合服务的QoS值，其中传进来的QoS正属性均是负数
    a = []
    for i in range(len(QoSAttribute)):
        if QoSAttribute[i] == 'Response Time' or QoSAttribute[i] == 'Latency':
            a.append(QoS[0][i]+max(QoS[1][i], QoS[2][i]+QoS[3][i], QoS[4][i])+QoS[5][i]+max(QoS[6][i], QoS[7][i]) +QoS[8][i])
        elif QoSAttribute[i] == 'Availability' or QoSAttribute[i] == 'Successability' or QoSAttribute[i] == 'Reliability' or QoSAttribute[i] == 'Compliance' or QoSAttribute[i] == 'Best Practices':
            a.append(-abs(QoS[0][i]*max(QoS[1][i], QoS[2][i]*QoS[3][i],QoS[4][i])*QoS[5][i]*QoS[6][i]*QoS[7][i]*QoS[8][i]))
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

def ParetoSearch(pop_XRaw, parameter, QoSAttribute):
    # 与前面的找寻帕累托解不同,该方法是针对学生群体而言,而前面的只是找出候选集中的帕累托解
    pop_X = copy.deepcopy(pop_XRaw)
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k] == 'Availability' or QoSAttribute[k] == 'Successability' or QoSAttribute[k] == 'Reliability' or QoSAttribute[k] == 'Throughput' or QoSAttribute[k] == 'Compliance' or QoSAttribute[k] == 'Best Practices':
                    pop_X[i][j][k] = -pop_X[i][j][k]
    pareto = []
    for i in pop_X:
        dominated = True
        A = QoS_Computing(i, QoSAttribute)
        for j in pop_X:
            B = QoS_Computing(j, QoSAttribute)
            if all(A == B):
                continue
            if A_dominates_B(B, A):
                dominated = False
        if dominated:
            pareto.append(i)

    for par in pareto:
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k] == 'Availability' or QoSAttribute[k] == 'Successability' or QoSAttribute[k] == 'Reliability' or QoSAttribute[k] == 'Throughput' or QoSAttribute[k] == 'Compliance' or QoSAttribute[k] == 'Best Practices':
                    par[j][k] = -par[j][k]
    return pareto

def Computing(QoS, QoSAttribute):
    # 计算复合服务的QoS值
    a = []
    for i in range(len(QoSAttribute)):
        if QoSAttribute[i] == 'Response Time' or QoSAttribute[i] == 'Latency':
            a.append(QoS[0][i]+max(QoS[1][i], QoS[2][i]+QoS[3][i], QoS[4][i])+QoS[5][i]+max(QoS[6][i], QoS[7][i]) +QoS[8][i])
        elif QoSAttribute[i] == 'Availability' or QoSAttribute[i] == 'Successability' or QoSAttribute[i] == 'Reliability' or QoSAttribute[i] == 'Compliance' or QoSAttribute[i] == 'Best Practices':
            a.append(QoS[0][i]*min(QoS[1][i],QoS[2][i]*QoS[3][i],QoS[4][i])*QoS[5][i]*QoS[6][i]*QoS[7][i]*QoS[8][i])
        elif QoSAttribute[i] == 'Throughput':
            a.append(min(QoS[0][i],min(QoS[1][i],min(QoS[2][i],QoS[3][i]),QoS[4][i]),QoS[5][i],QoS[6][i]+QoS[7][i],QoS[8][i]))
    mat = np.array(a)
    return mat

def get_Min_Max(tplAll, QoSAttribute):
    # 得到每个候选服务集里每个QoS属性的最大值和最小值
    dfA = tplAll[0]
    dfB = tplAll[1]
    dfC = tplAll[2]
    dfD = tplAll[3]
    dfE = tplAll[4]
    dfF = tplAll[5]
    dfG = tplAll[6]
    dfH = tplAll[7]
    dfI = tplAll[8]
    df_Max_Min = pd.DataFrame(index=['max', 'min'], columns=QoSAttribute)
    # 根据任务选出理想的最好与最差组合，为归一化提供数据
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
    df=pd.DataFrame(qos, columns=QoSAttribute)
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

def fitness(pop, tplAll, QoSAttribute):
    # 计算粒子群中每个个体的优化性能
    qos = []
    qos.append(Computing(pop, QoSAttribute))
    qos = np.array(qos)
    dfMaxMin = get_Min_Max(tplAll, QoSAttribute)
    qos = np.reshape(qos, (1, 3))
    dfNor = Normalization(qos, dfMaxMin, QoSAttribute)
    return dfNor.values.sum()

def find_teacher(population, parameter, tplAll, QoSAttribute):
    teacher = []
    paretoSet = copy.deepcopy(ParetoSearch(population,parameter, QoSAttribute))
    # 若pareto解集里只有一个解
    if len(paretoSet) == 1:
        teacher = copy.deepcopy(paretoSet[0])
    # 若pareto解集里有多个解
    else:
        fit = 0
        for i in range(len(paretoSet)):
            if fit < fitness(paretoSet[i], tplAll, QoSAttribute):
                fit = fitness(paretoSet[i], tplAll, QoSAttribute)
                teacher = copy.deepcopy(paretoSet[i])
    return teacher

def get_Mean(population,parameter, QoSAttribute):
    """获得种群中 每个任务 的平均值;
       参数为种群;
       返回值为每个任务平均值的列表
    """
    Mean = []
    for i in range(parameter['task_number']):
        getmean=[]
        for j in range(len(QoSAttribute)):
            mean=0
            for k in range(parameter['population_size']):
                mean=mean+population[k][i][j]
            mean=mean/parameter['population_size']
            getmean.append(mean)
        Mean.append(getmean)
    return Mean

def update (old_populationRaw, new_populationRaw, parameter,QoSAttribute):
    """这个函数用来更新种群:若新解支配旧解，则替换;否则保留"""
    old_population=copy.deepcopy(old_populationRaw)
    new_population=copy.deepcopy(new_populationRaw)
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k]=='Availability' or QoSAttribute[k]=='Successability' or QoSAttribute[k]=='Reliability' or QoSAttribute[k]=='Throughput' or QoSAttribute[k]=='Compliance' or QoSAttribute[k]=='Best Practices':
                    old_population[i][j][k]=-old_population[i][j][k]
                    new_population[i][j][k]=-new_population[i][j][k]
    updated_group = []
    for i in range(parameter['population_size']):
        # 如果新解支配旧解
        A=QoS_Computing(new_population[i], QoSAttribute)
        B=QoS_Computing(old_population[i], QoSAttribute)
        if A_dominates_B (A, B):
            updated_group.append(new_population[i])
        else:
            updated_group.append(old_population[i])
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k]=='Availability' or QoSAttribute[k]=='Successability' or QoSAttribute[k]=='Reliability' or QoSAttribute[k]=='Throughput' or QoSAttribute[k]=='Compliance' or QoSAttribute[k]=='Best Practices':
                    updated_group[i][j][k]=-updated_group[i][j][k]
    return updated_group

def teacher_phase(populationRaw, teacher, parameter, QoSAttribute,tplAll):
    """教师阶段:所有个体通过老师和个体平均值的差值像老师;
    学习参数是 种群列表 和 候选服务集的上下界列表"""
    population=copy.deepcopy(populationRaw)
    Mean = get_Mean(population,parameter, QoSAttribute)  # 每个任务的平均值列表
    old_population = copy.deepcopy(population)  # 保存算法开始前的种群
    # 这个循环遍历每个个体
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            TF = round(1 + random.random())  # 教学因素 = round[1 + rand(0, 1)]
            r = random.random()  # ri=rand(0,1), 学习步长
            for k in range(len(QoSAttribute)):
                population[i][j][k] += r * (teacher[j][k] - TF * Mean[j][k])

    new_population = copy.deepcopy(FuzzingMatch(population, tplAll, parameter))

    # 在教师阶段方法内直接调用update方法
    new_population = copy.deepcopy(update(old_population, new_population,parameter,QoSAttribute))

    return new_population

def student_phase(populationRaw, parameter, QoSAttribute,tplAll):
    """学生阶段"""
    old_population = copy.deepcopy(populationRaw)  # 保存算法开始前的旧种群
    population=copy.deepcopy(populationRaw)
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k]=='Availability' or QoSAttribute[k]=='Successability' or QoSAttribute[k]=='Reliability' or QoSAttribute[k]=='Throughput' or QoSAttribute[k]=='Compliance' or QoSAttribute[k]=='Best Practices':
                    population[i][j][k]=-population[i][j][k]
    new_population = []  # 初始化新种群
    for i in range(parameter['population_size']):
        num_list = list(range(0, parameter['population_size']))
        num_list.remove(i)  # 这两步获得一个除了自身以外的随机索引
        X = copy.deepcopy(population[i])
        index = random.choice(num_list)
        Y = copy.deepcopy(population[index])  # 被选中与X交叉的个体
        # 如果X支配Y, X比Y好
        A=QoS_Computing(X, QoSAttribute)
        B=QoS_Computing(Y, QoSAttribute)

        if A_dominates_B (A, B):
            r = random.random()  # 学习步长ri=rand(0,1)
            for j in range(parameter['task_number']):
                for k in range(len(QoSAttribute)):
                    X[j][k]+=r*(X[j][k]-Y[j][k])

        elif A_dominates_B (B, A):
            r = random.random()  # 学习步长ri=rand(0,1)
            for j in range(parameter['task_number']):
                for k in range(len(QoSAttribute)):
                    X[j][k] += r * (Y[j][k] - X[j][k])

        # 若互相不支配，则两个目标函数分别学习
        else:
            # 若X的时间目标强于Y，成本目标弱于Y
            r = random.random()  # 学习步长ri=rand(0,1)
            for j in range(parameter['task_number']):
                for k in range(len(QoSAttribute)):
                    if A[k]>B[k]:
                        X[j][k] += r * (X[j][k] - Y[j][k])
                    elif A[k]<B[k]:
                        X[j][k] += r * (Y[j][k] - X[j][k])

        new_population.append(X)
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k]=='Availability' or QoSAttribute[k]=='Successability' or QoSAttribute[k]=='Reliability' or QoSAttribute[k]=='Throughput' or QoSAttribute[k]=='Compliance' or QoSAttribute[k]=='Best Practices':
                    new_population[i][j][k]=-new_population[i][j][k]
    new_population = copy.deepcopy(FuzzingMatch(new_population, tplAll, parameter))
    new_population = copy.deepcopy(update(old_population, new_population, parameter,QoSAttribute))
    return new_population

def Fitness(population_X, tplAll, QoSAttribute):
    # 计算粒子群的适应度
    qos = []
    for i in population_X:
        qos.append(Computing(i, QoSAttribute))
    qos = np.array(qos)
    dfMaxMin = get_Min_Max(tplAll, QoSAttribute)
    dfNor = Normalization(qos, dfMaxMin, QoSAttribute)
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
            difference = 100000000
            E_distance = 0  # 初始化欧氏距离列表
            # 第j个任务的候选服务集
            candidates_j = tplAll[j].values
            # 欧氏距离中被替换个体的参数
            s1=population[i][j][0]
            s2=population[i][j][1]
            s3=population[i][j][2]
            # s4 = population[i][j][3]
            # s5 = population[i][j][4]

            for k in range(len(tplAll[j].index)):
                # 欧氏距离中替换个体的参数
                sj1 = candidates_j[k][0]
                sj2 = candidates_j[k][1]
                sj3 = candidates_j[k][2]
                # sj4 = candidates_j[k][3]
                # sj5 = candidates_j[k][4]
                E_distance = abs(s1 - sj1) + abs(s2 - sj2) + abs(s3 - sj3) #+ abs(s4-sj4) + abs(s5-sj5)
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

def TLBO(tplAllRaw, QoSAttribute):
    parameter = {'task_number': 9, 'population_size': 100, 'iteration_number': 100}

    ini = init(tplAllRaw, parameter, QoSAttribute)
    population = ini[0]
    tplAllRaw = ini[1]

    tplAll=[]
    for i in range(parameter['task_number']):
        tplAll.append(pd.DataFrame(tplAllRaw[i],columns=QoSAttribute))
    list=[]
    for i in range(parameter['iteration_number']):
        teacher = find_teacher(population, parameter, tplAll, QoSAttribute)
        teaching_population = teacher_phase(population, teacher, parameter, QoSAttribute, tplAll)
        new_population = student_phase(teaching_population, parameter, QoSAttribute, tplAll)
        list.append(Fitness(new_population, tplAll, QoSAttribute)/len(QoSAttribute))
        population=new_population
    max=Fitness(new_population, tplAll, QoSAttribute)/len(QoSAttribute)

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
    for i in range(3):
        print('please input the number of the ', end="")
        print(i+1, end="")
        print('th QoSAttribute:', end="")
        qos=eval(input())
        QoSAttribute.append(dic[qos])
    tplAll = data(QoSAttribute)
    TLBO(tplAll, QoSAttribute)

