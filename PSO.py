import random
import time

import matplotlib .pyplot as plt
import copy
import numpy as np
import pandas as pd

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
    return dfA,dfB,dfC,dfD,dfE,dfF,dfG,dfH,dfI

def Bound(tplAll, QoSAttribute, parameter):
    boundup=[]
    bounddown=[]
    for i in range(parameter['task_number']):
        temp=[0, 0,0]
        temp[0] = tplAll[i][QoSAttribute[0]].max()
        temp[1] = tplAll[i][QoSAttribute[1]].max()
        temp[2] = tplAll[i][QoSAttribute[2]].max()
        # temp[3] = tplAll[i][QoSAttribute[3]].max()
        # temp[4] = tplAll[i][QoSAttribute[4]].max()
        boundup.append(temp)
    for i in range(parameter['task_number']):
        temp=[0, 0,0]
        temp[0] = tplAll[i][QoSAttribute[0]].min()
        temp[1] = tplAll[i][QoSAttribute[1]].min()
        temp[2] = tplAll[i][QoSAttribute[2]].min()
        # temp[3] = tplAll[i][QoSAttribute[3]].min()
        # temp[4] = tplAll[i][QoSAttribute[4]].min()
        bounddown.append(temp)
    return boundup,bounddown

def init_X(tplAll,parameter):
    population = []
    for i in range(parameter['population_size']):
        temp = []
        for j in range(parameter['task_number']):
            k = random.randint(0, len(tplAll[j].values) - 1)
            temp.append(tplAll[j].values[k])
        population.append(temp)
    return population

def init_V(parameter,Vmax):
    population = []
    for i in range(parameter['population_size']):
        i_task = []
        for j in range(parameter['task_number']):
            temp = [0, 0,0]
            temp[0] = random.uniform(-Vmax[j][0], Vmax[j][0])
            temp[1] = random.uniform(-Vmax[j][1], Vmax[j][1])
            temp[2] = random.uniform(-Vmax[j][2], Vmax[j][2])
            # temp[3] = random.uniform(-Vmax[j][3], Vmax[j][3])
            # temp[4] = random.uniform(-Vmax[j][4], Vmax[j][4])
            i_task.append(temp)
        population.append(i_task)
    return population

def get_Vmax(tplAll, parameter, QoSAttribute):
    Vmax = []  # 每个任务的速度上界
    # 速度的上界,由于下界是上界的值加上符号，所以不重复计算
    for i in range(parameter['task_number']):
        temp = [0, 0,0]
        for j in range(len(QoSAttribute)):
            if QoSAttribute[j] == 'Response Time' or QoSAttribute[j] == 'Latency' or QoSAttribute[j] == 'Throughput':
                temp[j] = (tplAll[i][QoSAttribute[j]].max()-tplAll[i][QoSAttribute[j]].min())*parameter['VMax']
            elif QoSAttribute[j] == 'Availability' or QoSAttribute[j] == 'Successability' or QoSAttribute[j] == 'Reliability' or QoSAttribute[j] == 'Compliance' or QoSAttribute[j] == 'Best Practices':
                temp[j] = parameter['VMax']
        Vmax.append(temp)
    return Vmax

def update_X(population_X, population_V, parameter, bound):
    new_pop_X=[]
    for i in range(parameter['population_size']):
        temp = []
        for j in range(0, parameter['task_number']):
            new_X = [0, 0,0]
            new_X[0] = population_X[i][j][0] + population_V[i][j][0]
            new_X[1] = population_X[i][j][1] + population_V[i][j][1]
            new_X[2] = population_X[i][j][2] + population_V[i][j][2]
            # new_X[3] = population_X[i][j][3] + population_V[i][j][3]
            # new_X[4] = population_X[i][j][4] + population_V[i][j][4]

            # 判断是否越上界
            if new_X[0] > bound[0][j][0]:
                new_X[0] = bound[0][j][0]
            if new_X[1] > bound[0][j][1]:
                new_X[1] = bound[0][j][1]
            if new_X[2] > bound[0][j][2]:
                new_X[2] = bound[0][j][2]
            # if new_X[3] > bound[0][j][3]:
            #     new_X[3] = bound[0][j][3]
            # if new_X[4] > bound[0][j][4]:
            #     new_X[4] = bound[0][j][4]

            # 判断是否越下界
            if new_X[0] < bound[1][j][0]:
                new_X[0] = bound[1][j][0]
            if new_X[1] < bound[1][j][1]:
                new_X[1] = bound[1][j][1]
            if new_X[2] < bound[1][j][2]:
                new_X[2] = bound[1][j][2]
            # if new_X[3] < bound[1][j][3]:
            #     new_X[3] = bound[1][j][3]
            # if new_X[4] < bound[1][j][4]:
            #     new_X[4] = bound[1][j][4]
            temp.append(new_X)
        new_pop_X.append(temp)
    return new_pop_X

def update_V(population_X, population_V, parameter, pbest, gbest, Vmax):
    """更新速度"""
    new_pop_V = []  # 种群更新后的速度
    for i in range(parameter['population_size']):
        temp = []
        speed = [0, 0,0]
        for j in range(parameter['task_number']):
            r1 = random.random()
            r2 = random.random()
            speed[0] = parameter['w'] * population_V[i][j][0] + parameter['c1'] * r1 * (pbest[i][j][0] - population_X[i][j][0]) + parameter['c2'] * r2 * (gbest[j][0] - population_X[i][j][0])
            speed[1] = parameter['w'] * population_V[i][j][1] + parameter['c1'] * r1 * (pbest[i][j][1] - population_X[i][j][1]) + parameter['c2'] * r2 * (gbest[j][1] - population_X[i][j][1])
            speed[2] = parameter['w'] * population_V[i][j][2] + parameter['c1'] * r1 * (pbest[i][j][2] - population_X[i][j][2]) + parameter['c2'] * r2 * (gbest[j][2] - population_X[i][j][2])
            #speed[3] = parameter['w'] * population_V[i][j][3] + parameter['c1'] * r1 * (
                        # pbest[i][j][3] - population_X[i][j][3]) + parameter['c2'] * r2 * (
                        #            gbest[j][3] - population_X[i][j][3])
            # speed[4] = parameter['w'] * population_V[i][j][4] + parameter['c1'] * r1 * (
            #         pbest[i][j][4] - population_X[i][j][4]) + parameter['c2'] * r2 * (
            #                    gbest[j][4] - population_X[i][j][4])
            # 判断是否越上界
            if speed[0] > Vmax[j][0]:
                speed[0] = Vmax[j][0]
            if speed[1] > Vmax[j][1]:
                speed[1] = Vmax[j][1]
            if speed[2] > Vmax[j][2]:
                speed[2] = Vmax[j][2]
            # if speed[3] > Vmax[j][3]:
            #     speed[3] = Vmax[j][3]
            # if speed[4] > Vmax[j][4]:
            #     speed[4] = Vmax[j][4]
            # 判断是否越下界
            if speed[0] < -Vmax[j][0]:
                speed[0] = -Vmax[j][0]
            if speed[1] < -Vmax[j][0]:
                speed[1] = -Vmax[j][0]
            if speed[2] < -Vmax[j][0]:
                speed[2] = -Vmax[j][0]
            # if speed[3] < -Vmax[j][3]:
            #     speed[3] = -Vmax[j][3]
            # if speed[4] < -Vmax[j][4]:
            #     speed[4] = -Vmax[j][4]
            temp.append(speed)
        new_pop_V.append(temp)
    return new_pop_V

def QoS_Computing(QoS, QoSAttribute):
    # 计算复合服务的QoS值，其中传进来的QoS正属性均是负数
    a=[]
    for i in range(len(QoSAttribute)):
        if QoSAttribute[i] == 'Response Time' or QoSAttribute[i] == 'Latency':
            a.append(QoS[0][i]+max(QoS[1][i], QoS[2][i]+QoS[3][i], QoS[4][i])+QoS[5][i]+max(QoS[6][i], QoS[7][i]) +QoS[8][i])
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

def save_pbest(pop_XRaw,parameter, pbestRaw, QoSAttribute):
    """更新个体历史最优"""
    pop_X=copy.deepcopy(pop_XRaw)
    pbest=copy.deepcopy(pbestRaw)
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k]=='Availability' or QoSAttribute[k]=='Successability' or QoSAttribute[k]=='Reliability' or QoSAttribute[k]=='Throughput' or QoSAttribute[k]=='Compliance' or QoSAttribute[k]=='Best Practices':
                    pop_X[i][j][k]=-pop_X[i][j][k]
                    pbest[i][j][k]=-pbest[i][j][k]
    updated_pbest = []
    for i in range(parameter['population_size']):
        # 如果新解支配旧解
        A = QoS_Computing(pop_X[i], QoSAttribute)
        B = QoS_Computing(pbest[i], QoSAttribute)
        if A_dominates_B(A, B):
            updated_pbest.append(pop_X[i])
        else:
            updated_pbest.append(pbest[i])
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k]=='Availability' or QoSAttribute[k]=='Successability' or QoSAttribute[k]=='Reliability' or QoSAttribute[k]=='Throughput' or QoSAttribute[k]=='Compliance' or QoSAttribute[k]=='Best Practices':
                    updated_pbest[i][j][k]=-updated_pbest[i][j][k]
    return updated_pbest

def ParetoSearch(pop_XRaw,parameter, QoSAttribute):
    # 搜寻粒子群中帕累托解
    pop_X=copy.deepcopy(pop_XRaw)
    for i in range(parameter['population_size']):
        for j in range(parameter['task_number']):
            for k in range(len(QoSAttribute)):
                if QoSAttribute[k]=='Availability' or QoSAttribute[k]=='Successability' or QoSAttribute[k]=='Reliability' or QoSAttribute[k]=='Throughput' or QoSAttribute[k]=='Compliance' or QoSAttribute[k]=='Best Practices':
                    pop_X[i][j][k]=-pop_X[i][j][k]
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
                if QoSAttribute[k]=='Availability' or QoSAttribute[k]=='Successability' or QoSAttribute[k]=='Reliability' or QoSAttribute[k]=='Throughput' or QoSAttribute[k]=='Compliance' or QoSAttribute[k]=='Best Practices':
                    par[j][k]=-par[j][k]
    return pareto

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
    # 计算粒子群中每个个体的适应度（可以这么说）
    qos = []
    qos.append(Computing(pop, QoSAttribute))
    qos = np.array(qos)
    dfMaxMin=get_Min_Max(tplAll, QoSAttribute)
    qos = np.reshape(qos, (1, 3))
    dfNor=Normalization(qos,dfMaxMin,QoSAttribute)
    return dfNor.values.sum()

def save_gbest(pop_X,parameter, gbest, QoSAttribute,tplAll):
    """更新种群历史最优"""
    pareto = ParetoSearch(pop_X,parameter, QoSAttribute)
    for i in range(len(pareto)):
        if fitness(pareto[i], tplAll, QoSAttribute)>fitness(gbest, tplAll, QoSAttribute):
            gbest = copy.deepcopy(pareto[i])
    return gbest

def update_group(old_group, new_group,tplAll, parameter, QoSAttribute):
    # 更新种群
    updated_group=[]
    for i in range(parameter['population_size']):
        # 如果新解支配旧解
        a=fitness(old_group[i],tplAll,QoSAttribute)
        b=fitness(new_group[i],tplAll,QoSAttribute)
        if b>a:
            updated_group.append(new_group[i])
        else:
            updated_group.append(old_group[i])
    return updated_group

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
                E_distance = abs(s1 - sj1) + abs(s2 - sj2) + abs(s3 - sj3) #+ abs(s4-sj4) #+ abs(s5-sj5)
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

def Fitness(population_X, tplAll, QoSAttribute, parameter):
    # 计算粒子群的适应度
    population_X=FuzzingMatch(population_X, tplAll,parameter)
    qos = []
    for i in population_X:
        qos.append(Computing(i, QoSAttribute))
    qos = np.array(qos)
    dfMaxMin = get_Min_Max(tplAll, QoSAttribute)
    dfNor = Normalization(qos, dfMaxMin, QoSAttribute)
    return dfNor.values.sum()

def PSO(tplAll, QoSAttribute):
    # 定义参数
    parameter = {'w': 0.9, 'c1': 2, 'c2': 2, 'task_number': 9, 'population_size': 100, 'VMax': 0.05, 'iteration_number': 100}
    list=[]
    # bound[0]是上界，bound[1]是下界
    bound = Bound(tplAll, QoSAttribute, parameter)
    # 初始化粒子群
    population_X = init_X(tplAll, parameter)
    # 初始化pbest
    pbest=population_X
    # 初始化gbest
    gbest=population_X[0]
    Vmax = get_Vmax(tplAll, parameter, QoSAttribute)
    # 初始化速度
    population_V = init_V(parameter, Vmax)
    # 添加时间戳
    s1=time.perf_counter()
    count=0
    for i in range(100):
        # 记录刚开始粒子群的优化性能
        b = Fitness(population_X, tplAll, QoSAttribute, parameter) / len(QoSAttribute)
        # 生成移动后的粒子群
        new_pop_X=update_X(population_X, population_V, parameter, bound)
        # 根据移动后的粒子群生成速度
        new_pop_V=update_V(population_X, population_V, parameter, pbest, gbest, Vmax)
        # 更新pbest 和gbest
        pbest=save_pbest(new_pop_X, parameter, pbest, QoSAttribute)
        gbest=save_gbest(new_pop_X, parameter, gbest, QoSAttribute, tplAll)
        # 根据移动前的粒子群和移动后的粒子群优化性能来更新粒子群
        population_X=update_group(population_X,new_pop_X,tplAll, parameter, QoSAttribute)
        # 记录迭代一次后粒子群的优化性能
        a=Fitness(population_X, tplAll, QoSAttribute, parameter) / len(QoSAttribute)
        population_V = new_pop_V
        if (b>a):
            max=b
            population_X = new_pop_X
        else:
            max=a
        if abs(a-b) < 0.3:
            count += 1
        else:
            count = 0
        if count >= 5:
            break

        list.append(a)
    # 添加事件戳
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
    for i in range(3):
        print('please input the number of the ', end="")
        print(i+1, end="")
        print('th QoSAttribute:', end="")
        qos=eval(input())
        QoSAttribute.append(dic[qos])

    tplAll = data(QoSAttribute)
    PSO(tplAll, QoSAttribute)