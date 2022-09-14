# 基本配置
## 配备AMD锐龙5、CPU 2.38GHZ和16G内存的PC，运行在Windows 10 x64和Python 3.9上
# 解释说明
## GA、PSO、TLBO是使用原始的候选集作为数据的输入，而这些算法+Skyline则是挑出候选服务集中的Skyline服务作为最新的候选集
## SQMSC-MH全部使用过程都是找寻Skyline服务
## 以上的代码输入数据均是使用了qws2resetIndex数据集中的数据，qws2resetIndex数据集与真实数据集QWS2大致相同，不同的是我们打乱了他的行索引
# 参数
### GA、GA+Skyline：candidate_number 每个子任务的候选集中服务个数，crossover_probability 交叉概率，mutation_probability 突变概率，task_number 任务数，population_size 种群个体数
### PSO、PSO+Skylie：w 惯性权重，c1和c2 学习因子，Vmax 速度限制，task_number 任务数，population_size 粒子群个体数
### TLBO、TLBO+Skyline：task_number 任务数，population_size 学生群体数
### SQWSC：n QoS属性个数，candidate_number 每个子任务的候选集中服务个数，crossover_probability交叉概率，mutation_probability突变概率，task_number任务数，population_size种群个体数
