# Basic configuration
## All experiments are implemented on a PC with AMD Ryzen 5, CPU 2.38GHZ, and 16G RAM, running on Windows 10 x64 with Python 3.9
# Explanation
### GA, PSO and TLBO use the original candidate set as the data input, while these algorithms+Skyline select the Skyline service in the candidate service set as the latest candidate set
### The entire process of SQMSC-MH is to find Skyline services
### The input data of the above code are all data in the qws2resetIndex dataset. The qws2resetIndex dataset is roughly the same as the real dataset QWS2. The difference is that we have disturbed its row index
# Parameters
### GA、GA+Skyline：candidate_number 每个子任务的候选集中服务个数，crossover_probability 交叉概率，mutation_probability 突变概率，task_number 任务数，population_size 种群个体数
### PSO、PSO+Skylie：w 惯性权重，c1和c2 学习因子，Vmax 速度限制，task_number 任务数，population_size 粒子群个体数
### TLBO、TLBO+Skyline：task_number 任务数，population_size 学生群体数
### SQWSC：n QoS属性个数，candidate_number 每个子任务的候选集中服务个数，crossover_probability交叉概率，mutation_probability突变概率，task_number任务数，population_size种群个体数
