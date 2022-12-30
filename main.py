# 这是一个示例 Python 脚本。
import numpy as np
import networkx as nx
from scipy import sparse as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import random

kinetics = [[0, 0.01, 0], [0.01, 0, 0.01], [0, 0.01, 0]]                      # 速率因子
mol = np.array([5*(10**4), 5*(10**4)])                                        # 分子数
group_type = np.array([[1, 1, 1], [2, 0, 2]])                            # 每个分子的官能数
interval = 10                                                         # 循环间隔
cycles = 24*(10**8)                                                     # 循环次数
column = ['cycle', 'P', 'Mn', 'Mw', 'P_theory', 'PMw', 'Wmax', 'W_sol', 'sol_theory']
k= 0.01


N_mol = sum(mol)
N_branch = range(len(group_type[0]))
network = sp.dok_matrix((N_mol, N_mol))
group = np.repeat(group_type, mol, axis=0)
sum_gro = np.array([np.sum(group == i) for i in range(1, np.max(group)+1)])
data = pd.DataFrame(columns = column)
data_size = pd.DataFrame()
data_cycle = pd.DataFrame()
data_degree = pd.DataFrame()
mol_list = range(N_mol)
r = sum_gro[1] / sum_gro[0]
Pr = 0.967875
P_gel = (3**(1/2))/2

def react():
    if kinetics[0][1] > random.random():
        mol1, mol2 = random.choices(mol_list, k = 2)                                    # 抽两个分子
        rct1 = random.choice(range(len(group[mol1])))
        rct2 = random.choice(range(len(group[mol2])))
        gro1, gro2 = group[mol1][rct1], group[mol2][rct2]
        if mol1 != mol2 and gro1 > 0 and gro2 > 0 and gro1 != gro2 and  network[mol1, mol2] != 1 :  # 判断反应是否进行  and gro1 != gro2
            network[mol1, mol2], network[mol2, mol1] = 1, 1
            group[mol1][rct1], group[mol2][rct2] = -group[mol1][rct1], -group[mol2][rct2]


def cal(cycle):
    graph = nx.from_scipy_sparse_array(network)                                              # 构造nx图
    graph_component = nx.connected_components(graph)                                        # 获得graph_node所有子图
    tree_size = np.sort(np.array([graph.subgraph(i).number_of_nodes() for i in graph_component]))   # n_node储存了所有子图的节点数目
    # graph_component = nx.connected_components(graph)                                        # 生成器生成后需要重新调用，重新获得graph_node所有子图
    # list_tree = ([list(i) for i in graph_component ])                               # 获得子图的节点并转化为list形式
    # adistribute = pd.DataFrame([np.sum(np.array(i) >= mol[0]) for i in list_tree])  # 对a官能度进行统计
    # bdistribute = pd.DataFrame([np.sum(np.array(i) < mol[0]) for i in list_tree])  # 对b官能度进行统计
    # adistribute.append((N_mol - len(tree_size)) * [None]) #补齐dataframe长度
    # bdistribute.append((N_mol - len(tree_size)) * [None]) #补齐dataframe长度
    P = max((sum_gro - np.array([np.sum(group == i) for i in range(1, np.max(group) + 1)])) / sum_gro)  #反应程度                                             # 反应程度
    P_theory = (1 / 2) * k * cycle      #理论反应程度
    Mw = sum( np.array( tree_size, dtype = 'int64')** 2 ) / N_mol   # 重均分子量
    Wmax = tree_size[-1] ** 2 / N_mol      #最大的团簇质量分数
    PMw = sum(np.array( tree_size[:-1], dtype = 'int64') ** 2) / N_mol     #减去最大的团簇后的质量分数
    Mn = N_mol / len(tree_size)  # 数均分子量
    index = int(cycle)  # 数据的列数
    W_sol = 1
    sol_theory = 1
    if P > P_gel:
        loop = np.array([len(i) for i in nx.cycle_basis(graph)])    #loop为图中所有的环的大小
        loop = np.append(loop,(N_mol - len(loop)) * [None])  #补齐长度
        graph_component = nx.connected_components(graph)              #生成器生成graph的子图
        tree = np.array([len(graph.subgraph(i).nodes) for i in graph_component])   #遍历子图的节点数并保存
        tree = np.append(tree ,(N_mol - len(tree)) * [None])   #补齐长度
        graph_component = nx.connected_components(graph)     #生成器生成graph的子
        link = []
        for i in graph_component:
            sub_degree = graph.subgraph(i).degree()
            link.append(len([j[1] for j in sub_degree if j[1] > 2]))     #link为每个支化单元中节点度大于三的节点数目
        tree_degree = np.array(link)
        tree_degree = np.append(tree_degree, (N_mol - len(tree_degree)) * [None])
        # tree_degree.columns, tree.columns, loop.columns = str(P), str(P), str(P)
        W_sol = 1 - Wmax / N_mol
        # data_size[str(cycle)] = np.append(tree_size, (N_mol - len(tree_size)) * [None])
        sol_theory =  (1/2) *((( 1- r * (P ** 2) * (Pr ** 2)) / ( r * (P**2) * (Pr ** 2))) ** 3) + (1/2)*(((1-(2 * r) * (P ** 2)  * (Pr ** 2) + (r ** 2) * (P **3) * (Pr ** 3)) / ((r ** 2) * (P**3)  * (Pr ** 3))) ** 2)
        data_degree[str(cycle)] = tree_degree
        data_degree[str(P)] = tree
        data_cycle[str(P)] = loop
    data.loc[index, column] = [cycle, P, Mn, Mw, P_theory, PMw, Wmax, W_sol, sol_theory]
    # data_size[str(cycle)] = adistribute
    # data_size[str(P)] = bdistribute



def draw():
    plt.subplot(1, 1, 1)
    y = np.array(data.loc[:, 'Mw']).T
    y2 = np.array(data.loc[:, 'P']).T
    y3 = np.array(data.loc[:, 'P_theory']).T
    y4 = np.array(data.loc[:, 'Mw']).T / np.array(data.loc[:, 'Mn'])
    x = np.array(data.loc[:, 'cycle'])
    plt.subplot(1, 1, 1)
    plt.plot(x, y2, label='Mn_theory')
    # plt.legend()
    # plt.subplot(2, 2, 3)
    # plt.xlim(0, x[-1] + 0.1)
    # plt.ylim(0, y4[-1] + 1)
    # plt.plot(x, y4, label='D')
    # plt.legend()
    # plt.subplot(2, 2, 4)
    # plt.hist(node, bins=20, histtype='stepfilled', label='frequency')
    # plt.legend()
    plt.show()


for i in tqdm(range(cycles)):
    react()
    if (i/N_mol) % interval == 0:
        cal(i/N_mol)
# print(data_size)
# print(data)
# draw()
data_size.to_csv("./A2B3k={0}_size.csv".format(k), index=0)
data.to_csv("./A2B3k={0}.csv".format(k), index=0)
data_cycle.to_csv("./A2B3k={0}_cycle.csv".format(k), index=0)
data_degree.to_csv("./A2B3k={0}_degree.csv".format(k), index=0)
