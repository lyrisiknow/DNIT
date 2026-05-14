import numpy as np
from scipy.special import gammaln
import itertools
import math
from itertools import combinations
from sklearn.cluster import KMeans
import math
import networkx as nx
from tqdm import tqdm
from utils import generate_infections, result_record, calculate_F1, IC, Neighbour_finder, calculate_binary_auc
from SIDN import infer_sidn_network
from TENDS import tends_algorithm
from TWIND import run_twind_fast
from k_lifts import estimate_lifts, k_lifts_algorithm
from PIND import pind_inference
from collections import Counter

def get_size_factor(comm_id):
    community_counts = Counter(node_communities.values())
    size = community_counts[comm_id]
    # 使用 log 缩放可以防止参数爆炸，+1 是为了防止 log(1)=0
    return np.log1p(size)

if __name__ == '__main__':
    # --- 设定参数 ---
    edges = set()
    data_path = '../dataset/contact_in_workplace/'
    with open(data_path+'tij_InVS.dat', 'r') as f:
        for l in f:
            if l.strip() != '':
                edges.add((int(l.strip().split(' ')[1]), int(l.strip().split(' ')[2])))
                edges.add((int(l.strip().split(' ')[2]), int(l.strip().split(' ')[1])))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    N = len(G)
    node_communities = {}
    
    with open(data_path + 'department.txt', 'r') as f:
        for l in f:
            if l.strip() != '':
                # if len(l.strip().split('\t')) < 2:
                #     print(l)
                node_communities[int(l.strip().split()[0])] = l.strip().split()[1]
    nodes_idx = {}
    for i, node in enumerate(node_communities):
        nodes_idx[node] = i

    A = nx.to_numpy_array(G)
    P = np.zeros((N, N))
    for u, v in G.edges():
        c_u = node_communities[u]
        c_v = node_communities[v]
        if c_u == c_v:
            weight = np.random.uniform(0.05, 0.1)
        else:
            # 社区之间：核心修改点
            # 基础跨社区概率
            base_inter_weight = np.random.uniform(0.005, 0.02)
            
            # 规模增强因子：大社区与大社区之间概率更高
            # 归一化因子（例如除以平均规模的 log 值）以保持权重在合理区间
            size_boost = get_size_factor(c_u) * get_size_factor(c_v)
            
            # 最终权重映射，确保不会超过 0.5（IC 模型通常不建议单边概率过高）
            weight = base_inter_weight * size_boost
        
        # 保证概率上限
        weight = min(weight, 0.4)
        ui = nodes_idx[u]
        vi = nodes_idx[v]
        P[ui, vi] = weight
        P[vi, ui] = weight
    A = A * P
    # 调用适配的生成函数
    # 建议 num_sim 设大一些（如 500+）以获得更准的 Lift 估计
    S = generate_infections(A, num_sim=100) 
    G = nx.from_numpy_array(A)
    nodes = list(G.nodes())
    print('TWIND:')
    IG = nx.DiGraph()
    IG.add_nodes_from(nodes)
    IG.add_edges_from(run_twind_fast(S))
    result_record("TWIND", calculate_binary_auc(IG, G), "workplace", param='auc')
    
    print('TENDS:')
    IG = nx.DiGraph()
    IG.add_nodes_from(nodes)
    IG.add_edges_from(tends_algorithm(N, S))
    result_record("TENDS", calculate_binary_auc(IG, G), "workplace", param='auc')
    
    print('SIDN:')
    IG = nx.DiGraph()
    IG = nx.from_numpy_array(infer_sidn_network(S), create_using=nx.DiGraph)
    result_record("SIDN", calculate_binary_auc(IG, G), "workplace", param='auc')
    
    print('PIND:')
    IG = nx.DiGraph()
    IG = nx.from_numpy_array(pind_inference(S), create_using=nx.DiGraph)
    result_record("PIND", calculate_binary_auc(IG, G), "workplace", param='auc')