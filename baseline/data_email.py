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
import time

def get_size_factor(comm_id):
    community_counts = Counter(node_communities.values())
    size = community_counts[comm_id]
    # 使用 log 缩放可以防止参数爆炸，+1 是为了防止 log(1)=0
    return np.log1p(size)

if __name__ == '__main__':
    # --- 设定参数 ---
    edges = set()
    data_path = '../dataset/email-Eu-core/'
    with open(data_path+'email-Eu-core.txt', 'r') as f:
        for l in f:
            if l.strip() != '':
                edges.add((int(l.strip().split(' ')[0]), int(l.strip().split(' ')[1])))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    N = len(G)
    node_communities = {}
    
    with open(data_path + 'email-Eu-core-department-labels.txt', 'r') as f:
        for l in f:
            if l.strip() != '':
                node_communities[int(l.strip().split(' ')[0])] = l.strip().split(' ')[1]

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
        P[u, v] = weight
        P[v, u] = weight
    A = A * P
    # 调用适配的生成函数
    # 建议 num_sim 设大一些（如 500+）以获得更准的 Lift 估计
    S = generate_infections(A, num_sim=100) 
    G = nx.from_numpy_array(A)
    nodes = list(G.nodes())
    print('TWIND:')
    IG = nx.DiGraph()
    IG.add_nodes_from(nodes)
    start_time = time.time()  # 开始计时
    twind_edges = run_twind_fast(S)
    end_time = time.time()    # 结束计时
    twind_cost = end_time - start_time
    IG.add_edges_from(twind_edges)
    result_record("TWIND", calculate_binary_auc(IG, G), "mastodon", f"n{N}auc", 'resultn.jsonl')
    result_record("TWIND", twind_cost, "mastodon", f"n{N}", 'timen.jsonl')
    result_record("TWIND", calculate_F1(IG, G), "mastodon", f"n{N}f1", 'resultn.jsonl')
    
    print('TENDS:')
    IG = nx.DiGraph()
    IG.add_nodes_from(nodes)
    start_time = time.time()
    tends_edges = tends_algorithm(N, S)
    end_time = time.time()
    tends_cost = end_time - start_time
    IG.add_edges_from(tends_edges)
    result_record("TENDS", calculate_binary_auc(IG, G), "mastodon", f"n{N}auc", 'resultn.jsonl')
    result_record("TENDS", tends_cost, "mastodon", f"n{N}", 'timen.jsonl')
    result_record("TENDS", calculate_F1(IG, G), "mastodon", f"n{N}f1", 'resultn.jsonl')
    
    print('SIDN:')
    start_time = time.time()
    sidn_matrix = infer_sidn_network(S)
    end_time = time.time()
    sidn_cost = end_time - start_time

    IG = nx.from_numpy_array(sidn_matrix, create_using=nx.DiGraph)
    result_record("SIDN", calculate_binary_auc(IG, G), "mastodon", f"n{N}auc", 'resultn.jsonl')
    result_record("SIDN", sidn_cost, "mastodon", f"n{N}", 'timen.jsonl')

    result_record("SIDN", calculate_F1(IG, G), "mastodon", f"n{N}f1", 'resultn.jsonl')
    
    print('PIND:')
    start_time = time.time()
    pind_matrix = pind_inference(S)
    end_time = time.time()
    pind_cost = end_time - start_time

    IG = nx.from_numpy_array(pind_matrix, create_using=nx.DiGraph)
    result_record("PIND", calculate_binary_auc(IG, G), "mastodon", f"n{N}auc", 'resultn.jsonl')
    result_record("PIND", pind_cost, "mastodon", f"n{N}", 'timen.jsonl')
    result_record("PIND", calculate_F1(IG, G), "mastodon", f"n{N}f1", 'resultn.jsonl')