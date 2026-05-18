import networkx as nx
from IC_model import IC
from inverse_sigmod import run_torch_version
from MCEM import inference as em_inference
import numpy as np
from utils import calculate_MI, modified_kmeans, result_record
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

def get_size_factor(comm_id):
    community_counts = Counter(node_communities.values())
    size = community_counts[comm_id]
    # 使用 log 缩放可以防止参数爆炸，+1 是为了防止 log(1)=0
    return np.log1p(size)

def generate_infections(A, num_sim = 100):

    N = A.shape[0]
    S = np.zeros([num_sim, N])
    nx_graph = nx.from_numpy_array(A)
    trees = []
    while len(trees) < num_sim:
        seed = np.random.choice(np.arange(0, N), size=1)
        cascade, tree = IC(Networkx_Graph=nx_graph, Seed_Set=seed, Probability=A)
        if len(tree.nodes) >= 3:
            S[len(trees), cascade] = 1
            trees.append(tree)
    average_paths = 0
    for tree in trees:
        average_paths += len(tree.nodes())

    print("average length of infections: ", average_paths / len(trees))
    return S

if __name__ == '__main__':
    # --- 设定参数 ---
    edges = set()
    data_path = '../dataset/mastodon/'
    with open(data_path+'edges.txt', 'r') as f:
        for l in f:
            if l.strip() != '':
                edges.add((l.strip().split(' ')[0], l.strip().split(' ')[1]))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    N = len(G)
    node_communities = {}
    
    with open(data_path + 'instance.txt', 'r') as f:
        for l in f:
            if l.strip() != '':
                # if len(l.strip().split('\t')) < 2:
                #     print(l)
                node_communities[l.strip().split()[0]] = l.strip().split()[1]
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
    
    mi_matrix, p_matrix = calculate_MI(S.T)
    cluster, fixed_cluster = modified_kmeans(mi_matrix)
    threshold = max(fixed_cluster.values())
    prune_network = np.zeros([N, N])
    prune_network[mi_matrix > threshold] = 1.0
    prune_network[mi_matrix <= threshold] = 0.0

    #-------------------MCEM---------------------------
    # em_inference(S, A, sample_size = 10, prune_network = prune_network, iterations = 400)

    # -------------------inverse sigmod--------------------
    result_record("DNIT", run_torch_version(A, S, iterations=10000, prune_network=prune_network), "mastodon", param='auc')