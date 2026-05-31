import networkx as nx
import numpy as np
import itertools
from utils import result_record, calculate_F1, calculate_binary_auc
from collections import Counter
import time

def get_size_factor(comm_id):
    community_counts = Counter(node_communities.values())
    size = community_counts[comm_id]
    # 使用 log 缩放可以防止参数爆炸，+1 是为了防止 log(1)=0
    return np.log1p(size)


# --- 1. 适配函数：模拟扩散过程 (IC模型) ---
def generate_infections(A_weighted, num_sim=100, seed_ratio=0.01):
    """
    根据权重矩阵 A 模拟扩散，生成观测数据。
    返回 observations: list of (seed_set, final_active_set)
    """
    N = A_weighted.shape[0]
    observations = []
    
    for _ in range(num_sim):
        # 随机选择种子节点 (Passive Seeds 场景)
        num_seeds = max(1, int(N * seed_ratio))
        seeds = set(np.random.choice(N, num_seeds, replace=False))
        
        active = set(seeds)
        newly_active = list(seeds)
        
        # 模拟扩散直到没有新感染
        while newly_active:
            next_round = []
            for u in newly_active:
                # 找到 u 的邻居（权重 > 0 的点）
                neighbors = np.where(A_weighted[u] > 0)[0]
                for v in neighbors:
                    if v not in active:
                        # 按照 A[u,v] 的概率尝试感染
                        if np.random.random() < A_weighted[u, v]:
                            active.add(v)
                            next_round.append(v)
            newly_active = next_round
        observations.append((seeds, active))
    return observations

# --- 2. 适配函数：计算提升度 (Lift) ---
def estimate_lifts(vertices, observations):
    M = len(observations)
    n_A = {v: 0 for v in vertices}
    n_S = {u: 0 for u in vertices}
    n_SA = {u: {v: 0 for v in vertices} for u in vertices}
    
    for S_i, A_i in observations:
        for v in A_i: n_A[v] += 1
        for u in S_i:
            n_S[u] += 1
            for v in A_i: n_SA[u][v] += 1
                
    lift_estimates = {}
    for u in vertices:
        for v in vertices:
            if u != v and n_S[u] > 0:
                # 论文公式: P(v|u) - P(v)
                lift_estimates[(u, v)] = (n_SA[u][v] / n_S[u]) - (n_A[v] / M)
    return lift_estimates

# --- 3. 核心算法：K-Lifts ---
def k_lifts_algorithm(vertices, lift_estimates, K):
    E_hat = set()
    candidate_edges = []
    # 遍历所有可能的节点对
    for u, v in itertools.combinations(vertices, 2):
        l_uv = lift_estimates.get((u, v), -1e9)
        l_vu = lift_estimates.get((v, u), -1e9)
        # 取双向提升度的最大值作为无向边判定依据
        candidate_edges.append((max(l_uv, l_vu), (u, v)))
    
    # 贪心选择前 K 个最大 Lift 的边
    candidate_edges.sort(key=lambda x: x[0], reverse=True)
    for _, edge in candidate_edges[:K]:
        E_hat.add(tuple(sorted(edge)))
    return E_hat

if __name__ == '__main__':
    # ==========================================
    # 这里插入你给出的原始代码 (保持不动)
    # ==========================================
    for n_sim in [50,100,150,200,250]:
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

        S = generate_infections(A, num_sim=n_sim) 

        # ==========================================
        # 剩下的推断与评估部分
        # ==========================================

        # 执行 K-Lifts 推断
        print("正在计算 Lift 指标并推断网络结构...")
        vertices = list(range(N))
        start_time = time.time()
        lifts = estimate_lifts(vertices, S)
        K_target = G.number_of_edges() # 设定推断边数等于真实边数
        predicted_edges = k_lifts_algorithm(vertices, lifts, K_target)
        end_time = time.time()
        nodes = list(G.nodes())
        IG = nx.DiGraph()
        IG.add_nodes_from(nodes)
        IG.add_edges_from(predicted_edges)

        result_record("klifts", calculate_binary_auc(IG, G), "workplace", f"p{n_sim}auc", 'process.jsonl')
        result_record("klifts", end_time - start_time, "workplace", f"p{n_sim}", 'time.jsonl')