import numpy as np
from scipy.special import gammaln
import itertools
import math
from itertools import combinations
from sklearn.cluster import KMeans
import math
import networkx as nx
from tqdm import tqdm
from utils import generate_infections, result_record, calculate_F1, IC, Neighbour_finder

def calculate_mi(S, i, j):
    """计算互信息，用于剪枝"""
    Xi = S[:, i]
    Xj = S[:, j]
    mi = 0.0
    for val_i in [0, 1]:
        p_i = np.mean(Xi == val_i)
        if p_i == 0: continue
        for val_j in [0, 1]:
            p_j = np.mean(Xj == val_j)
            p_ij = np.mean((Xi == val_i) & (Xj == val_j))
            if p_ij > 0 and p_j > 0:
                mi += p_ij * np.log2(p_ij / (p_i * p_j))
    return mi

def get_pruning_threshold(mi_dict):
    """使用固定中心为0的K-means计算阈值tau"""
    mis = np.array(list(mi_dict.values()))
    if len(mis) == 0: return 0.0
    
    c0 = 0.0
    c1 = np.max(mis)
    for _ in range(50): # 快速迭代
        d0 = np.abs(mis - c0)
        d1 = np.abs(mis - c1)
        cluster1 = mis[d1 < d0]
        if len(cluster1) == 0: break
        new_c1 = np.mean(cluster1)
        if abs(new_c1 - c1) < 1e-6: break
        c1 = new_c1
    
    # 属于 c0 簇的最大值作为阈值
    return np.max(mis[np.abs(mis - c0) <= np.abs(mis - c1)])

def fast_score(S, i, F_i_list):
    """
    优化的评分函数：利用状态压缩和快速计数
    """
    if not F_i_list: return -float('inf')
    
    beta = S.shape[0]
    Xi = S[:, i]
    # 将父节点集合的状态压缩为元组进行统计
    parents_data = S[:, F_i_list]
    
    # 统计 N_ij1 和 N_ij2
    counts = {}
    for l in range(beta):
        # 转换为 tuple 以便作为字典键
        p_state = tuple(parents_data[l])
        if p_state not in counts:
            counts[p_state] = [0, 0]
        counts[p_state][int(Xi[l])] += 1
    
    log2_e = np.log2(np.e)
    total_score = 0.0
    for n1, n2 in counts.values():
        # g(v_i, F_i) = log(n1!) + log(n2!) - log((n1+n2+1)!)
        total_score += (gammaln(n1 + 1) + gammaln(n2 + 1) - gammaln(n1 + n2 + 2)) * log2_e
    return total_score

def run_twind_fast(S):
    beta, n = S.shape
    E = set()

    # 1. 计算所有配对的 MI 并剪枝
    print("Step 1: Pruning with Mutual Information...")
    all_mi = {}
    for i in tqdm(range(n)):
        for j in range(n):
            if i != j:
                all_mi[(i, j)] = calculate_mi(S, i, j)
    
    tau = get_pruning_threshold(all_mi)
    
    # 2. 计算理论上限 eta
    log_part = np.log2(np.e * (beta + 1) / 2)
    eta = math.ceil(np.log2((beta + 1) * log_part))
    print(f"Step 2: Max parents allowed (eta) = {eta}")

    # 3. 启发式贪心搜索
    print("Step 3: Heuristic Greedy Search...")
    for i in tqdm(range(n)):
        # 获取候选父节点集合 Pi
        P_i = [j for j in range(n) if i != j and all_mi[(i, j)] > tau]
        
        F_i = [] # 当前选定的父节点集
        current_best_score = -float('inf')
        
        # 贪心逐个添加节点，直到达到 eta 或分数不再提升
        for _ in range(eta):
            best_node_to_add = None
            
            for candidate in P_i:
                if candidate in F_i: continue
                
                # 尝试将 candidate 加入 F_i 看分数变化
                test_F = F_i + [candidate]
                score = fast_score(S, i, test_F)
                
                if score > current_best_score:
                    current_best_score = score
                    best_node_to_add = candidate
            
            if best_node_to_add is not None:
                F_i.append(best_node_to_add)
            else:
                break # 找不到更好的了，提前结束
        
        for p in F_i:
            E.add((p, i))
            
    return E

# ==========================================
# 示例用法
# ==========================================
if __name__ == '__main__':
    # --- 设定参数 ---
    np.random.seed(2023)
    N = 1000       # -N 1000-3000
    AVG_K = 15     # -k 15 (average_degree)
    MAX_K = 50     # -maxk 50 (max_degree)
    MU = 0.1       # -mu 0.1 (mu)
    MIN_C = 20     # -minc 20 (min_community)
    MAX_C = 50     # -maxc 50 (max_community)

    # 必须指定的幂律指数 (使用常用值)
    TAU1 = 2.0     # 度分布幂律指数
    TAU2 = 2.0     # 社区规模幂律指数

    # 由于参数约束较严格，我们增加最大迭代次数以防 ExceededMaxIterations 错误
    MAX_I = 100000

    # --- 生成 LFR Benchmark 图 ---
    try:
        G = nx.generators.community.LFR_benchmark_graph(
            n=N, 
            tau1=TAU1, 
            tau2=TAU2, 
            mu=MU, 
            average_degree=AVG_K, 
            max_degree=MAX_K,        # 指定最大度
            min_community=MIN_C, 
            max_community=MAX_C,     # 指定最大社区规模
            max_iters=MAX_I,         # 增加迭代次数
            seed=42
        )

        print(f"✅ LFR网络生成成功！")
        print(f"生成的LFR网络节点数: {G.number_of_nodes()}")
        print(f"生成的LFR网络边数: {G.number_of_edges()}")

        # 获取地面真值社区
        communities = {frozenset(G.nodes[v]['community']) for v in G}
        print(f"真实的社区数量: {len(communities)}")
        
    except nx.ExceededMaxIterations as e:
        print(f"❌ 生成失败: {e}")
        print("请尝试进一步调整参数（例如增加MAX_I或略微放宽社区规模约束）。")
    
    node_communities = {n: G.nodes[n]['community'] for n in G.nodes}

    A = nx.to_numpy_array(G)
    P = np.zeros((N, N))
    for u, v in G.edges():
        if node_communities[u] & node_communities[v]:
            weight = np.random.uniform(0.05, 0.1)
        else:
            weight = np.random.uniform(0.01, 0.05)
        P[u, v] = weight
        P[v, u] = weight
    A = A * P
    S = generate_infections(A, num_sim=100)
    
    IG = nx.DiGraph()
    IG.add_edges_from(run_twind_fast(S))
    
    result_record("TWIND", calculate_F1(IG, G), "LFR", f"n{N}")