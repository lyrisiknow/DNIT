import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
import math
import networkx as nx
from tqdm import tqdm
from utils import generate_infections, result_record, calculate_F1

def infer_sidn_network(infection_data, lambda_val=None):
    """
    SIDN 算法：基于节点最终感染状态推断扩散网络结构
    
    参数:
    :param infection_data: 二维 numpy 数组, 形状为 (beta, n), beta 为观察次数, n 为节点数
                           1 表示感染, 0 表示未感染
    :param lambda_val: 复杂度惩罚系数, 若为 None 则自动设为 log2(beta)
    
    返回:
    :return inferred_adj: 推断出的邻接矩阵 (n x n)
    """
    beta, n = infection_data.shape
    inferred_adj = np.zeros((n, n))
    
    # 1. 修正惩罚因子 (论文逻辑: log2(beta) / beta)
    # 如果 beta 很大，log2(beta) 会让惩罚过重，除以 beta 是为了与平均熵量级匹配
    penalty_factor = lambda_val if lambda_val is not None else (math.log2(beta) / beta)
    
    # 【修复关键行】：定义 max_parents
    # 根据论文 Theorem 2，父节点集大小上限约等于 log(2 * beta / log(beta))
    if beta > 2:
        max_parents = int(math.log2(2 * beta / math.log2(beta)))
    else:
        max_parents = 1
    
    # 限制一个合理的物理上限（例如 10），防止计算爆炸
    max_parents = min(max_parents, 10) 

    for i in tqdm(range(n), desc="Inferring Nodes"):
        current_fi = []
        
        # 内部函数：计算 g 指标
        def calculate_g_score(target_idx, parents):
            if not parents:
                p1 = np.mean(infection_data[:, target_idx])
                p0 = 1 - p1
                h_cond = 0
                if p1 > 0: h_cond -= p1 * math.log2(p1)
                if p0 > 0: h_cond -= p0 * math.log2(p0)
            else:
                # 提取父节点数据并统计组合
                parents_data = infection_data[:, parents]
                # 使用二进制转十进制的方法快速分组（比 np.unique 快）
                weights = 2**np.arange(len(parents))
                combo_ids = parents_data.dot(weights).astype(int)
                
                counts = np.bincount(combo_ids, minlength=2**len(parents))
                h_cond = 0
                for combo_id, count in enumerate(counts):
                    if count == 0: continue
                    p_combo = count / beta
                    # 找到该组合对应的行
                    mask = (combo_ids == combo_id)
                    child_states = infection_data[mask, target_idx]
                    p1_given = np.mean(child_states)
                    p0_given = 1 - p1_given
                    
                    ent = 0
                    if p1_given > 0: ent -= p1_given * math.log2(p1_given)
                    if p0_given > 0: ent -= p0_given * math.log2(p0_given)
                    h_cond += p_combo * ent
            
            return 2 * h_cond + penalty_factor * (2 ** len(parents))

        best_g = calculate_g_score(i, current_fi)
        
        # 贪心搜索
        while len(current_fi) < max_parents:
            candidate_parents = [node for node in range(n) if node != i and node not in current_fi]
            best_cand = None
            
            for cand in candidate_parents:
                test_fi = current_fi + [cand]
                score = calculate_g_score(i, test_fi)
                if score < best_g:
                    best_g = score
                    best_cand = cand
            
            if best_cand is not None:
                current_fi.append(best_cand)
            else:
                break
        
        for p_idx in current_fi:
            inferred_adj[p_idx, i] = 1
            
    return inferred_adj

if __name__ == '__main__':
    # --- 设定参数 ---
    np.random.seed(2023)
    N = 1000       # -N 1000
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
    
    IG = nx.from_numpy_array(infer_sidn_network(S), create_using=nx.DiGraph)
    
    result_record("SIDN", calculate_F1(IG, G), "LFR", f"n{N}")
    