import numpy as np
from itertools import combinations
import math
import networkx as nx
from tqdm import tqdm
from utils import generate_infections, result_record, calculate_F1
# sklearn 的 mutual_info_score 已经被向量化实现取代，无需导入

def infer_sidn_network(infection_data, lambda_val=None):
    """
    SIDN 算法实现：基于最终感染状态推断扩散网络结构
    (优化版：通过向量化和矩阵乘法大幅提升计算速度)
    
    参数:
    :param infection_data: 二维 numpy 数组 (beta, n), 1 表示感染, 0 表示未感染 [cite: 786]
    :param lambda_val: 惩罚系数。若为 None，根据论文建议设为 log2(beta) 
    :return: inferred_adj (n x n) 邻接矩阵, A[i, j]=1 表示 i -> j 有向边 [cite: 780]
    """
    beta, n = infection_data.shape
    inferred_adj = np.zeros((n, n), dtype=np.int8)
    
    # 1. 设置惩罚因子 λ
    penalty_factor = lambda_val if lambda_val is not None else math.log2(beta)
    
    # 2. 计算搜索上限 max_parents
    max_parents = int(math.log2(2 * beta / math.log2(beta))) if beta > 2 else 1
    
    # ========================== 核心加速区 1 ==========================
    # 预计算所有节点的先验概率分布 P(X) 和 基础熵 H(X) 
    p1_all = infection_data.sum(axis=0) / beta
    p0_all = 1.0 - p1_all
    
    base_entropy = np.zeros(n)
    m1, m0 = p1_all > 0, p0_all > 0
    base_entropy[m1] -= p1_all[m1] * np.log2(p1_all[m1])
    base_entropy[m0] -= p0_all[m0] * np.log2(p0_all[m0])

    # 向量化计算全局互信息矩阵 MI(X_i, X_j) 
    # 利用矩阵乘法快速得出联合分布的概率 P(X_i=1, X_j=1)
    p11 = (infection_data.T @ infection_data) / beta
    p1_i = p1_all[:, None]
    p1_j = p1_all[None, :]
    p0_i = p0_all[:, None]
    p0_j = p0_all[None, :]
    
    p10 = p1_i - p11
    p01 = p1_j - p11
    p00 = p0_i - p01
    
    # 防止 log2(0) 造成的数值错误
    eps = 1e-15
    p11_c = np.clip(p11, eps, 1.0)
    p10_c = np.clip(p10, eps, 1.0)
    p01_c = np.clip(p01, eps, 1.0)
    p00_c = np.clip(p00, eps, 1.0)
    
    d11 = np.clip(p1_i * p1_j, eps, 1.0)
    d10 = np.clip(p1_i * p0_j, eps, 1.0)
    d01 = np.clip(p0_i * p1_j, eps, 1.0)
    d00 = np.clip(p0_i * p0_j, eps, 1.0)
    
    mi_matrix = (
        p11_c * np.log2(p11_c / d11) +
        p10_c * np.log2(p10_c / d10) +
        p01_c * np.log2(p01_c / d01) +
        p00_c * np.log2(p00_c / d00)
    )
    np.fill_diagonal(mi_matrix, 0)
    mi_matrix[mi_matrix < 1e-10] = 0  # 消除极小浮点误差
    # ==================================================================

    # 内部评分计算函数 (优化版)
    def calculate_g_score(target_idx, parents):
        if not parents:
            # O(1) 直接返回预先计算好的基础熵 
            return 2 * base_entropy[target_idx] + penalty_factor / beta
        
        # 使用位运算和带权 bincount 取代低效的 mask 切片操作 
        parents_data = infection_data[:, parents]
        weights = 1 << np.arange(len(parents))
        combo_ids = np.dot(parents_data, weights).astype(np.int64)
        
        counts = np.bincount(combo_ids)
        valid_mask = counts > 0
        
        counts_valid = counts[valid_mask]
        p_combo = counts_valid / beta
        
        target_data = infection_data[:, target_idx]
        sum_ones = np.bincount(combo_ids, weights=target_data)[valid_mask]
        
        p1_given = sum_ones / counts_valid
        p0_given = 1.0 - p1_given
        
        # 计算有效组合的条件熵，自动跳过0概率避免log(0)
        p1_safe = p1_given[p1_given > 0]
        p0_safe = p0_given[p0_given > 0]
        
        h_cond = -(np.sum(p_combo[p1_given > 0] * p1_safe * np.log2(p1_safe)) + 
                   np.sum(p_combo[p0_given > 0] * p0_safe * np.log2(p0_safe)))
        
        return 2 * h_cond + (penalty_factor * (1 << len(parents)) / beta)

    # 3. 遍历每个节点寻找其父节点集 [cite: 720]
    for i in tqdm(range(n), desc="SIDN 结构推断"):
        # 4. 预剪枝 (Pruning)：从预计算好的全局互信息矩阵直接拉取数据 [cite: 691]
        mi_scores = mi_matrix[i]
        valid_cands = np.where(mi_scores > 0)[0]
        valid_cands = valid_cands[valid_cands != i]
        
        # 按 MI 降序排列候选节点
        candidate_parents = valid_cands[np.argsort(-mi_scores[valid_cands])]
        
        # 5. 贪心搜索父节点集 Fi 
        current_fi = []
        best_g = calculate_g_score(i, current_fi)
        
        while len(current_fi) < max_parents:
            best_cand = None
            for cand in candidate_parents:
                if cand in current_fi: continue
                test_fi = current_fi + [cand]
                score = calculate_g_score(i, test_fi)
                
                if score < best_g:
                    best_g = score
                    best_cand = cand
            
            if best_cand is not None:
                current_fi.append(best_cand)
            else:
                break 
        
        # 6. 记录推断出的边 [cite: 780]
        for p_idx in current_fi:
            inferred_adj[p_idx, i] = 1
            
    return inferred_adj

if __name__ == '__main__':
    # --- 设定参数 ---
    np.random.seed(2023)
    for N in [100,150,200,250,300]:
    #N = 2000       # -N 1000
#N = 2000       # -N 1000-3000
        # AVG_K = 15     # -k 15 (average_degree)
        # MAX_K = 50     # -maxk 50 (max_degree)
        # MU = 0.1       # -mu 0.1 (mu)
        # MIN_C = 20     # -minc 20 (min_community)
        # MAX_C = 50     # -maxc 50 (max_community)
        
        AVG_K = 10     # 降低平均度
        MAX_K = 30     # 降低最大度
        MIN_C = 30     # 增加最小社区规模，确保能容纳度数较高的节点
        MAX_C = 60
        MU = 0.1       # -mu 0.1 (mu)

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
    