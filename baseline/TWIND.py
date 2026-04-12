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
    """
    计算节点 vi 和 vj 感染状态之间的互信息 (Eq. 17)
    """
    Xi = S[:, i]
    Xj = S[:, j]
    
    mi = 0.0
    # 遍历所有可能的状态组合 (0,0), (0,1), (1,0), (1,1)
    for val_i in [0, 1]:
        for val_j in [0, 1]:
            p_i = np.mean(Xi == val_i)
            p_j = np.mean(Xj == val_j)
            p_ij = np.mean((Xi == val_i) & (Xj == val_j))
            
            if p_ij > 0:
                mi += p_ij * np.log2(p_ij / (p_i * p_j))
    return mi

def kmeans_fixed_zero(mi_values, max_iters=100):
    """
    修改版的 K-means 算法 (K=2)，其中一个聚类中心固定为 0
    用于计算剪枝阈值 tau
    """
    mi_array = np.array(list(mi_values.values()))
    if len(mi_array) == 0:
        return 0.0
        
    center_0 = 0.0 # 固定为0的中心
    center_1 = np.max(mi_array) # 初始非零中心设为最大值
    
    for _ in range(max_iters):
        # 分配簇：距离哪个中心更近
        dist_0 = np.abs(mi_array - center_0)
        dist_1 = np.abs(mi_array - center_1)
        
        cluster_0_mask = dist_0 <= dist_1
        cluster_1_mask = dist_0 > dist_1
        
        # 重新计算非零中心
        if np.any(cluster_1_mask):
            new_center_1 = np.mean(mi_array[cluster_1_mask])
        else:
            break
            
        if np.abs(new_center_1 - center_1) < 1e-5:
            break
        center_1 = new_center_1
        
    # 返回均值接近 0 的那个簇中的最大 MI 值作为阈值 tau
    return np.max(mi_array[cluster_0_mask]) if np.any(cluster_0_mask) else 0.0

def calculate_score(S, num_cascades, i, W):
    """
    评分函数 g(vi, Fi) (Eq. 9)
    """
    score = 0.0
    Xi = S[:, i]
    
    # 获取候选父节点状态的所有组合实例
    if len(W) == 0:
        return score
        
    X_W = S[:, list(W)]
    
    # 统计 N_ij1 (未感染) 和 N_ij2 (已感染)
    counts = {}
    for l in range(num_cascades):
        parent_state = tuple(X_W[l])
        node_state = Xi[l]
        
        if parent_state not in counts:
            counts[parent_state] = {0: 0, 1: 0}
        counts[parent_state][node_state] += 1
        
    # 处理阶乘，使用对数伽马函数 (ln(N!) = gammaln(N+1))
    log2_e = np.log2(np.e)
    
    for count_dict in counts.values():
        N_ij1 = count_dict[0]
        N_ij2 = count_dict[1]
        
        log_N_ij1_fact = gammaln(N_ij1 + 1) * log2_e
        log_N_ij2_fact = gammaln(N_ij2 + 1) * log2_e
        log_sum_fact = gammaln(N_ij1 + N_ij2 + 1 + 1) * log2_e
        
        score += (log_N_ij1_fact + log_N_ij2_fact - log_sum_fact)
        
    return score

def run_twind_inference(S, num_nodes, num_cascades):
    """
    执行 TWIND 算法的主函数流程
    :param S: 形状为 (num_cascades, num_nodes) 的 numpy array
    :param num_nodes: 节点总数
    :param num_cascades: 观测到的扩散过程总数
    :return: 推断出的有向边集合 E (父节点 -> 子节点)
    """
    E = set()
    
    # 1. 计算互信息
    print("Calculating Mutual Information...")
    mi_values = {}
    for i in tqdm(range(num_nodes)):
        for j in range(num_nodes):
            if i != j:
                mi_values[(i, j)] = calculate_mi(S, i, j)
                
    # 2. 计算剪枝阈值 tau
    print("Pruning candidate parents...")
    tau = kmeans_fixed_zero(mi_values)
    
    # 3. 计算父节点数量上限 eta
    inner_log = np.log2(np.e * (num_cascades + 1) / 2)
    eta = math.ceil(np.log2((num_cascades + 1) * inner_log))
    print(f"Parent node limit (eta) calculated as: {eta}")
    
    # 4. 贪心搜索最优父节点集
    print("Greedy searching for optimal parent sets...")
    for i in tqdm(range(num_nodes)):
        F_i = set()
        P_i = set()
        
        # 筛选出互信息大于 tau 的候选父节点
        for j in range(num_nodes):
            if i != j and mi_values[(i, j)] > tau:
                P_i.add(j)
                
        C_i = []
        
        # 生成候选子集并打分
        for r in range(1, min(len(P_i), eta) + 1):
            for w_tuple in itertools.combinations(P_i, r):
                W = set(w_tuple)
                score = calculate_score(S, num_cascades, i, W)
                C_i.append({'W': W, 'score': score})
        
        # 贪心选择
        while C_i:
            best_item = max(C_i, key=lambda x: x['score'])
            W_star = best_item['W']
            
            if len(F_i.union(W_star)) <= eta:
                F_i = F_i.union(W_star)
            
            C_i.remove(best_item)
            
        # 记录发现的有向边
        for parent in F_i:
            E.add((parent, i))
            
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
    IG.add_edges_from(run_twind_inference(S, N, 100))
    
    result_record("TWIND", calculate_F1(IG, G), "LFR", f"n{N}")