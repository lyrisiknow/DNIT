import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
import math
import networkx as nx
from tqdm import tqdm
from utils import generate_infections, result_record, calculate_F1, IC, Neighbour_finder

def calculate_probability(S, condition_dict):
    """
    辅助函数：计算在状态矩阵 S 中，满足 condition_dict 条件的概率 P。
    例如 condition_dict = {i: 1, j: 0} 表示计算节点 i 感染且节点 j 未感染的概率。
    
    参数:
    S: numpy.ndarray, 形状为 (beta, n)，其中 beta 为扩散过程的次数，n 为节点总数。
    condition_dict: dict, 键(key)为节点索引(int)，值(value)为期望的节点状态(0或1)。
    
    返回:
    float, 满足所有指定状态组合的概率(频率估算值)
    """
    # 扩散过程的总次数 (S 的行数)
    beta = S.shape[0]
    
    # 如果条件字典为空，代表没有任何限制条件，概率为 1.0
    if not condition_dict:
        return 1.0
        
    # 初始化一个长度为 beta 的全 True 布尔掩码 (mask)
    # 代表初始状态下，所有实验过程都处于候选状态
    mask = np.ones(beta, dtype=bool)
    
    # 遍历字典中的所有条件，逐步进行 AND (与) 操作缩小满足条件的范围
    for node_idx, target_state in condition_dict.items():
        # S[:, node_idx] 提取所有扩散过程中特定节点的状态
        # 比较结果是一个布尔数组，将其与当前的 mask 进行按位与(&)操作
        mask = mask & (S[:, node_idx] == target_state)
        
    # np.sum(mask) 会统计 mask 中 True 的个数，即满足所有条件的扩散过程次数
    match_count = np.sum(mask)
    
    # 返回概率（频率）
    return match_count / beta * 1.0

def calculate_mi_component(S, i, j, state_i, state_j):
    """
    计算互信息的组成部分 MI(X_i = state_i, X_j = state_j)
    对应论文公式 (24) 的单项
    """
    p_ij = calculate_probability(S, {i: state_i, j: state_j})
    p_i = calculate_probability(S, {i: state_i})
    p_j = calculate_probability(S, {j: state_j})
    
    if p_ij == 0 or p_i == 0 or p_j == 0:
        return 0
    return p_ij * math.log2(p_ij / (p_i * p_j))

def calculate_imi(S, i, j):
    """
    计算感染互信息 IMI(X_i, X_j)
    对应论文公式 (25)
    """
    mi_11 = calculate_mi_component(S, i, j, 1, 1)
    mi_00 = calculate_mi_component(S, i, j, 0, 0)
    mi_10 = calculate_mi_component(S, i, j, 1, 0)
    mi_01 = calculate_mi_component(S, i, j, 0, 1)
    
    imi = mi_11 + mi_00 - abs(mi_10) - abs(mi_01)
    return imi

def calculate_delta(S, i):
    """
    计算 delta_i
    对应论文公式 (17)
    """
    beta = S.shape[0]  # diffusion processes 的数量
    p_1 = calculate_probability(S, {i: 0})
    N_1 = p_1 * beta
    p_2 = calculate_probability(S, {i: 1})
    N_2 = p_2 * beta
    
    term1 = 2 * N_1 * math.log2(beta / N_1) if N_1 > 0 else 0
    term2 = 2 * N_2 * math.log2(beta / N_2) if N_2 > 0 else 0
    term3 = math.log2(beta + 1)
    
    return term1 + term2 + term3

def check_upper_bound(S, i, F_set):
    """
    检查父节点集大小是否超过理论上限 |F_i| <= log2(phi_F_i + delta_i)
    对应论文公式 (16)
    """
    if not F_set:
        return True
    
    delta_i = calculate_delta(S, i)
    # phi_F_i 是 S 中不存在的 F_set 状态组合数 (需要根据 S 统计)
    phi_F_i = 0 # 需按实际数据计算
    
    upper_bound = math.log2(phi_F_i + delta_i)
    return len(F_set) <= upper_bound

def calculate_g_score(S, i, F_set):
    """
    计算局部评分 g(v_i, F_i)
    对应论文公式 (13): g(v_i, F_i) = log2(L(v_i, F_i)) - 0.5 * sum(log2(N_ij + 1))
    
    参数:
    S: numpy.ndarray, 形状为 (beta, n) 的状态矩阵
    i: int, 目标节点 v_i 的索引
    F_set: set 或 list, 节点 v_i 的父节点集合 F_i
    
    返回:
    float, 局部评分 g(v_i, F_i)
    """
    beta = S.shape[0]
    
    # --- 边界情况：父节点集合为空 (对应论文公式 18) ---
    if not F_set:
        N1 = np.sum(S[:, i] == 0)
        N2 = np.sum(S[:, i] == 1)
        
        log_L = 0.0
        if N1 > 0: log_L += N1 * math.log2(N1 / beta)
        if N2 > 0: log_L += N2 * math.log2(N2 / beta)
        
        penalty = 0.5 * math.log2(beta + 1)
        return log_L - penalty

    # --- 一般情况：父节点集合不为空 ---
    F_list = list(F_set)
    
    # 提取父节点集合在所有传播过程中的状态子矩阵 (beta x |F_i|)
    S_F = S[:, F_list]
    # 提取目标节点 v_i 的状态列
    S_i = S[:, i]
    
    # 利用 np.unique 找出 S_F 中所有出现过的状态组合 (pi_ij) 及其频次 (N_ij)
    # unique_F: 出现过的组合矩阵; counts_F: 每个组合出现的次数 N_ij
    unique_F, counts_F = np.unique(S_F, axis=0, return_counts=True)
    
    log_L = 0.0
    penalty = 0.0
    
    # 遍历在 S 中实际出现过的每一种父节点状态组合 pi_ij
    for idx, val_F in enumerate(unique_F):
        N_ij = counts_F[idx]
        
        # 计算该组合对惩罚项的贡献
        penalty += 0.5 * math.log2(N_ij + 1)
        
        # 找到矩阵 S_F 中等于当前组合 val_F 的所有行的掩码 (mask)
        mask = np.all(S_F == val_F, axis=1)
        
        # 统计在父节点组合为 pi_ij 时，目标节点 v_i 状态为 0 的次数 (N_ij1)
        N_ij1 = np.sum((S_i == 0) & mask)
        if N_ij1 > 0:
            log_L += N_ij1 * math.log2(N_ij1 / N_ij)
            
        # 统计在父节点组合为 pi_ij 时，目标节点 v_i 状态为 1 的次数 (N_ij2)
        N_ij2 = np.sum((S_i == 1) & mask)
        if N_ij2 > 0:
            log_L += N_ij2 * math.log2(N_ij2 / N_ij)
            
    # 返回最终评分 g(v_i, F_i)
    return log_L - penalty

def tends_algorithm(n, S):
    """
    TENDS 主算法 (Algorithm 1)
    """
    E = set()
    
    # ---------------------------------------------------------
    # 阶段 1：全局计算与剪枝阈值 (步骤 2-5)
    # ---------------------------------------------------------
    print('---prepruning---')
    imi_matrix = np.zeros((n, n))
    non_negative_imis = []
    
    # 计算所有节点对的 IMI 
    # (假设前面的 calculate_imi 已经定义)
    for i in tqdm(range(n)):
        for j in range(n):
            if i != j:
                imi = calculate_imi(S, i, j) 
                imi_matrix[i, j] = imi
                if imi >= 0:
                    non_negative_imis.append(imi)
    
    # K-means 聚类提取剪枝阈值 tau
    if len(non_negative_imis) >= 2:
        kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(non_negative_imis).reshape(-1, 1))
        cluster_0_label = np.argmin(kmeans.cluster_centers_)
        tau = np.max(np.array(non_negative_imis)[kmeans.labels_ == cluster_0_label])
    else:
        tau = 0

    # ---------------------------------------------------------
    # 阶段 2：局部父节点推断与贪心搜索 (步骤 6-21 优化版)
    # ---------------------------------------------------------
    print("---parent---")
    for i in tqdm(range(n)):
        P_i = set()
        
        # 依据阈值 tau 构建候选父节点集 P_i
        for j in range(n):
            if i != j and imi_matrix[i, j] > tau:
                P_i.add(j)
        
        # 初始化最优父节点集 F_i 及其基准得分
        F_i = set() 
        current_g_score = calculate_g_score(S, i, F_i)
        
        # 前向贪心搜索
        while True:
            best_v = None
            best_g_score = current_g_score
            
            # 仅遍历尚未加入 F_i 的候选节点
            for v in (P_i - F_i):
                temp_F = F_i.union({v})
                
                # 理论上限剪枝
                # (假设前面的 check_upper_bound 已经定义)
                if check_upper_bound(S, i, temp_F):
                    temp_g_score = calculate_g_score(S, i, temp_F)
                    
                    if temp_g_score > best_g_score:
                        best_g_score = temp_g_score
                        best_v = v
                        
            # 贪心决策：正式加入得分提升最大的节点
            if best_v is not None:
                F_i.add(best_v)
                current_g_score = best_g_score
            else:
                # 局部最优，退出循环
                break
                
        # 将节点 i 的所有确定的父子关系加入边集
        for parent in F_i:
            E.add((parent, i))
            
    return E

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
    
    IG = nx.DiGraph()
    IG.add_edges_from(tends_algorithm(N, S))
    
    result_record("TENDS", calculate_F1(IG, G), "LFR", f"n{N}")
    