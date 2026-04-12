import networkx as nx
import numpy as np
import itertools
from utils import result_record, calculate_F1

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

# ==========================================
# 这里插入你给出的原始代码 (保持不动)
# ==========================================
np.random.seed(2023)
N = 1500       # -N 1000
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
        max_degree=MAX_K,         # 指定最大度
        min_community=MIN_C, 
        max_community=MAX_C,      # 指定最大社区规模
        max_iters=MAX_I,          # 增加迭代次数
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
# 调用适配的生成函数
# 建议 num_sim 设大一些（如 500+）以获得更准的 Lift 估计
S = generate_infections(A, num_sim=200) 

# ==========================================
# 剩下的推断与评估部分
# ==========================================

# 执行 K-Lifts 推断
print("正在计算 Lift 指标并推断网络结构...")
vertices = list(range(N))
lifts = estimate_lifts(vertices, S)
K_target = G.number_of_edges() # 设定推断边数等于真实边数
predicted_edges = k_lifts_algorithm(vertices, lifts, K_target)

IG = nx.DiGraph()
IG.add_edges_from(predicted_edges)

result_record("klifts", calculate_F1(IG, G), "LFR", f"n{N}")