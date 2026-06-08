import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import networkx as nx
from tqdm import tqdm
from utils import generate_infections, result_record, calculate_F1, IC, Neighbour_finder
from concurrent.futures import ProcessPoolExecutor, as_completed

def compute_mutual_information(prob_matrix):
    """
    第一阶段：计算节点间的互信息 (简化版)
    prob_matrix: (C, N) 矩阵，C个级联，N个节点，存储感染概率
    """
    C, N = prob_matrix.shape
    mi_matrix = np.zeros((N, N))
    
    # 这里为了演示使用相关性或简化互信息计算
    # 实际论文中需基于边缘分布和联合分布计算 I(Ui; Uj)
    for i in tqdm(range(N)):
        for j in range(N):
            if i == j: continue
            # 简化：计算概率向量的协方差作为相关性度量
            mi_matrix[i, j] = np.abs(np.cov(prob_matrix[:, i], prob_matrix[:, j])[0, 1])
            
    return mi_matrix

def pruning_stage(mi_matrix):
    """
    第一阶段：改进的互信息剪枝（增加异常处理）
    """
    N = mi_matrix.shape[0]
    candidate_parents = {}
    
    for j in range(N):
        scores = mi_matrix[:, j].reshape(-1, 1)
        
        # 检查数据点中不同数值的数量
        unique_scores = np.unique(scores)
        
        # 如果数据点太少，或者所有互信息分数都一样
        if len(unique_scores) < 2:
            # 策略：如果分数全为 0，则没有候选父节点；否则将所有非零节点视为候选
            if unique_scores[0] > 0:
                candidate_parents[j] = np.where(mi_matrix[:, j] > 0)[0]
            else:
                candidate_parents[j] = np.array([])
            continue

        # 只有在至少有两个不同值时才运行 K-means
        kmeans = KMeans(n_clusters=2, n_init=10).fit(scores)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_.flatten()
        
        high_corr_label = np.argmax(centers)
        candidate_parents[j] = np.where(labels == high_corr_label)[0]
        
    return candidate_parents

def objective_function(x, alpha, j, parents, prob_matrix):
    """
    第二阶段：似然函数的目标函数 (负对数似然)
    x: 待优化的边概率向量 (针对节点 j 的所有候选父节点)
    """
    C = prob_matrix.shape[0]
    u_j = prob_matrix[:, j]
    u_parents = prob_matrix[:, parents]
    
    # 核心模型：1 - product(1 - x_ij * alpha_ij * u_i)
    # 这里简化 alpha 为固定值或统一处理
    prob_infection = 1 - np.prod(1 - x * alpha * u_parents, axis=1)
    
    # 避免数值不稳定性
    prob_infection = np.clip(prob_infection, 1e-10, 1 - 1e-10)
    
    # 对数似然：u_j * log(P) + (1-u_j) * log(1-P)
    log_likelihood = u_j * np.log(prob_infection) + (1 - u_j) * np.log(1 - prob_infection)
    return -np.sum(log_likelihood)

def _optimize_node(j, parents, prob_matrix, max_iter):
    # 如果没有候选父节点，直接返回空结果
    if len(parents) == 0:
        return j, parents, None
        
    num_p = len(parents)
    x = np.full(num_p, 0.5)
    alpha = 0.5 
    
    # 【优化点1】将边界条件移出内层循环，避免不必要的重复内存分配
    bounds = [(0, 1)] * num_p
    
    for i in range(max_iter):
        # 固定 alpha，优化 x
        res = minimize(objective_function, x, args=(alpha, j, parents, prob_matrix), 
                       bounds=bounds, method='L-BFGS-B')
        x = res.x
        
        # (可选) 固定 x，优化 alpha 的逻辑...
        
    return j, parents, x

def pind_inference(prob_matrix, max_iter=10, n_workers=None):
    """
    PIND 主函数 (多进程加速版)
    prob_matrix: 形状为 (级联数, 节点数) 的概率矩阵
    n_workers: 进程数。默认为 None (使用所有可用 CPU 核心)
    """
    C, N = prob_matrix.shape
    
    # 1. 剪枝
    print("正在进行互信息剪枝...")
    # 假设这两个函数在外部已定义
    mi_matrix = compute_mutual_information(prob_matrix)
    candidate_parents = pruning_stage(mi_matrix)
    
    # 初始化网络矩阵 A (N x N)
    estimated_adj = np.zeros((N, N))
    
    # 2. 交替最大化迭代 (并行化)
    print("正在进行非线性回归推断 (多核并行加速中)...")
    
    # 【优化点2】使用进程池并发执行独立的节点推断任务
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # 构建任务字典，保存 future 对应的节点索引 j
        futures = {
            executor.submit(_optimize_node, j, candidate_parents[j], prob_matrix, max_iter): j 
            for j in range(N)
        }
        
        # 使用 tqdm 结合 as_completed 实现准确的并行进度条
        for future in tqdm(as_completed(futures), total=N):
            j, parents, x = future.result()
            
            # 记录结果
            if x is not None:
                estimated_adj[parents, j] = x
                
    return estimated_adj

if __name__ == '__main__':
    # --- 设定参数 ---
    np.random.seed(2023)
    for N in [100,150,200,250,300]:
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
        
        IG = nx.from_numpy_array(pind_inference(S), create_using=nx.DiGraph)
        
        result_record("PIND", calculate_F1(IG, G), "LFR", f"n{N}")