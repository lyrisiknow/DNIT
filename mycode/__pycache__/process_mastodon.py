import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from model1 import run_torch_version
import json
from collections import Counter

def get_size_factor(comm_id):
    community_counts = Counter(node_communities.values())
    size = community_counts[comm_id]
    # 使用 log 缩放可以防止参数爆炸，+1 是为了防止 log(1)=0
    return np.log1p(size)

def result_record(alg_name, ret, dataset, param='', file="result.jsonl"):
    # 1. 动态构造键名 (Key)
    key = f"{dataset}_{alg_name}_{param}" if param else f"{dataset}_{alg_name}"
    
    # 2. 构造字典对象
    record_dict = {key: ret}
    
    # 3. 追加写入文件
    with open(file, 'a') as f:
        # json.dumps 会自动给键加上双引号，并将元组 (0.6..., ...) 转换为列表 [0.6..., ...]
        f.write(json.dumps(record_dict) + '\n')

def modified_kmeans_fast(mi_matrix, tolerance=1e-7):
    n = mi_matrix.shape[0]
    
    # 1. 提取上三角非零元素 (因为 mi_matrix 对称，且 j > i)
    # 这一步将 O(n^2) 的搜索范围缩小到实际有效的值
    triu_indices = np.triu_indices(n, k=1)
    all_values = mi_matrix[triu_indices]
    
    # 过滤掉 <= 0 的值 (对应原代码 if mi_matrix[i,j] <= 0: continue)
    valid_mask = all_values > 0
    values = all_values[valid_mask]
    # 记录这些有效值在原矩阵中的位置，最后还原字典用
    rows = triu_indices[0][valid_mask]
    cols = triu_indices[1][valid_mask]

    # 2. 初始化中心点
    fixed_centroid = 0.0
    centroid = np.max(values) if len(values) > 0 else 0.0
    
    is_stable = False
    
    while not is_stable:
        # 3. 向量化分类：计算每个点到两个中心点的距离
        # 原条件: (val - fixed_centroid) <= abs(val - centroid)
        dist_to_fixed = np.abs(values - fixed_centroid)
        dist_to_active = np.abs(values - centroid)
        
        # active_mask 为 True 表示该值应属于 cluster (动态簇)
        # 对应原代码 else 分支
        active_mask = dist_to_active < dist_to_fixed
        
        # 4. 更新动态中心点 (centroid)
        if np.any(active_mask):
            new_centroid = np.mean(values[active_mask])
        else:
            new_centroid = centroid
            
        # 5. 检查收敛条件：中心点偏移量小于阈值则稳定
        if abs(new_centroid - centroid) < tolerance:
            is_stable = True
        
        centroid = new_centroid

    # 6. 一次性还原为字典输出 (如果你的后续逻辑必须用字典)
    # 注意：如果 n 非常大，建议直接返回 mask 以节省内存
    fixed_mask = ~active_mask
    
    cluster = dict(zip(zip(rows[active_mask], cols[active_mask]), values[active_mask]))
    fixed_cluster = dict(zip(zip(rows[fixed_mask], cols[fixed_mask]), values[fixed_mask]))

    return cluster, fixed_cluster

def fast_mi_and_prob(x):
    # 假设 x 的形状是 (n_features, m_samples)
    n, m = x.shape
    
    # 1. 计算每个变量为 1 和 0 的概率
    # 使用 .reshape(-1) 确保它们是一维数组，方便后续计算
    count_1 = x.sum(axis=1).get() if hasattr(x, 'get') else x.sum(axis=1)
    count_1 = count_1.astype(float)
    count_0 = m - count_1
    
    p_i1 = count_1 / m
    p_i0 = count_0 / m

    # 2. 计算联合分布计数 (n x n)
    # count_11[i, j] 是 i=1 且 j=1 的样本数
    count_11 = x @ x.T
    
    # 3. 这里的 count_1 是一维的 (n,)，利用广播机制计算其他组合
    # count_1[:, None] 将其变为 (n, 1)
    count_1_col = count_1[:, np.newaxis]
    count_1_row = count_1[np.newaxis, :]
    
    count_10 = count_1_col - count_11
    count_01 = count_1_row - count_11
    count_00 = m - (count_11 + count_10 + count_01)

    # 4. 计算条件概率矩阵 p[i, j] = p(j=1 | i=1)
    # 注意：这里 i 是行，j 是列。原代码逻辑 p[i,j] = p_i1_j1 / p_i1
    p_matrix = count_11 / (count_1_col + 1e-12)

    # 5. 计算互信息 MI
    mi_matrix = np.zeros((n, n))
    
    # 组合列表：(联合概率, 边际概率1, 边际概率2)
    # p_i 和 p_j 均为形状为 (n,) 的一维数组
    pairs = [
        (count_11, p_i1, p_i1), # (1,1)
        (count_10, p_i1, p_i0), # (1,0)
        (count_01, p_i0, p_i1), # (0,1)
        (count_00, p_i0, p_i0)  # (0,0)
    ]

    for c_ij, p_i_vec, p_j_vec in pairs:
        p_ij = c_ij / m
        # 计算边际概率的乘积矩阵 P(i)*P(j)
        # np.outer(p_i_vec, p_j_vec) 会生成 (n, n) 矩阵
        p_i_p_j = np.outer(p_i_vec, p_j_vec)
        
        # 掩码计算：只有当联合概率和边际概率乘积均大于 0 时才计算
        mask = (p_ij > 1e-12) & (p_i_p_j > 1e-12)
        
        # MI 公式项
        mi_matrix[mask] += p_ij[mask] * np.log(p_ij[mask] / p_i_p_j[mask])

    return p_matrix, mi_matrix

def fast_imi_and_prob(x):
    # 假设 x 的形状是 (n_features, m_samples)
    if hasattr(x, 'get'): x = x.get() # 如果是 cupy 数组转为 numpy
    n, m = x.shape
    
    # 1. 计算每个变量为 1 和 0 的概率
    count_1 = x.sum(axis=1).astype(float)
    count_0 = m - count_1
    
    p_i1 = count_1 / m
    p_i0 = count_0 / m

    # 2. 计算联合分布计数 (n x n)
    count_11 = x @ x.T
    
    # 3. 利用广播机制计算其他组合
    count_1_col = count_1[:, np.newaxis]
    count_1_row = count_1[np.newaxis, :]
    
    count_10 = count_1_col - count_11
    count_01 = count_1_row - count_11
    count_00 = m - (count_11 + count_10 + count_01)

    # 4. 计算条件概率矩阵 p[i, j] = p(j=1 | i=1)
    p_matrix = count_11 / (count_1_col + 1e-12)

    # 5. 计算 IMI
    imi_matrix = np.zeros((n, n))
    
    # 定义四个分量的配置：(联合计数, 行边缘概率, 列边缘概率, 是否为负贡献)
    # 这里的贡献符号 sign 对应公式中的 + 或 -
    components = [
        (count_11, p_i1, p_i1, 1),  # MI(1,1) -> 正向
        (count_00, p_i0, p_i0, 1),  # MI(0,0) -> 正向
        (count_10, p_i1, p_i0, -1), # -|MI(1,0)| -> 负向
        (count_01, p_i0, p_i1, -1)  # -|MI(0,1)| -> 负向
    ]

    for c_ij, p_row_vec, p_col_vec, sign in components:
        p_ij = c_ij / m
        # 计算 P(Xi)*P(Xj) 矩阵
        p_i_p_j = np.outer(p_row_vec, p_col_vec)
        
        # 避免 log(0) 或 除以 0
        mask = (p_ij > 1e-12) & (p_i_p_j > 1e-12)
        
        # 计算单项 MI
        term = np.zeros((n, n))
        term[mask] = p_ij[mask] * np.log(p_ij[mask] / p_i_p_j[mask])
        
        # 根据 sign 累加到最终矩阵
        if sign == 1:
            imi_matrix += term
        else:
            # 公式要求减去绝对值: -|MI|
            imi_matrix -= np.abs(term)

    return p_matrix, imi_matrix

def IC(Networkx_Graph, Seed_Set, Probability):

    tree = nx.DiGraph()
    tree.add_node(Seed_Set[0])
    new_active, Ans = Seed_Set.tolist(), Seed_Set.tolist()
    while new_active:
        # Getting neighbour nodes of newly activate node
        (targets, edges) = Neighbour_finder(Networkx_Graph, Probability, new_active)
        # Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.

        new_active = []

        for (node, target) in edges:
            if np.random.uniform(0, 1) < Probability[node, target]:
                if target not in Ans: #success infected
                    tree.add_edge(node, target)
                    new_active.append(target)
                    Ans.append(target)
        # Checking which ones in new_ones are not in our Ans...only adding them to our Ans so that no duplicate in Ans.

    return Ans, tree


def Neighbour_finder(g, p, new_active):
    targets = []
    edges = []
    for node in new_active:
        node_neighbors = list(g.neighbors(node))
        targets += node_neighbors
        for target in node_neighbors:
            edges.append((node,target))

    return (targets, edges)

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
    for n_sim in [500,1000,1500,2000,2500]:
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
        S = generate_infections(A, num_sim=n_sim) 
        G = nx.from_numpy_array(A)
        
        print('pre_pruning')
        mi_matrix, p_matrix = fast_imi_and_prob(S.T)
        cluster, fixed_cluster = modified_kmeans_fast(mi_matrix)
        threshold = max(fixed_cluster.values())
        prune_network = np.zeros([N, N])
        prune_network[mi_matrix > threshold] = 1.0
        prune_network[mi_matrix <= threshold] = 0.0
        
        C = {}

        
        l = set()
        for node in node_communities:
            l.add(node_communities[node])
        print(len(l))
        
        dict_c = dict()
        for i, item in enumerate(l):
            dict_c[item] = i
            
        for node in node_communities:
            C[nodes_idx[node]] = dict_c[node_communities[node]]
            
        gamma = 0.01
            
        iterations = 10000
        lr = 0.01
        auc, cost_time = run_torch_version(G, N, S, C, A, gamma, prune_network, iterations = iterations, lr=lr)
        result_record("mymodel", auc, "mastodon", f'p{n_sim}auc', 'process.jsonl')
        result_record("mymodel", cost_time, "mastodon", f'p{n_sim}', 'time.jsonl')