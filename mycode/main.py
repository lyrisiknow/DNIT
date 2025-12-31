import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from model import run_torch_version

def modified_kmeans(mi_matrix):

    fixed_centroid = 0.0
    centroid = np.max(mi_matrix)

    fixed_cluster = {}
    cluster = {}

    [n,_] = mi_matrix.shape

    for i in range(n):
        for j in range(i+1,n):
            if mi_matrix[i,j] <= 0:
                continue

            if (mi_matrix[i,j]) - fixed_centroid <= (mi_matrix[i,j] - centroid):
                fixed_cluster[(i,j)] = mi_matrix[i,j]
            else:
                cluster[(i, j)] = mi_matrix[i, j]


    is_stable = False

    while is_stable == False:

        centroid = np.mean([cluster[key] for key in cluster.keys()])
        modified_count = 0

        for i in range(n):
            for j in range(i + 1, n):

                if (mi_matrix[i, j] - fixed_centroid) <= abs(mi_matrix[i, j] - centroid):
                    if (i,j) not in fixed_cluster.keys():
                        modified_count += 1
                        fixed_cluster[(i, j)] = mi_matrix[i, j]

                        if (i,j) in cluster.keys():
                            cluster.pop((i, j))
                else:
                    if (i, j) not in cluster.keys():
                        modified_count += 1
                        cluster[(i, j)] = mi_matrix[i, j]

                        if (i, j) in fixed_cluster.keys():
                            fixed_cluster.pop((i, j))

                    cluster[(i, j)] = mi_matrix[i, j]

        if modified_count > 0:
            is_stable = False

        else:
            is_stable = True

    return cluster, fixed_cluster

def calculate_MI(x):
    '''
    x: [n,beta]
    '''
    [n,m] = x.shape

    mi_matrix = np.zeros([n,n])
    p = np.zeros([n,n])


    #TODO Optimize
    for i in range(n):

        v_i = x[i, :]
        v_i_1 = np.where(v_i == 1)[0]
        v_i_0 =  np.where(v_i == 0)[0]
        p_i1 = len(v_i_1) / m * 1.0
        p_i0 = len(v_i_0) / m * 1.0

        for j in range(i+1,n):
            v_j = x[j,:]
            v_j_1 = np.where(v_j == 1)[0]
            v_j_0 = np.where(v_j == 0)[0]

            p_j1 = len(v_j_1) / m * 1.0
            p_j0 = len(v_j_0) / m * 1.0

            p_i1_j1 = len(np.intersect1d(v_j_1, v_i_1)) / m * 1.0
            p_i1_j0 = len(np.intersect1d(v_j_0, v_i_1)) / m * 1.0
            p_i0_j1 = len(np.intersect1d(v_j_1, v_i_0)) / m * 1.0
            p_i0_j0 = len(np.intersect1d(v_j_0, v_i_0)) / m * 1.0

            if p_i1 > 0:
                p[i,j] = p_i1_j1 / p_i1

            if p_j1 > 0:
                p[j,i] = p_i1_j1 / p_j1

            if  p_i1_j1 > 0:
                mi_matrix[i,j] += p_i1_j1 * np.log(p_i1_j1 / (p_i1 * p_j1))
                mi_matrix[j,i] += p_i1_j1 * np.log(p_i1_j1 / (p_i1 * p_j1))

            if  p_i1_j0 > 0:
                mi_matrix[i, j] += p_i1_j0 * np.log(p_i1_j0 / (p_i1 * p_j0))
                mi_matrix[j, i] += p_i1_j0 * np.log(p_i1_j0 / (p_i1 * p_j0))

            if  p_i0_j1 > 0:
                mi_matrix[i, j] += p_i0_j1 * np.log(p_i0_j1 / (p_i0 * p_j1))
                mi_matrix[j, i] += p_i0_j1 * np.log(p_i0_j1 / (p_i0 * p_j1))

            if  p_i0_j0 > 0:
                mi_matrix[i, j] += p_i0_j0 * np.log(p_i0_j0 / (p_i0 * p_j0))
                mi_matrix[j, i] += p_i0_j0 * np.log(p_i0_j0 / (p_i0 * p_j0))

    return mi_matrix, p

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
    
    mi_matrix, p_matrix = calculate_MI(S.T)
    cluster, fixed_cluster = modified_kmeans(mi_matrix)
    threshold = max(fixed_cluster.values())
    prune_network = np.zeros([N, N])
    prune_network[mi_matrix > threshold] = 1.0
    prune_network[mi_matrix <= threshold] = 0.0
    
    C = {}
# 遍历图 G 中的每一个节点及其属性
    for node, data in G.nodes(data=True):
        # LFR 生成器将社区信息存储在 'community' 属性中
        # 注意：'community' 属性的值是一个集合（set），包含一个社区ID（整数）
        community_set = data.get('community') 
        
        if community_set:
            # 提取集合中的唯一社区ID
            community_id = list(community_set)[0]
            C[node] = community_id
        else:
            # 处理没有社区信息（理论上LFR图不会发生）
            C[node] = -1 # 可以用一个特殊值表示未分配

    
    l = set()
    for node in C:
        l.add(C[node])
    print(len(l))
    
    dict_c = dict()
    for i, item in enumerate(l):
        dict_c[item] = i
        
    for node in C:
        C[node] = dict_c[C[node]]
        
    gamma = 0.01
    run_torch_version(G, N, S, C, gamma, prune_network, iterations=500, lr=0.01)