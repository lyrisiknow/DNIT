import networkx as nx
import numpy as np
import json
from sklearn.metrics import roc_auc_score, average_precision_score

def not_infected_matrix(S):

    num_sim, num_node = S.shape

    not_infected = np.zeros([num_node, num_node])

    for i in range(num_node):
        for j in range(num_node):
            if i == j:
                continue
            s_i_j = S[:,i] - S[:,j]
            not_infected[i,j] = len(np.where(s_i_j == 1)[0])

    return not_infected

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


def calculate_F1(IG,RG):

    ig_edges = IG.edges
    rg_edges = RG.edges

    TP = 0.0
    FP = 0.0
    FN = 0.0

    for (i,j) in ig_edges:
        if (i,j) in rg_edges or (j,i) in rg_edges:
            TP += 1.0
        else:
            FP += 1.0

    for (i,j) in rg_edges:
        if (i,j) not in ig_edges and (j,i) not in ig_edges:
            FN += 1.0

    P = TP / (TP+FP)
    R = TP / (TP+FN)

    return round(P,3),round(R,3),round(2*P*R / (P+R),3)


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

def post_processing(estimated_A):
    #predicted BEP
    '''
    Ensemble method
    '''

    thresholds = np.linspace(start = 1e-5, stop=5e-1, num=10000)

    FP_FN_diff = np.zeros([len(thresholds)])

    for i, t in enumerate(thresholds):

        predicted_FN = np.sum(estimated_A[estimated_A < t])
        predicted_FP = np.sum(1.0 - estimated_A[estimated_A >= t])

        FP_FN_diff[i] = np.abs(predicted_FP - predicted_FN)

    best_t = thresholds[np.argmin(FP_FN_diff)]
    IG = np.zeros_like(estimated_A)
    IG[estimated_A >= best_t] = 1
    IG[estimated_A < best_t] = 0

    IG = nx.from_numpy_array(IG)

    return best_t, IG

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

def result_record(alg_name, ret, dataset, param=''):
    # 1. 动态构造键名 (Key)
    key = f"{dataset}_{alg_name}_{param}" if param else f"{dataset}_{alg_name}"
    
    # 2. 构造字典对象
    record_dict = {key: ret}
    
    # 3. 追加写入文件
    with open("result.jsonl", 'a') as f:
        # json.dumps 会自动给键加上双引号，并将元组 (0.6..., ...) 转换为列表 [0.6..., ...]
        f.write(json.dumps(record_dict) + '\n')

def calculate_binary_auc(IG, G):
    """
    基于二值化的预测图 IG 和真实图 G 计算指标。
    注意：这里的 AUC 仅代表该特定阈值下的单点表现。
    """
    # 1. 转换为邻接矩阵
    # 确保节点顺序一致
    nodes = sorted(G.nodes())
    adj_predict = nx.to_numpy_array(IG, nodelist=nodes)
    adj_true = nx.to_numpy_array(G, nodelist=nodes)
    
    # 2. 提取上三角部分（忽略对角线，适用于无向图）
    iu = np.triu_indices(len(nodes), k=1)
    y_true = (adj_true[iu] > 0).astype(int)
    y_predict = adj_predict[iu].astype(int)
    
    # 3. 安全检查
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    
    # 4. 计算指标
    # 注意：此时 y_predict 是 0/1，roc_auc 相当于计算梯形的面积
    binary_roc_auc = roc_auc_score(y_true, y_predict)
    binary_pr_auc = average_precision_score(y_true, y_predict)
    
    print(f"Binary ROC-AUC: {binary_roc_auc:.4f}")
    print(f"Binary PR-AUC: {binary_pr_auc:.4f}")
    
    return round(binary_roc_auc,4), round(binary_pr_auc,4)
