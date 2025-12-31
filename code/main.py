import networkx as nx
from IC_model import IC
from inverse_sigmod import run_torch_version
from MCEM import inference as em_inference
import numpy as np
from utils import calculate_MI, modified_kmeans
import warnings

warnings.filterwarnings('ignore')

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
    print(len(G.nodes),len(G.edges))
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

    #-------------------MCEM---------------------------
    # em_inference(S, A, sample_size = 10, prune_network = prune_network, iterations = 400)

    # -------------------inverse sigmod--------------------
    run_torch_version(A, S, iterations=1000, prune_network=prune_network)






