import networkx as nx
from IC_model import IC
from inverse_sigmod import run_torch_version
from MCEM import inference as em_inference
import numpy as np
from utils import calculate_MI, modified_kmeans,result_record
import warnings
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings('ignore')

def get_size_factor(comm_id):
    community_counts = Counter(node_communities.values())
    size = community_counts[comm_id]
    # 使用 log 缩放可以防止参数爆炸，+1 是为了防止 log(1)=0
    return np.log1p(size)

def generate_infections(A, num_sim = 100):
    N = A.shape[0]
    S = np.zeros([num_sim, N])
    nx_graph = nx.from_numpy_array(A)
    trees = []
    
    # 1. 初始化进度条，总数为目标数量 num_sim
    pbar = tqdm(total=num_sim, desc="Generating Infections")
    
    attempts = 0 # 用于记录总循环次数（包括失败的尝试）
    
    while len(trees) < num_sim:
        attempts += 1
        seed = np.random.choice(np.arange(0, N), size=1)
        cascade, tree = IC(Networkx_Graph=nx_graph, Seed_Set=seed, Probability=A)
        
        # 满足条件的感染才会被记录
        if len(tree.nodes) >= 3:
            S[len(trees), cascade] = 1
            trees.append(tree)
            
            # 2. 成功添加一个 tree 后，更新进度条
            pbar.update(1)
        
        # 3. 每轮循环更新后缀信息，显示当前的尝试次数
        if attempts % 10 == 0:
            pbar.set_postfix({"Total_Cycles": attempts, "Current_Trees": len(trees)})

    pbar.close() # 循环结束，关闭进度条

    average_paths = sum(len(tree.nodes()) for tree in trees)
    print(f"\naverage length of infections: {average_paths / len(trees)}")
    
    return S

if __name__ == '__main__':
    for n_sim in [2500]:
        np.random.seed(2023)
        N = 1000       # -N 1000
        AVG_K = 15     # -k 15 (average_degree)
        MAX_K = 50     # -maxk 50 (max_degree)
        MU = 0.1       # -mu 0.1 (mu)
        MIN_C = 20     # -minc 20 (min_community)
        MAX_C = 50     # -maxc 50 (max_community)
        # AVG_K = 10     # 降低平均度
        # MAX_K = 30     # 降低最大度
        # MIN_C = 30     # 增加最小社区规模，确保能容纳度数较高的节点
        # MAX_C = 60
        # MU = 0.1       # -mu 0.1 (mu)

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
        unique_comms = sorted(list(set(tuple(G.nodes[n]['community']) for n in G.nodes)))

        # 2. 创建一个映射字典：{原始集合: 新的数字编号}
        comm_to_id = {comm: i for i, comm in enumerate(unique_comms)}

        # 3. 生成最终的 node_communities，其 value 全部为数字
        node_communities = {n: comm_to_id[tuple(G.nodes[n]['community'])] for n in G.nodes}

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
            P[u, v] = weight
            P[v, u] = weight
        A = A * P
        # 调用适配的生成函数
        # 建议 num_sim 设大一些（如 500+）以获得更准的 Lift 估计
        S = generate_infections(A, num_sim=n_sim) 

        mi_matrix, p_matrix = calculate_MI(S.T)
        cluster, fixed_cluster = modified_kmeans(mi_matrix)
        threshold = max(fixed_cluster.values())
        prune_network = np.zeros([N, N])
        prune_network[mi_matrix > threshold] = 1.0
        prune_network[mi_matrix <= threshold] = 0.0

        #-------------------MCEM---------------------------
        # em_inference(S, A, sample_size = 10, prune_network = prune_network, iterations = 400)

        # -------------------inverse sigmod--------------------
        auc, cost_time = run_torch_version(A, S, iterations=2000, prune_network=prune_network)
        result_record("DNIT", auc, "LFR", f"n{N}p{n_sim}auc", 'process.jsonl')
        result_record("DNIT", cost_time, "LFR", f"n{N}p{n_sim}", 'time.jsonl')