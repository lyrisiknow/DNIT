import networkx as nx
import numpy as np
import itertools
from utils import generate_infections, result_record, calculate_F1
from TENDS import tends_algorithm
from TWIND import run_twind_fast
from SIDN import infer_sidn_network
from PIND import pind_inference


if __name__ == '__main__':
    # ==========================================
    # 这里插入你给出的原始代码 (保持不动)
    # ==========================================
    for N in [1000,1500,2000,2500,3000]:
        np.random.seed(2023)
        #N = 2000       # -N 1000-3000
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
        S = generate_infections(A, num_sim=100) 

        # ==========================================
        # 剩下的推断与评估部分
        # ==========================================
        G = nx.from_numpy_array(A)
        # 执行 K-Lifts 推断
        print(f"===={N}====")
        print('TWIND:')
        IG = nx.DiGraph()
        IG.add_edges_from(run_twind_fast(S))
        result_record("TWIND", calculate_F1(IG, G), "LFR", f"n{N}")
        
        print('TENDS:')
        IG = nx.DiGraph()
        IG.add_edges_from(tends_algorithm(N, S))
        result_record("TENDS", calculate_F1(IG, G), "LFR", f"n{N}")
        
        print('SIDN:')
        IG = nx.DiGraph()
        IG = nx.from_numpy_array(infer_sidn_network(S), create_using=nx.DiGraph)
        result_record("SIDN", calculate_F1(IG, G), "LFR", f"n{N}")
        
        print('PIND:')
        IG = nx.DiGraph()
        IG = nx.from_numpy_array(pind_inference(S), create_using=nx.DiGraph)
        result_record("PIND", calculate_F1(IG, G), "LFR", f"n{N}")
