import networkx as nx
import numpy as np
import itertools
from utils import generate_infections, result_record, calculate_F1, calculate_binary_auc
from TENDS import tends_algorithm
from TWIND import run_twind_fast
from SIDN import infer_sidn_network
from PIND import pind_inference
from collections import Counter
import time


def get_size_factor(comm_id):
    community_counts = Counter(node_communities.values())
    size = community_counts[comm_id]
    # 使用 log 缩放可以防止参数爆炸，+1 是为了防止 log(1)=0
    return np.log1p(size)

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
        S = generate_infections(A, num_sim=1000) 

        # ==========================================
        # 剩下的推断与评估部分
        # ==========================================
        G = nx.from_numpy_array(A)
        nodes = list(G.nodes())
        # 执行 K-Lifts 推断
        print(f"===={N}====")
        print('TWIND:')
        IG = nx.DiGraph()
        IG.add_nodes_from(nodes)
        start_time = time.time()  # 开始计时
        twind_edges = run_twind_fast(S)
        end_time = time.time()    # 结束计时
        twind_cost = end_time - start_time
        IG.add_edges_from(twind_edges)
        result_record("TWIND", calculate_binary_auc(IG, G), "LFR", f"n{N}auc", 'resultn.jsonl')
        result_record("TWIND", twind_cost, "LFR", f"n{N}", 'timen.jsonl')
        result_record("TWIND", calculate_F1(IG, G), "LFR", f"n{N}f1", 'resultn.jsonl')
        
        print('TENDS:')
        IG = nx.DiGraph()
        IG.add_nodes_from(nodes)
        start_time = time.time()
        tends_edges = tends_algorithm(N, S)
        end_time = time.time()
        tends_cost = end_time - start_time
        IG.add_edges_from(tends_edges)
        result_record("TENDS", calculate_binary_auc(IG, G), "LFR", f"n{N}auc", 'resultn.jsonl')
        result_record("TENDS", tends_cost, "LFR", f"n{N}", 'timen.jsonl')
        result_record("TENDS", calculate_F1(IG, G), "LFR", f"n{N}f1", 'resultn.jsonl')
        
        print('SIDN:')
        start_time = time.time()
        sidn_matrix = infer_sidn_network(S)
        end_time = time.time()
        sidn_cost = end_time - start_time

        IG = nx.from_numpy_array(sidn_matrix, create_using=nx.DiGraph)
        result_record("SIDN", calculate_binary_auc(IG, G), "LFR", f"n{N}auc", 'resultn.jsonl')
        result_record("SIDN", sidn_cost, "LFR", f"n{N}", 'timen.jsonl')

        result_record("SIDN", calculate_F1(IG, G), "LFR", f"n{N}f1", 'resultn.jsonl')
        
        print('PIND:')
        start_time = time.time()
        pind_matrix = pind_inference(S)
        end_time = time.time()
        pind_cost = end_time - start_time

        IG = nx.from_numpy_array(pind_matrix, create_using=nx.DiGraph)
        result_record("PIND", calculate_binary_auc(IG, G), "LFR", f"n{N}auc", 'resultn.jsonl')
        result_record("PIND", pind_cost, "LFR", f"n{N}", 'timen.jsonl')
        result_record("PIND", calculate_F1(IG, G), "LFR", f"n{N}f1", 'resultn.jsonl')
for N in [100,150,200,250,300]:
        np.random.seed(2023)
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
        S = generate_infections(A, num_sim=100) 

        # ==========================================
        # 剩下的推断与评估部分
        # ==========================================
        G = nx.from_numpy_array(A)
        nodes = list(G.nodes())
        # 执行 K-Lifts 推断
        print(f"===={N}====")
        print('TWIND:')
        IG = nx.DiGraph()
        IG.add_nodes_from(nodes)
        start_time = time.time()  # 开始计时
        twind_edges = run_twind_fast(S)
        end_time = time.time()    # 结束计时
        twind_cost = end_time - start_time
        IG.add_edges_from(twind_edges)
        result_record("TWIND", calculate_binary_auc(IG, G), "LFR", f"n{N}auc", 'resultn.jsonl')
        result_record("TWIND", twind_cost, "LFR", f"n{N}", 'timen.jsonl')
        result_record("TWIND", calculate_F1(IG, G), "LFR", f"n{N}f1", 'resultn.jsonl')
        
        print('TENDS:')
        IG = nx.DiGraph()
        IG.add_nodes_from(nodes)
        start_time = time.time()
        tends_edges = tends_algorithm(N, S)
        end_time = time.time()
        tends_cost = end_time - start_time
        IG.add_edges_from(tends_edges)
        result_record("TENDS", calculate_binary_auc(IG, G), "LFR", f"n{N}auc", 'resultn.jsonl')
        result_record("TENDS", tends_cost, "LFR", f"n{N}", 'timen.jsonl')
        result_record("TENDS", calculate_F1(IG, G), "LFR", f"n{N}f1", 'resultn.jsonl')
        
        print('SIDN:')
        start_time = time.time()
        sidn_matrix = infer_sidn_network(S)
        end_time = time.time()
        sidn_cost = end_time - start_time

        IG = nx.from_numpy_array(sidn_matrix, create_using=nx.DiGraph)
        result_record("SIDN", calculate_binary_auc(IG, G), "LFR", f"n{N}auc", 'resultn.jsonl')
        result_record("SIDN", sidn_cost, "LFR", f"n{N}", 'timen.jsonl')

        result_record("SIDN", calculate_F1(IG, G), "LFR", f"n{N}f1", 'resultn.jsonl')
        
        print('PIND:')
        start_time = time.time()
        pind_matrix = pind_inference(S)
        end_time = time.time()
        pind_cost = end_time - start_time

        IG = nx.from_numpy_array(pind_matrix, create_using=nx.DiGraph)
        result_record("PIND", calculate_binary_auc(IG, G), "LFR", f"n{N}auc", 'resultn.jsonl')
        result_record("PIND", pind_cost, "LFR", f"n{N}", 'timen.jsonl')
        result_record("PIND", calculate_F1(IG, G), "LFR", f"n{N}f1", 'resultn.jsonl')
