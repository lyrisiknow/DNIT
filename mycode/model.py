import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.linalg import inv, slogdet
import numpy as np
import networkx as nx

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

class RegularizedInference(nn.Module):
    def __init__(self, N, Cascades, InstancePartition, gamma, prune_network):
        super(RegularizedInference, self).__init__()
        
        self.N = N
        self.gamma = gamma
        
        # 优化参数
        self.A_param = nn.Parameter(torch.full((N, N), -2.0))
        
        # 处理剪枝网络并注册为 buffer (自动随模型 to(device))
        prune_network[prune_network == 0] = 1e-5
        self.register_buffer('prune_network_tensor', torch.from_numpy(prune_network).float())

        # =================================================================
        # 预计算 1: 级联交互频率矩阵 K (彻底消除 forward 中的级联循环)
        # =================================================================
        S_np = np.array(Cascades)
        self.num_cascades = S_np.shape[0]
        
        # S_I 为 1 (感染), S_O 为 0 (未感染)
        S_I_tensor = torch.tensor((S_np == 1), dtype=torch.float32)
        S_O_tensor = torch.tensor((S_np == 0), dtype=torch.float32)
        
        # K_matrix[i, j] 记录了多少次级联中：节点 i 感染 且 节点 j 未感染
        K_matrix = torch.matmul(S_I_tensor.T, S_O_tensor) 
        self.register_buffer('K_matrix', K_matrix)

        # =================================================================
        # 预计算 2: 实例划分掩码矩阵 M (彻底消除 regularization 中的实例循环)
        # =================================================================
        # 假设 InstancePartition 是 {node_id: instance_id} 或者列表
        unique_insts = list(set(InstancePartition.values()) if isinstance(InstancePartition, dict) else set(InstancePartition))
        num_insts = len(unique_insts)
        M = torch.zeros(N, num_insts, dtype=torch.float32)
        
        for u in range(N):
            inst_id = InstancePartition[u] if isinstance(InstancePartition, dict) else InstancePartition[u]
            idx = unique_insts.index(inst_id)
            M[u, idx] = 1.0
            
        self.register_buffer('M_matrix', M)
        
        # 计算每个实例包含的节点数，防止除以 0
        counts = M.sum(dim=0).unsqueeze(1) # shape: (num_insts, 1)
        counts[counts == 0] = 1.0 
        self.register_buffer('M_counts', counts)

    def _get_prob_matrix(self):
        A_prob = torch.sigmoid(self.A_param)
        # 屏蔽对角线并应用剪枝网络
        A_prob = A_prob * (1.0 - torch.eye(self.N, device=A_prob.device))
        A_prob = A_prob * self.prune_network_tensor
        return A_prob

    def construct_W_hat(self, A_prob):
        # 向量化构造 W_hat (移除原有的 for 循环)
        sub_A = A_prob * (1.0 - torch.eye(self.N, device=A_prob.device))
        W = -sub_A
        # 将列和直接加到对角线上
        W = W + torch.diag(torch.sum(sub_A, dim=0))
        return W

    def forward(self):
        A_prob = self._get_prob_matrix()
        eps = 1e-8
        
        # =================================================================
        # 1. 高速计算 Negative Log Likelihood
        # =================================================================
        # Term 1: 矩阵点乘代替循环
        # K_matrix 已经包含了所有的感染/未感染关系频次，直接做内积并求和
        term1_total = torch.sum(self.K_matrix * torch.log(1.0 - A_prob + eps))
        
        # Term 2: W_hat 矩阵仅受 A_prob 影响，与具体的级联无关
        # 我们只需计算一次 slogdet，然后乘以总级联数！(速度飙升的关键点)
        W_hat_l = self.construct_W_hat(A_prob)
        sign, log_abs_det = torch.linalg.slogdet(W_hat_l)
        term2_total = log_abs_det * self.num_cascades
        
        log_L = (term1_total + term2_total) / (self.num_cascades * self.N)
        NLL = -log_L
        
        # =================================================================
        # 2. 高速计算 Regularization (Omega)
        # =================================================================
        # 步骤 A: 计算每个 cluster 内部的 A_prob 均值向量 -> shape: (num_insts, N)
        A_sum = torch.matmul(self.M_matrix.T, A_prob)
        A_mean = A_sum / self.M_counts 
        
        # 步骤 B: 将均值重构回 (N, N) 用于计算方差
        A_approx = torch.matmul(self.M_matrix, A_mean) 
        
        # 步骤 C: 均方差计算 (直接利用 tensor 操作，无需遍历)
        Omega = torch.sum((A_prob - A_approx)**2) / (self.N * self.N) 
        
        # 3. 最终返回 Loss
        return NLL + self.gamma * Omega

def run_torch_version(G, N, S, C, gamma, prune_network, iterations=500, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    np.fill_diagonal(prune_network, 0.0)
    
    # 模型初始化时所有的预计算矩阵会被创建
    model = RegularizedInference(N=N, 
                                 Cascades=S, 
                                 InstancePartition=C, 
                                 gamma=gamma,
                                 prune_network=prune_network).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr) 
    
    model.train()
    for i in tqdm.tqdm(range(iterations), desc="Optimizing"):
        optimizer.zero_grad()
        loss = model() 
        loss.backward() 
        optimizer.step() 
        
        # 为了避免影响 tqdm 的输出，建议降低打印频率 (比如每50轮打印一次)
        if (i+1) % 50 == 0:
            tqdm.tqdm.write(f"Iteration {i+1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        A_star = model._get_prob_matrix().cpu().numpy()
        
    A_star = A_star * prune_network
    A_star[A_star <= 1e-5] = 0.0
    
    best_t, IG = post_processing(A_star)
    P, R, F1 = calculate_F1(IG, G)
    print(f"BEP point : {best_t:.5f} | P: {P}, R: {R}, F1: {F1}")
    
    return calculate_F1(IG, G)