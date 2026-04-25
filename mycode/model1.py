import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.linalg import inv, slogdet
import numpy as np
import networkx as nx
from kneed import KneeLocator

class RegularizedInferenceIC(nn.Module):
    def __init__(self, N, Cascades, InstancePartition, gamma, prune_network):
        super(RegularizedInferenceIC, self).__init__()
        
        self.N = N
        self.gamma = gamma
        
        # 优化参数：网络边的概率对数 (对应 alpha)
        self.A_param = nn.Parameter(torch.zeros((N, N)))
        
        # 处理剪枝网络并注册为 buffer
        prune_network[prune_network == 0] = 1e-5
        self.register_buffer('prune_network_tensor', torch.from_numpy(prune_network).float())

        # =================================================================
        # 预计算 1: 级联状态矩阵 X (直接利用 GPU 矩阵乘法替代循环)
        # =================================================================
        # Cascades 形状为 (L, N)，即 L 个级联，N 个节点。X_li 表示级联 l 中节点 i 的状态
        X_np = np.array(Cascades)
        self.num_cascades = X_np.shape[0]
        self.register_buffer('X', torch.tensor(X_np, dtype=torch.float32))

        # =================================================================
        # 预计算 2: 实例划分掩码矩阵 M (用于计算正则化 Omega)
        # =================================================================
        unique_insts = list(set(InstancePartition.values()) if isinstance(InstancePartition, dict) else set(InstancePartition))
        num_insts = len(unique_insts)
        M = torch.zeros(N, num_insts, dtype=torch.float32)
        
        for u in range(N):
            inst_id = InstancePartition[u] if isinstance(InstancePartition, dict) else InstancePartition[u]
            idx = unique_insts.index(inst_id)
            M[u, idx] = 1.0
            
        self.register_buffer('M_matrix', M)
        
        counts = M.sum(dim=0).unsqueeze(1)
        counts[counts == 0] = 1.0 
        self.register_buffer('M_counts', counts)

    def _get_prob_matrix(self):
        A_prob = torch.sigmoid(self.A_param)
        # 屏蔽对角线并应用剪枝网络
        A_prob = A_prob * (1.0 - torch.eye(self.N, device=A_prob.device))
        A_prob = A_prob * self.prune_network_tensor
        return A_prob

    def forward(self):
        A_prob = self._get_prob_matrix()
        eps = 1e-8
        
        # =================================================================
        # 1. 负对数似然 (NLL) 高速张量计算 (对应公式推导)
        # =================================================================
        # w_ij = -log(1 - alpha_ij)
        W = -torch.log(1.0 - A_prob + eps)
        
        # 计算 y_i^l = \sum_j x_j^l w_ij
        # X 形状 (L, N)，W.T 形状 (N, N)，结果 Y 形状 (L, N)
        Y = torch.matmul(self.X, W.T)
        
        # 计算 Loss = sum [ (1 - X) * Y  -  X * log(1 - e^{-Y}) ]
        # 项 1: -(1 - x_i^l)y_i^l 的相反数
        term1 = (1.0 - self.X) * Y
        
        # 项 2: x_i^l * log(1 - e^{-y_i^l}) 的相反数
        term2 = self.X * torch.log(1.0 - torch.exp(-Y) + eps)
        
        # 对所有级联和节点求和，并除以总数做归一化，防止 loss 爆掉
        NLL = torch.sum(term1 - term2) / (self.num_cascades * self.N)
        
        # =================================================================
        # 2. 高速计算 Regularization (Omega)
        # =================================================================
        A_sum = torch.matmul(self.M_matrix.T, A_prob)
        A_mean = A_sum / self.M_counts 
        A_approx = torch.matmul(self.M_matrix, A_mean) 
        Omega = torch.sum((A_prob - A_approx)**2) / (self.N * self.N) 
        
        # 3. 返回最终的损失
        return NLL + self.gamma * Omega


def post_processing(estimated_A, beta=1.0):
    """
    beta: Recall 偏置系数。
    beta > 1.0 会让模型更厌恶漏报(FN)，从而降低阈值提高 Recall。
    """
    thresholds = np.linspace(start=1e-6, stop=0.5, num=10000)
    FP_FN_diff = np.zeros([len(thresholds)])

    for i, t in enumerate(thresholds):
        # 估计的漏报项 (本应是边但被阈值切掉了)
        predicted_FN = np.sum(estimated_A[estimated_A < t])
        # 估计的误报项 (本不该是边但被保留了)
        predicted_FP = np.sum(1.0 - estimated_A[estimated_A >= t])

        # 修改点：通过权重 beta 强迫模型降低阈值
        # 当 beta=2.0 时，1个 FN 的代价等于 2个 FP
        FP_FN_diff[i] = np.abs(predicted_FP - beta * predicted_FN)

    best_t = thresholds[np.argmin(FP_FN_diff)]
    
    IG_mat = np.zeros_like(estimated_A)
    IG_mat[estimated_A >= best_t] = 1
    IG = nx.from_numpy_array(IG_mat)

    return best_t, IG

def post_processing_kneed(estimated_A):
    # 1. 获取排序后的概率
    probs = np.sort(estimated_A.flatten())[::-1]
    probs = probs[probs > 1e-5]
    x = np.arange(len(probs))
    
    # 2. 调用 KneeLocator
    # curve='convex': 曲线是凸的
    # direction='decreasing': 曲线是递减的
    kneedle = KneeLocator(x, probs, S=1.0, curve='convex', direction='decreasing')
    
    # 3. 获取拐点对应的索引和阈值
    best_idx = kneedle.knee # 拐点的索引
    if best_idx is None:
        best_t = 0.05 # 备选保守阈值
    else:
        best_t = probs[best_idx]
        
    # 4. 绘图展示 (可选，调试时非常有用)
    # kneedle.plot_knee() 
    
    IG_mat = (estimated_A >= best_t).astype(float)
    return best_t, nx.from_numpy_array(IG_mat)

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
    
    print(TP,FP,FN)

    P = TP / (TP+FP)
    R = TP / (TP+FN)

    return round(P,3),round(R,3),round(2*P*R / (P+R),3)

def run_torch_version(G, N, S, C, gamma, prune_network, iterations=500, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    np.fill_diagonal(prune_network, 0.0)
    
    # 模型初始化时所有的预计算矩阵会被创建
    model = RegularizedInferenceIC(N=N, 
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
    
    # np.savetxt('A_star_matrix.csv', A_star, delimiter=',')
    # np.savetxt('prune_network.csv', prune_network, delimiter=',')
    best_t, IG = post_processing(A_star)
    P, R, F1 = calculate_F1(IG, G)
    print(f"BEP point : {best_t:.5f} | P: {P}, R: {R}, F1: {F1}")
    
    return calculate_F1(IG, G)