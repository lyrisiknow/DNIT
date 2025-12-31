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
        self.S = Cascades              # 观测级联数据
        self.C = InstancePartition     # 实例/社区划分
        self.gamma = gamma             # 正则化强度
        
        # 传播概率矩阵 A 的潜在参数 (Pre-sigmoid parameter)
        # 将其设置为 nn.Parameter，PyTorch 才能优化它
        # 初始值可以基于启发式或零初始化
        self.A_param = nn.Parameter(torch.zeros(N, N, dtype=torch.float32))
        
        prune_network[prune_network == 0] = 1e-5
        self.prune_network = torch.from_numpy(prune_network).float()
        # 存储实例信息，用于正则化项计算
        self.Instance_Info = self._preprocess_instance_info()

    def _get_prob_matrix(self):
        """将潜在参数 A_param 转换成实际的传播概率矩阵 A (0 <= A <= 1)"""
        # 使用 sigmoid 激活函数保证 A 在 [0, 1] 范围内
        A_prob = torch.sigmoid(self.A_param)
        # 确保节点不能传播给自己 (对角线为零)
        A_prob = A_prob * (1.0 - torch.eye(self.N, device=A_prob.device))
        A_prob = A_prob * self.prune_network
        return A_prob

    def _preprocess_instance_info(self):
        """预处理实例信息，计算 Nu 和 A_avg 的辅助数据结构"""
        # 假设 C 是一个字典 {node_id: instance_id}
        # 找出每个实例的成员
        instance_members = {}
        for u in range(self.N):
            inst_id = self.C[u]
            if inst_id not in instance_members:
                instance_members[inst_id] = []
            instance_members[inst_id].append(u)
            
        return instance_members

    def calculate_log_likelihood(self, A_prob):
        num_cascades = len(self.S)
        N = A_prob.shape[0]
        log_L = torch.tensor(0.0, device=A_prob.device)
        eps = 1e-8 # 防止 log(0)

        for cascade in self.S:
            S_I = np.where(cascade == 1)[0]
            S_O = np.where(cascade == 0)[0]
            
            # --- Term 1: 归一化失败传播项 ---
            S_I_tensor = torch.tensor(S_I, dtype=torch.long, device=A_prob.device)
            S_O_tensor = torch.tensor(S_O, dtype=torch.long, device=A_prob.device)
            
            A_slice = A_prob[S_I_tensor[:, None], S_O_tensor[None, :]]
            
            # 使用 clamp 保证数值在 [eps, 1-eps] 之间
            term1 = torch.sum(torch.log(1.0 - A_slice + eps))
            
            # --- Term 2: Log Det 项 ---
            W_hat_l = self.construct_W_hat(A_prob) 
            sign, log_abs_det = torch.linalg.slogdet(W_hat_l)
            
            # 这里的 log_abs_det 也可以根据该次级联的感染规模进行缩放
            # term2 = log_abs_det / len(S_I) 
            term2 = log_abs_det

            log_L += (term1 + term2)

        # --- 最终归一化 ---
        # 1. 除以级联次数，使 Loss 与模拟次数无关
        # 2. 如果数值依然很大，可以除以 N（节点数）
        return log_L / (num_cascades * N)

    def calculate_regularization(self, A_prob):
        """
        计算正则化项 Omega(A, C)
        Omega(A, C) = Sum_k Sum_{i, j in C_k, i != j} || A_i,. - A_j,. ||^2
        """
        Omega = torch.tensor(0.0, device=A_prob.device)
        
        for inst_id, members in self.Instance_Info.items():
            if len(members) > 1:
                # 提取该实例内部所有节点的传播向量 A_i,.
                A_block = A_prob[members, :]  # 维度: |I_k| x N
                
                # 计算 A_block 的平均向量
                A_mean = torch.mean(A_block, dim=0, keepdim=True) # 1 x N
                
                # 使用中心化矩阵 X = A_block - A_mean
                X = A_block - A_mean
                
                intra_variance = torch.sum(X**2)
                
                Omega += intra_variance
                
        return Omega

    def construct_W_hat(self, A_prob):
        """
        *** 请在这里实现 W_hat 矩阵的构造逻辑 ***
        这个函数是特定于您的模型的。
        它必须返回一个 PyTorch 张量，且是 A_prob 的可微分函数。
        """
        n = A_prob.size(0)
        # 移除对角线元素
        mask = (1.0 - torch.eye(n, device=A_prob.device, dtype=A_prob.dtype))
        sub_A = A_prob * mask
        W = -sub_A
        for i in range(sub_A.shape[0]):
            W[i, i] = torch.sum(sub_A[:, i])

        return W


    def forward(self):
        """
        计算要最小化的目标函数 (负目标函数)
        Minimize: -L(A) = -(log L(A|S) - gamma * Omega(A, C))
                        = -log L(A|S) + gamma * Omega(A, C)
        """
        
        # 1. 转换传播概率矩阵 A
        A_prob = self._get_prob_matrix()
        
        # 2. 计算负对数似然项
        # PyTorch 会自动微分这里的复杂项
        log_L = self.calculate_log_likelihood(A_prob)
        NLL = -log_L # Negative Log Likelihood
        
        # 3. 计算正则化项
        Omega = self.calculate_regularization(A_prob)
        
        # 4. 最终损失函数 (要最小化的值)
        Loss = NLL + self.gamma * Omega
        
        return Loss

def run_torch_version(G, N, S, C, gamma, prune_network, iterations=500, lr=0.01):
    # 1. 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    np.fill_diagonal(prune_network, 0.0)
    # 假设数据 S, C 已经被转换为 PyTorch tensors 或 Python 结构
    model = RegularizedInference(N=N, 
                                Cascades=S, 
                                InstancePartition=C, 
                                gamma=gamma,
                                prune_network=prune_network).to(device)

    # 使用 Adam 优化器优化模型的参数 A_param
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    
    # 2. 训练循环
    for i in tqdm.tqdm(range(iterations)):
        optimizer.zero_grad()
        
        # forward() 计算 Loss = -log L + gamma * Omega
        loss = model() 
        
        # **反向传播：PyTorch 计算所有梯度，包括 log det 项的复杂梯度**
        loss.backward() 
        
        # **优化器步骤：更新参数 A_param**
        optimizer.step() 
        
        print(f"Iteration {i}, Loss: {loss.item():.4f}")

    # 3. 评估和返回结果
    # 将潜在参数 A_param 转换回传播概率矩阵 A*
    A_star = model._get_prob_matrix().cpu().detach().numpy()
    
    A_star = A_star * prune_network
    A_star[A_star <= 1e-5] = 0.0
    
    best_t, IG = post_processing(A_star)
    print( "BEP point : ", best_t, "P , R,  F1 : ", calculate_F1(IG, G))


