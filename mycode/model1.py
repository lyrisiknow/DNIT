import torch
import torch.nn as nn
import torch.optim as optim
from torch.linalg import inv, slogdet
import numpy as np
import networkx as nx
from kneed import KneeLocator
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        self.register_buffer('X', torch.tensor(X_np, dtype=torch.float32))
        
        #+++2
        co_occurrence = np.dot(X_np.T, X_np)
        # 归一化，避免数值过大
        co_occurrence = co_occurrence / (X_np.shape[0] + 1e-8)
        # 将共现频率映射到参数空间
        # 因为后面用 softplus，所以这里可以做个逆映射或简单的线性缩放
        # 我们希望共现高的边，初始 w 更大
        init_weight = torch.from_numpy(co_occurrence).float() * 0.5
        self.A_param = nn.Parameter(init_weight)

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
    
    #+++2
    def _get_weights(self):
        # =================================================================
        # 优化 2: 直接建模 w (Softplus)
        # softplus(x) = log(1 + exp(x))，保证 w > 0 且在大值区梯度不消失
        # =================================================================
        W = F.softplus(self.A_param)
        
        # 应用剪枝和对角线屏蔽
        W = W * (1.0 - torch.eye(self.N, device=W.device))
        W = W * self.prune_network_tensor
        return W

    def forward(self):
        #+++2
        # A_prob = self._get_prob_matrix()
        eps = 1e-8
        
        # =================================================================
        # 1. 负对数似然 (NLL) 高速张量计算 (对应公式推导)
        # =================================================================
        # w_ij = -log(1 - alpha_ij)
        #+++2
        W = self._get_weights()
        A_prob = -torch.expm1(-W)
        
        # 计算 y_i^l = \sum_j x_j^l w_ij
        # X 形状 (L, N)，W.T 形状 (N, N)，结果 Y 形状 (L, N)
        Y = torch.matmul(self.X, W.T)
        
        # 计算 Loss = sum [ (1 - X) * Y  -  X * log(1 - e^{-Y}) ]
        # 项 1: -(1 - x_i^l)y_i^l 的相反数
        term1 = (1.0 - self.X) * Y
        
        #+++3
        denominator = Y + eps
        log_prob_active = torch.log(-torch.expm1(-Y) + eps)
        term2 = self.X * log_prob_active
        # 项 2: x_i^l * log(1 - e^{-y_i^l}) 的相反数
        # term2 = self.X * torch.log(1.0 - torch.exp(-Y) + eps)
        
        #+++1
        negative_mask = ((1.0 - self.X) * Y > 0).float()
        # 对负样本进行随机下采样 (假设只保留 10% 的负样本惩罚)
        sampling_rate = 0.1
        random_mask = (torch.rand_like(Y) < sampling_rate).float()
        effective_negative_mask = negative_mask * random_mask
        term1_sampled = term1 * effective_negative_mask
        
        # 对所有级联和节点求和，并除以总数做归一化，防止 loss 爆掉
        NLL = torch.sum(term1_sampled - term2) / (self.N * self.N)
        
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

def post_processing_with_community(estimated_A, node_communities, beta=1.0):
    N = estimated_A.shape[0]
    comm_labels = np.array([node_communities[i] for i in range(N)])
    same_comm_mask = (comm_labels[:, None] == comm_labels[None, :])
    np.fill_diagonal(same_comm_mask, False)
    diff_comm_mask = ~same_comm_mask
    np.fill_diagonal(diff_comm_mask, False)

    # --- 1. 数据驱动：计算社区先验优势 (Advantage Ratio) ---
    # 意思是：模型自己认为同社区的边比跨社区活跃多少倍？
    mean_inner = np.mean(estimated_A[same_comm_mask])
    mean_outer = np.mean(estimated_A[diff_comm_mask]) + 1e-9
    # 比如 mean_inner=0.08, mean_outer=0.02，那么 ratio 就是 4.0
    advantage_ratio = np.clip(mean_inner / mean_outer, 1.0, 5.0) 

    thresholds = np.linspace(1e-6, 0.5, 1000)

    # --- 2. 搜索同社区阈值 t_inner (带有偏爱) ---
    # 核心：给同社区的 beta 乘上 advantage_ratio
    # 漏掉一条同社区的边，现在的代价是正常情况的 advantage_ratio 倍
    beta_inner = beta * advantage_ratio
    diff_inner = np.zeros(len(thresholds))
    
    for i, t in enumerate(thresholds):
        pred_fn = np.sum(estimated_A[same_comm_mask & (estimated_A < t)])
        pred_fp = np.sum(1.0 - estimated_A[same_comm_mask & (estimated_A >= t)])
        diff_inner[i] = np.abs(pred_fp - beta_inner * pred_fn)
        
    t_inner = thresholds[np.argmin(diff_inner)]

    # --- 3. 搜索跨社区阈值 t_outer (严苛标准) ---
    # 跨社区保持正常的 beta，没有优待
    diff_outer = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        pred_fn = np.sum(estimated_A[diff_comm_mask & (estimated_A < t)])
        pred_fp = np.sum(1.0 - estimated_A[diff_comm_mask & (estimated_A >= t)])
        diff_outer[i] = np.abs(pred_fp - beta * pred_fn)
        
    t_outer = thresholds[np.argmin(diff_outer)]

    # 打印观察结果（可注释掉）
    print(f"Data-driven Advantage Ratio: {advantage_ratio:.2f}")
    print(f"Inner Threshold: {t_inner:.5f} | Outer Threshold: {t_outer:.5f}")

    # --- 4. 构造最终图 ---
    IG_mat = np.zeros_like(estimated_A)
    IG_mat[same_comm_mask & (estimated_A >= t_inner)] = 1
    IG_mat[diff_comm_mask & (estimated_A >= t_outer)] = 1
    
    IG = nx.from_numpy_array(IG_mat)

    # 保持 return 不变
    best_t = (t_inner + t_outer) / 2
    return best_t, IG

def calculate_metrics_sklearn(A_pred, A_true):
    # sklearn 的输入需要是平铺的向量或矩阵
    # 它会自动处理多维矩阵
    mse = mean_squared_error(A_true, A_pred)
    mae = mean_absolute_error(A_true, A_pred)
    
    return mse, mae


def post_processing_with_community_strict_inner(estimated_A, node_communities, beta=1.0):
    N = estimated_A.shape[0]
    comm_labels = np.array([node_communities[i] for i in range(N)])
    same_comm_mask = (comm_labels[:, None] == comm_labels[None, :])
    np.fill_diagonal(same_comm_mask, False)
    diff_comm_mask = ~same_comm_mask
    np.fill_diagonal(diff_comm_mask, False)

    # --- 1. 数据驱动：计算社区优势 ---
    mean_inner = np.mean(estimated_A[same_comm_mask])
    mean_outer = np.mean(estimated_A[diff_comm_mask]) + 1e-9
    # 依然计算这个比例，反映社区聚集效应的强度
    advantage_ratio = np.clip(mean_inner / mean_outer, 1.0, 5.0) 

    thresholds = np.linspace(1e-6, 0.5, 1000)

    # --- 2. 搜索同社区阈值 t_inner (执行更严格的标准) ---
    # 修改点：将 beta 除以 advantage_ratio
    # 逻辑：减小 beta 意味着我们更讨厌“误报”(FP)，即对同社区的边要求更苛刻
    beta_inner = beta / advantage_ratio 
    
    diff_inner = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        # pred_fn: 漏报（有边没连上）的概率累积
        # pred_fp: 误报（无边强行连）的概率累积
        pred_fn = np.sum(estimated_A[same_comm_mask & (estimated_A < t)])
        pred_fp = np.sum(1.0 - estimated_A[same_comm_mask & (estimated_A >= t)])
        # 此时 beta_inner 变小，公式平衡点会向更大的 t 移动
        diff_inner[i] = np.abs(pred_fp - beta_inner * pred_fn)
        
    t_inner = thresholds[np.argmin(diff_inner)]

    # --- 3. 搜索跨社区阈值 t_outer (保持常规) ---
    diff_outer = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        pred_fn = np.sum(estimated_A[diff_comm_mask & (estimated_A < t)])
        pred_fp = np.sum(1.0 - estimated_A[diff_comm_mask & (estimated_A >= t)])
        diff_outer[i] = np.abs(pred_fp - beta * pred_fn)
        
    t_outer = thresholds[np.argmin(diff_outer)]

    print(f"Data-driven Advantage Ratio: {advantage_ratio:.2f}")
    print(f"Strict Inner Threshold: {t_inner:.5f} | Normal Outer Threshold: {t_outer:.5f}")

    # --- 4. 构造最终图 ---
    IG_mat = np.zeros_like(estimated_A)
    IG_mat[same_comm_mask & (estimated_A >= t_inner)] = 1
    IG_mat[diff_comm_mask & (estimated_A >= t_outer)] = 1
    
    IG = nx.from_numpy_array(IG_mat)
    return (t_inner + t_outer) / 2, IG

def run_torch_version(G, N, S, C, A, gamma, prune_network, iterations=500, lr=0.01):
    patience = 300      # 连续 300 轮 Loss 不下降则早停
    best_loss = float('inf')
    counter = 0         # 早停计数器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on device: {device}")
    np.fill_diagonal(prune_network, 0.0)

    # 1. 模型初始化
    model = RegularizedInferenceIC(N=N, 
                                    Cascades=S, 
                                    InstancePartition=C, 
                                    gamma=gamma,
                                    prune_network=prune_network).to(device)

    # 2. 优化器与调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    # 用于保存表现最好的模型权重
    best_model_wts = copy.deepcopy(model.state_dict())

    model.train()
    pbar = tqdm(range(iterations), desc="Optimizing")

    for i in pbar:
        optimizer.zero_grad()
        loss = model() 
        loss.backward() 
        
        # 可选：梯度裁剪，防止 NLL 导致的梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step() 
        
        curr_loss = loss.item()
        
        # --- Early Stopping 逻辑 ---
        # 如果当前 Loss 有进步 (设定一个微小的阈值 1e-6)
        if curr_loss < best_loss - 1e-6:
            best_loss = curr_loss
            best_model_wts = copy.deepcopy(model.state_dict()) # 记录最佳状态
            counter = 0 # 重置计数器
        else:
            counter += 1
        
        # 更新 tqdm 的后缀显示
        if i % 10 == 0:
            pbar.set_postfix({"Loss": f"{curr_loss:.4f}", "Best": f"{best_loss:.4f}", "Patience": f"{counter}/{patience}"})

        # 定期输出详细信息
        if (i+1) % 1000 == 0:
            tqdm.write(f"Iteration {i+1}, Loss: {curr_loss:.4f}, LR: {optimizer.param_groups[0]['lr']}")

        # 触发早停
        if counter >= patience:
            tqdm.write(f"Early stopping at iteration {i+1}. Recovering best weights...")
            break

    # --- 训练结束，恢复最佳权重 ---
    model.load_state_dict(best_model_wts)

    # --- 评估与后处理 ---
    model.eval()
    with torch.no_grad():
        # 提取估计的概率矩阵
        A_star = model._get_prob_matrix().cpu().numpy()

    # 再次确保对角线和剪枝约束
    A_star = A_star * prune_network
    A_star[A_star <= 1e-5] = 0.0

    # 调用后处理逻辑
    # 建议：如果 FN 依然很高，尝试传入 beta=1.5 或 2.0
    best_t, IG = post_processing_with_community(A_star, C) 
    return calculate_binary_auc(IG, G), calculate_F1(IG, G), calculate_metrics_sklearn(A_star, A)