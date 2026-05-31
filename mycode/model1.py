import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import copy
import time

# =================================================================
# 1. 核心模型：基于因果推断去交叠的 IC 推断网络
# =================================================================
class CausalInferenceIC(nn.Module):
    def __init__(self, N, Cascades, InstancePartition, gamma, l1_lambda, prune_network):
        super(CausalInferenceIC, self).__init__()
        
        self.N = N
        self.gamma = gamma
        self.l1_lambda = l1_lambda 
        
        # 剪枝网络处理 (防止对数或除法中的 0)
        prune_network = prune_network.copy()
        prune_network[prune_network == 0] = 1e-5
        self.register_buffer('prune_network_tensor', torch.from_numpy(prune_network).float())

        # ---------------------------------------------------------
        # 预计算 1: 级联矩阵与参数逆向初始化
        # ---------------------------------------------------------
        X_np = np.array(Cascades)
        self.register_buffer('X', torch.tensor(X_np, dtype=torch.float32))

        # Inverse Softplus 初始化，加速收敛
        co_occurrence = np.dot(X_np.T, X_np) / (X_np.shape[0] + 1e-8)
        scaled_co = np.clip(co_occurrence * 0.5, 1e-4, 0.99)
        init_val = np.log(np.exp(scaled_co) - 1.0)
        self.A_param = nn.Parameter(torch.from_numpy(init_val).float())

        # ---------------------------------------------------------
        # 预计算 2: 社区掩码矩阵 M (用于计算 Omega 社区平滑)
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 预计算 3 [因果推断核心]: 倾向得分 (Propensity Score)
        # ---------------------------------------------------------
        # 计算每个节点的基础活跃频率 (混淆变量)
        node_freq = X_np.mean(axis=0)
        propensity = np.clip(node_freq, 1e-4, 0.95)
        
        # [因果组件 A]：IPW (Inverse Propensity Weight) 权重
        ipw = 1.0 / propensity
        ipw = ipw / ipw.mean() # 归一化以维持整体 Loss 规模
        self.register_buffer('ipw_weights', torch.tensor(ipw, dtype=torch.float32).unsqueeze(0))
        
        # [因果组件 B]：基于共现预期的因果惩罚矩阵
        # E_ij = e(i) * e(j)
        E_matrix = np.outer(propensity, propensity)
        self.register_buffer('causal_penalty_matrix', torch.tensor(E_matrix, dtype=torch.float32))

    def _get_prob_matrix(self):
        W = self._get_weights()
        return -torch.expm1(-W)
    
    def _get_weights(self):
        W = F.softplus(self.A_param)
        
        # 可选：如果推断的底层网络是强无向图，可以取消下方注释强制对称化
        # W = (W + W.T) / 2.0 
        
        W = W * (1.0 - torch.eye(self.N, device=W.device))
        W = W * self.prune_network_tensor
        return W

    def forward(self):
        eps = 1e-8
        W = self._get_weights()
        
        # 1. 基础 NLL 计算
        Y = torch.matmul(self.X, W.T)
        term1 = (1.0 - self.X) * Y
        log_prob_active = torch.log(-torch.expm1(-Y) + eps)
        term2 = self.X * log_prob_active
        
        # 【因果革新 1】：IPW-NLL 损失加权
        node_level_loss = term1 - term2 
        weighted_loss = node_level_loss * self.ipw_weights
        NLL = torch.sum(weighted_loss) / (self.X.shape[0] * self.N)
        
        # 2. 社区规整化
        A_prob = -torch.expm1(-W)
        A_sum = torch.matmul(self.M_matrix.T, A_prob)
        A_mean = A_sum / self.M_counts 
        A_approx = torch.matmul(self.M_matrix, A_mean) 
        Omega = torch.sum((A_prob - A_approx)**2) / (self.N * self.N) 
        
        # 【因果革新 2】：Causal L1 Penalty (基于混淆变量的稀疏化)
        L1_causal = torch.sum(W * self.causal_penalty_matrix) / (self.N * self.N)
        
        return NLL + (self.gamma * Omega) + (self.l1_lambda * L1_causal)


# =================================================================
# 2. 评估与后处理函数
# =================================================================
def calculate_continuous_auc(A_pred, G):
    nodes = sorted(G.nodes())
    adj_true = nx.to_numpy_array(G, nodelist=nodes)
    iu = np.triu_indices(len(nodes), k=1) 
    y_true = (adj_true[iu] > 0).astype(int)
    y_scores = A_pred[iu]
    
    if len(np.unique(y_true)) < 2: 
        return 0.5, 0.0
        
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    print(f"Continuous ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    return round(roc_auc, 4), round(pr_auc, 4)

def post_processing_with_community(estimated_A, node_communities, beta=1.0):
    N = estimated_A.shape[0]
    comm_labels = np.array([node_communities[i] for i in range(N)])
    same_comm_mask = (comm_labels[:, None] == comm_labels[None, :])
    np.fill_diagonal(same_comm_mask, False)
    diff_comm_mask = ~same_comm_mask
    np.fill_diagonal(diff_comm_mask, False)

    mean_inner = np.mean(estimated_A[same_comm_mask])
    mean_outer = np.mean(estimated_A[diff_comm_mask]) + 1e-9
    advantage_ratio = np.clip(mean_inner / mean_outer, 1.0, 5.0) 

    thresholds = np.linspace(1e-6, 0.9, 2000)

    beta_inner = beta * advantage_ratio
    diff_inner = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        pred_fn = np.sum(estimated_A[same_comm_mask & (estimated_A < t)])
        pred_fp = np.sum(1.0 - estimated_A[same_comm_mask & (estimated_A >= t)])
        diff_inner[i] = np.abs(pred_fp - beta_inner * pred_fn)
    t_inner = thresholds[np.argmin(diff_inner)]

    diff_outer = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        pred_fn = np.sum(estimated_A[diff_comm_mask & (estimated_A < t)])
        pred_fp = np.sum(1.0 - estimated_A[diff_comm_mask & (estimated_A >= t)])
        diff_outer[i] = np.abs(pred_fp - beta * pred_fn)
    t_outer = thresholds[np.argmin(diff_outer)]

    print(f"Community Advantage Ratio: {advantage_ratio:.2f}")
    print(f"Optimized Thresholds -> Inner: {t_inner:.5f} | Outer: {t_outer:.5f}")

    IG_mat = np.zeros_like(estimated_A)
    IG_mat[same_comm_mask & (estimated_A >= t_inner)] = 1
    IG_mat[diff_comm_mask & (estimated_A >= t_outer)] = 1
    
    return (t_inner + t_outer) / 2, nx.from_numpy_array(IG_mat)

def calculate_F1(IG, G):
    nodes = sorted(G.nodes())
    adj_predict = nx.to_numpy_array(IG, nodelist=nodes)
    adj_true = nx.to_numpy_array(G, nodelist=nodes)
    
    iu = np.triu_indices(len(nodes), k=1)
    y_true = (adj_true[iu] > 0).astype(int)
    y_pred = adj_predict[iu].astype(int)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)
    
    print(f"Binarized Stats -> TP: {int(TP)}, FP: {int(FP)}, FN: {int(FN)}")
    print(f"Precision: {P:.3f} | Recall: {R:.3f} | F1-Score: {F1:.3f}")
    return round(P, 3), round(R, 3), round(F1, 3)

# =================================================================
# 3. 主训练循环 (Full-Batch Gradient Descent)
# =================================================================
def run_torch_version(G, N, S, C, A, gamma, prune_network, iterations=1000, lr=0.01, l1_lambda=0.001):
    patience = 100
    best_loss = float('inf')
    counter = 0 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Using Causal Full-Batch Gradient Descent")

    # 初始化模型
    model = CausalInferenceIC(N=N, Cascades=S, InstancePartition=C, 
                              gamma=gamma, l1_lambda=l1_lambda, 
                              prune_network=prune_network).to(device)
                                   
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) 
    best_model_wts = copy.deepcopy(model.state_dict())

    model.train()
    pbar = tqdm(range(iterations), desc="Optimizing")
    st = time.time()

    for i in pbar:
        optimizer.zero_grad()
        
        # 全量前向传播
        loss = model() 
        loss.backward() 
        
        # 梯度裁剪防爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() 
        
        curr_loss = loss.item()
        
        # Early Stopping 逻辑
        if curr_loss < best_loss - 1e-5:
            best_loss = curr_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            
        if i % 10 == 0:
            pbar.set_postfix({"Loss": f"{curr_loss:.4f}", "Best": f"{best_loss:.4f}", "Patience": f"{counter}/{patience}"})

        if counter >= patience:
            tqdm.write(f"Early stopping at iteration {i+1}. Recovering best weights...")
            break
            
    end_time = time.time() - st
    
    # 恢复最佳权重
    model.load_state_dict(best_model_wts)

    # =================================================================
    # 4. 评估与输出
    # =================================================================
    model.eval()
    with torch.no_grad():
        A_star = model._get_prob_matrix().cpu().numpy()

    # 剔除剪枝节点与自环
    A_star = A_star * (prune_network > 0).astype(float)
    np.fill_diagonal(A_star, 0.0)

    print("\n--- Evaluation Results ---")
    # 1. 连续概率指标 (真实反映模型排序能力的核心)
    continuous_metrics = calculate_continuous_auc(A_star, G)
    
    # 2. 社区感知二值化切分
    best_t, IG = post_processing_with_community(A_star, C, beta=1.0) 
    
    # 3. 打印二值化后的图指标
    f1 = calculate_F1(IG, G)
    
    return continuous_metrics, end_time, f1