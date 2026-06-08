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
# 1. 核心模型：消融掉社区平滑的因果推断网络
# =================================================================
class CausalInferenceIC(nn.Module):
    # 【消融修改】：去除了 InstancePartition 和 gamma 参数
    def __init__(self, N, Cascades, l1_lambda, prune_network):
        super(CausalInferenceIC, self).__init__()
        
        self.N = N
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

        # 【消融修改】：删除了 预计算 2 (社区掩码矩阵 M 相关的初始化代码) 节约显存

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
        
        # 【消融修改】：删除了基于 M_matrix 计算 Omega 的部分
        
        # 【因果革新 2】：Causal L1 Penalty (基于混淆变量的稀疏化)
        L1_causal = torch.sum(W * self.causal_penalty_matrix) / (self.N * self.N)
        
        # 【消融修改】：仅保留 NLL 和 因果 L1 惩罚
        return NLL + (self.l1_lambda * L1_causal)


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

# 【消融修改】：替换为无社区先验的全局二值化后处理
def post_processing_global(estimated_A, beta=1.0):
    # 忽略对角线（自环）的影响进行统计
    mask = ~np.eye(estimated_A.shape[0], dtype=bool)
    valid_probs = estimated_A[mask]
    
    thresholds = np.linspace(1e-6, 0.9, 2000)
    diff = np.zeros(len(thresholds))
    
    # 全局寻找最小化 FP 和 beta * FN 差异的阈值
    for i, t in enumerate(thresholds):
        pred_fn = np.sum(valid_probs[valid_probs < t])
        pred_fp = np.sum(1.0 - valid_probs[valid_probs >= t])
        diff[i] = np.abs(pred_fp - beta * pred_fn)
        
    best_t = thresholds[np.argmin(diff)]
    
    print(f"Global Optimized Threshold: {best_t:.5f}")

    # 生成二值化推断图
    IG_mat = np.zeros_like(estimated_A)
    IG_mat[estimated_A >= best_t] = 1
    
    return best_t, nx.from_numpy_array(IG_mat)

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
# 【说明】：为了防止破坏你外部调用函数的代码，参数签名保留了 C (社区划分) 和 gamma，但内部并未被使用。
def run_torch_version(G, N, S, C, A, gamma, prune_network, iterations=1000, lr=0.01, l1_lambda=0.001):
    patience = 300
    best_loss = float('inf')
    counter = 0 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Using Causal Full-Batch Gradient Descent (Ablation: No Community)")

    # 【消融修改】：初始化模型时去除了不必要的 C 和 gamma 参数
    model = CausalInferenceIC(N=N, Cascades=S, 
                              l1_lambda=l1_lambda, 
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
    
    # 【消融修改】：替换为无社区信息的全局二值化切分
    best_t, IG = post_processing_global(A_star, beta=1.0) 
    
    # 3. 打印二值化后的图指标
    f1 = calculate_F1(IG, G)
    
    return continuous_metrics, end_time, f1