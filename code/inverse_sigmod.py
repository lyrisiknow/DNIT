import time
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import tqdm

from utils import calculate_F1, post_processing, not_infected_matrix, result_record, calculate_binary_auc

class torch_solver(nn.Module):

    def __init__(self, N, S, non_infected, prune_network, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.N = N
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.S = torch.tensor(S).to(self.device)
        self.non_infected = torch.tensor(non_infected).to(self.device)

        A = torch.ones([N, N]) * 0.2

        if prune_network is not None:
            prune_network[prune_network == 0] = 1e-5
            A = A * prune_network

        A = -torch.log(1.0 / A - 1.0)
        A = self.diag_zero(A).to(self.device)
        self.register_parameter('A', nn.Parameter(A, requires_grad=True))

    def diag_zero(self, A):
        diag_A = torch.diag(A)
        A = A - torch.diag_embed(diag_A)
        return A

    def forward(self):
        M, N = self.S.shape
        l = 0.0
        for s_i in range(M):
            s_i_1 = torch.where(self.S[s_i] == 1)[0]

            sub_A = self.A[s_i_1, :]
            sub_A = sub_A[:, s_i_1]

            sub_A = torch.sigmoid(sub_A)
            W = self.create_W(sub_A)
            W[0, :] = 1.0  # augmented
            l += torch.logdet(W)

        l += torch.sum(self.non_infected * torch.log(1.0 - self.diag_zero(torch.sigmoid(self.A)) + 1e-10))

        return -l

    def create_W(self, sub_A):
        sub_A = self.diag_zero(sub_A)
        W = -sub_A
        for i in range(sub_A.shape[0]):
            W[i, i] = torch.sum(sub_A[:, i])

        return W


def run_torch_version(A, S, iterations=500, prune_network=None, patience=10, min_delta=1e-4):
    G = nx.from_numpy_array(A)
    M, N = S.shape
    np.fill_diagonal(prune_network, 0.0)
    not_infected = not_infected_matrix(S)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch_solver(N=N, S=S, non_infected=not_infected, prune_network=prune_network).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    
    # --- 早停机制初始化 ---
    best_loss = float('inf')
    counter = 0
    st = time.time()
    
    pbar = tqdm.tqdm(range(iterations))
    for i in pbar:
        optimizer.zero_grad()
        # 假设 model() 返回的是负对数似然 (Negative Log-Likelihood)
        # 如果 model() 返回的是正的似然，请将其取负数变为 Loss
        loss = model() 
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        
        # --- 早停逻辑判断 ---
        # 检查 loss 是否有足够明显的下降
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            counter = 0  # 重置计数器
        else:
            counter += 1 # 损失没有明显改善
        
        # 更新 tqdm 进度条显示当前 Loss
        pbar.set_postfix(loss=f"{current_loss:.4f}", patience=f"{counter}/{patience}")

        if counter >= patience:
            print(f"\nEarly stopping at iteration {i}. Loss has not improved for {patience} epochs.")
            break

    end_time = time.time() - st
    # 后处理逻辑保持不变
    A_final = model.diag_zero(torch.sigmoid(model.A.cpu().detach())).numpy()
    A_final = A_final * prune_network
    A_final[A_final <= 1e-5] = 0.0
    
    best_t, IG = post_processing(A_final)
    print("RUN TIME : ", end_time, "BEP point : ", best_t, "P , R, F1 : ", calculate_F1(IG, G))
    
    return calculate_binary_auc(IG, G), end_time, calculate_F1(IG,G)


