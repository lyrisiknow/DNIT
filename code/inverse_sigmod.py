import time
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import tqdm

from utils import calculate_F1, post_processing, not_infected_matrix

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


def run_torch_version(A, S, iterations = 500, prune_network = None):

    G = nx.from_numpy_array(A)
    M, N = S.shape
    # prune
    np.fill_diagonal(prune_network, 0.0)
    not_infected = not_infected_matrix(S)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch_solver(N=N, S=S, non_infected=not_infected, prune_network=prune_network).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    st = time.time()
    for _ in tqdm.tqdm(range(iterations)):
        optimizer.zero_grad()
        log_likelihood = model()
        log_likelihood.backward()
        optimizer.step()

    end_time = time.time() - st
    A = model.diag_zero(torch.sigmoid(model.A.cpu().detach()))
    A = A.numpy()
    A = A * prune_network
    A[A <= 1e-5] = 0.0
    best_t, IG = post_processing(A)
    print("RUN TIME : ", end_time, "BEP point : ", best_t, "P , R,  F1 : ", calculate_F1(IG, G))


