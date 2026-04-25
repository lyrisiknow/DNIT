import numpy as np
from scipy.special import gammaln
import itertools
import math
from itertools import combinations
from sklearn.cluster import KMeans
import math
import networkx as nx
from tqdm import tqdm
from utils import generate_infections, result_record, calculate_F1, IC, Neighbour_finder
from SIDN import infer_sidn_network
from TENDS import tends_algorithm
from TWIND import run_twind_fast
from k_lifts import estimate_lifts, k_lifts_algorithm
from PIND import pind_inference

if __name__ == '__main__':
    # --- 设定参数 ---
    edges = set()
    data_path = '../dataset/email-Eu-core/'
    with open(data_path+'email-Eu-core.txt', 'r') as f:
        for l in f:
            if l.strip() != '':
                edges.add((int(l.strip().split(' ')[0]), int(l.strip().split(' ')[1])))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    N = len(G)
    node_communities = {}
    
    with open(data_path + 'email-Eu-core-department-labels.txt', 'r') as f:
        for l in f:
            if l.strip() != '':
                node_communities[int(l.strip().split(' ')[0])] = l.strip().split(' ')[1]

    A = nx.to_numpy_array(G)
    P = np.zeros((N, N))
    for u, v in G.edges():
        if node_communities[u] == node_communities[v]:
            weight = np.random.uniform(0.05, 0.1)
        else:
            weight = np.random.uniform(0.01, 0.05)
        P[u, v] = weight
        P[v, u] = weight
    A = A * P
    S = generate_infections(A, num_sim=500)
    G = nx.from_numpy_array(A)
    print('TWIND:')
    IG = nx.DiGraph()
    IG.add_edges_from(run_twind_fast(S))
    result_record("TWIND", calculate_F1(IG, G), "email", param='s500')
    
    print('TENDS:')
    IG = nx.DiGraph()
    IG.add_edges_from(tends_algorithm(N, S))
    result_record("TENDS", calculate_F1(IG, G), "email", param='s500')
    
    print('SIDN:')
    IG = nx.DiGraph()
    IG = nx.from_numpy_array(infer_sidn_network(S), create_using=nx.DiGraph)
    result_record("SIDN", calculate_F1(IG, G), "email", param='s500')
    
    print('PIND:')
    IG = nx.DiGraph()
    IG = nx.from_numpy_array(pind_inference(S), create_using=nx.DiGraph)
    result_record("PIND", calculate_F1(IG, G), "email", param='s500')