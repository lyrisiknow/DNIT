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

    l = set()
    for node in node_communities:
        l.add(node_communities[node])
    print(len(l))