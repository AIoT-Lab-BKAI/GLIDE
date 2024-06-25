import pandas as pd
from baselines.notears.notears import utils as ut
import numpy as np
from causallearn.search.ConstraintBased.CDNOD import cdnod
from baselines.FL_FedCDH.mycausallearn.utils.data_utils import get_cpdag_from_cdnod, get_dag_from_pdag
from causallearn.utils.cit import fisherz
from utils.plot_utils import true_edge, spur_edge, fals_edge, miss_edge
from time import time

def evaluate(B_true, adj_mtx):
    etrue = true_edge(B_true, adj_mtx)
    espur = spur_edge(B_true, adj_mtx)
    efals = fals_edge(B_true, adj_mtx)
    emiss = miss_edge(B_true, adj_mtx)
    return len(etrue), len(espur), len(emiss), len(efals)

summary_csv = pd.DataFrame(columns=["e", "n", "d", "s", "graph", "dtype", "baseline", "etrue", "espur", "emiss", "efals", "time"])


for j in [0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3]:#, 4, 5]:
    print(j, end=": ")
    for run in range(10):
        start = time()
        ut.set_random_seed(2025)
        n, d, s0, graph_type, sem_type = 1000, int(100 * j), int(100 * j), 'ER', 'logistic'
        B_true = ut.simulate_dag(d, s0, graph_type)
        W_true = ut.simulate_parameter(B_true)

        n_env = 10
        datas = []
        for i in range(n_env):
            ut.set_random_seed(i**2 + 2025 * run + 2024)
            X = ut.simulate_linear_sem(W_true, n, sem_type)
            datas.append(X)
        all_vars = [f'X{i+1}' for i in range(d)]
        
        c_indx = np.repeat(range(1, n_env + 1), n).reshape(-1, 1).astype(float)
        cg = cdnod(np.concatenate(datas), c_indx, 0.05, fisherz, show_progress=False, verbose=False)
        est_graph = cg.G.graph[0:len(all_vars), 0:len(all_vars)]
        est_cpdag = get_cpdag_from_cdnod(est_graph) # est_graph[i,j]=-1 & est_graph[j,i]=1  ->  est_graph_cpdag[i,j]=1
        est_dag_from_pdag = get_dag_from_pdag(est_cpdag) # return a DAG from a PDAG in causaldag
        adj_mtx = get_dag_from_pdag(est_cpdag) # return a DAG from a PDAG in causaldag
        end = time()
        
        etrue, espur, emiss, efals = evaluate(B_true, adj_mtx)
        summary_csv.loc[-1] = [n_env, n, d, s0, graph_type, sem_type, "CDNOD", etrue, espur, emiss, efals, end-start]
        summary_csv.index = summary_csv.index + 1
        summary_csv = summary_csv.sort_index()
        print(run, end=" ")
    print()
    
summary_csv.to_csv("res/CDNOD-federated.csv", index=False)
    
    