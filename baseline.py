import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from utils.plot_utils import true_edge, spur_edge, fals_edge, miss_edge
import bnlearn as bn
from causallearn.search.ConstraintBased.CDNOD import cdnod
from baselines.FL_FedCDH.mycausallearn.utils.data_utils import get_cpdag_from_cdnod, get_dag_from_pdag
from causallearn.utils.cit import fisherz
from causallearn.search.ConstraintBased.FCI import fci
import time
from cdt.causality.graph import GIES


def read_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="asia")
    parser.add_argument("--folder", type=str, default="m3_d1_n10")
    parser.add_argument("--output", type=str, default="res.csv")
    parser.add_argument("--baseline", type=str, choices=["PC", "GES", "FCI", "GIES", "CDNOD", "Chow-Liu"], default="res.csv")
    options = vars(parser.parse_args())
    return options


def load_data(options):
    dataname = options["dataname"]
    folder = options["folder"]

    folderpath = f"./data/distributed/{dataname}/{folder}"
    groundtruth = np.loadtxt(f"./data/distributed/{dataname}/adj.txt")

    silos = []
    if not Path(folderpath).exists():
        print("Folder", folderpath, "not exist!")
        exit()
    
    for file in tqdm(sorted(os.listdir(folderpath))):
        filename = os.path.join(folderpath, file)
        silo_data = pd.read_csv(filename)
        silos.append(silo_data)
        all_vars = silos[0].columns

    all_vars = list(all_vars)
    return all_vars, groundtruth, silos


if __name__ == "__main__":
    options = read_opts()
    all_vars, groundtruth, silos = load_data(options)
    data = pd.concat(silos, axis=0)
    start = time.time()
    
    if options['baseline'] == "PC":
        model = bn.structure_learning.fit(data, methodtype='cs', verbose=0)
        adj_mtx = np.zeros([len(all_vars), len(all_vars)])
        for edge in model['dag_edges']:     # type:ignore
            source, target = edge
            source_id = int(source[1:]) - 1
            target_id = int(target[1:]) - 1
            adj_mtx[source_id][target_id] = 1
    
    
    if options['baseline'] == "Chow-Liu":
        basis_index = [i for i in range(groundtruth.shape[0]) if np.sum(groundtruth[:,i]) == 0]
        sources = np.array(all_vars)[np.array(basis_index)].tolist()
        dfhot, dfnum = bn.df2onehot(data)
        model = bn.structure_learning.fit(dfnum, methodtype='cl', verbose=0, root_node=sources[0])
        adj_mtx = model['adjmat'].to_numpy() * 1.0      # type:ignore
    
    
    elif options['baseline'] == "CDNOD":
        c_indx = []
        for i in range(len(silos)):
            data = silos[i]
            c_indx += [i+1] * len(data)
        c_indx = np.array(c_indx).reshape(len(silos)*len(silos[0]), 1)

        cg = cdnod(data.to_numpy(), c_indx, 0.05, fisherz)
        est_graph = cg.G.graph[0:len(all_vars), 0:len(all_vars)]
        est_cpdag = get_cpdag_from_cdnod(est_graph) # est_graph[i,j]=-1 & est_graph[j,i]=1  ->  est_graph_cpdag[i,j]=1
        adj_mtx = get_dag_from_pdag(est_cpdag) # return a DAG from a PDAG in causaldag
        
    
    elif options["baseline"] == "FCI":
        g, edges = fci(data.to_numpy())
        adj_mtx = np.zeros([len(all_vars), len(all_vars)])
        for edge in edges:     # type:ignore
            source_id = int(edge.get_node1().get_name()[1:]) - 1
            target_id = int(edge.get_node2().get_name()[1:]) - 1
            
            if edge.get_numerical_endpoint1() == -1:
                adj_mtx[source_id][target_id] = 1
            elif edge.get_numerical_endpoint1() != 2:
                adj_mtx[target_id][source_id] = 1
    
    
    elif options["baseline"] == "GES":
        model = bn.structure_learning.fit(data, methodtype='hc', verbose=0)
        adj_mtx = model['adjmat'].to_numpy() * 1.0      # type:ignore
        
        
    elif options['baseline'] == "GIES":
        obj = GIES()
        output = obj.predict(data)
        for edge in output.edges:     # type:ignore
            source, target = edge
            source_id = int(source[1:]) - 1
            target_id = int(target[1:]) - 1
            adj_mtx[source_id][target_id] = 1
            
            if adj_mtx[target_id][source_id] == 1:
                rand_num = np.random.rand()
                if rand_num > 0.5:
                    adj_mtx[source_id][target_id] = 0
                    adj_mtx[target_id][source_id] = 1
                else:
                    adj_mtx[source_id][target_id] = 1
                    adj_mtx[target_id][source_id] = 0
    
    
    finish = time.time()
    
    etrue = true_edge(groundtruth, adj_mtx)
    espur = spur_edge(groundtruth, adj_mtx)
    efals = fals_edge(groundtruth, adj_mtx)
    emiss = miss_edge(groundtruth, adj_mtx)

    f = open(options["output"], "a")
    f.write("{},{},{},{},{},{},{},{}\n".format(
        options['dataname'], options['folder'], options['baseline'], len(etrue), len(espur), len(emiss), len(efals), finish - start
    ))
    
    f.close()
    print("Writting results done!")