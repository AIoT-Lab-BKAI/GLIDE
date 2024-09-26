import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
from utils.plot_utils import true_edge, spur_edge, fals_edge, miss_edge
import bnlearn as bn
from causallearn.search.ConstraintBased.CDNOD import cdnod
from baselines.FL_FedCDH.mycausallearn.utils.data_utils import get_cpdag_from_cdnod, get_dag_from_pdag
from causallearn.utils.cit import fisherz
from causallearn.search.ConstraintBased.FCI import fci
import time
from cdt.causality.graph import GIES
from baselines.notears.notears.linear import notears_linear
from baselines.notears.notears.nonlinear import notears_nonlinear, NotearsMLP
from baselines.DAS.src.modules.algorithms.cd import DAS, SCORE


def read_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="asia")
    parser.add_argument("--folder", type=str, default="m3_d1_n10")
    parser.add_argument("--output", type=str, default="res.csv")
    parser.add_argument("--ntype", type=str, default="linear", choices=["linear", "nonlinear", 
                                                                        "sf_linear", "sf_nonlinear",
                                                                        "bp_linear", "bp_nonlinear"])
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--s", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--baseline", type=str, 
                        choices=["PC", "GES", "FCI", "GIES", "CDNOD", #"ICP", "Causal-de-Finetti",
                                 "Notears", "MLPNotears", "Chow-Liu", "DAS", "SCORE"], 
                        default="PC")
    
    options = vars(parser.parse_args())
    return options


def load_data(options):
    dataname = options["dataname"]
    
    if dataname == 'notears':
        d, ntype = options['d'], options['ntype']
        s = options['s'] if options['s'] is not None else d
        folderpath = f"./data/{dataname}/{ntype}Gaussian/raw/X_{d}_{s}.csv"
        merged_df = pd.read_csv(folderpath, names=[f'X{i}' for i in range(1, d+1)])
        groundtruth = np.loadtxt(f"./data/{dataname}/{ntype}Gaussian/W_true_{d}_{s}.csv", delimiter=',')
        all_vars = list(merged_df.columns)
        
        if not Path(options['output']).exists():
            f = open(options["output"], "w")
            f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                'dataname', 'd', 's', 'baseline',
                'etrue', 'espur', 'emiss', 'efals', 
                'shd', 'tpr','time'))
            f.close()
            
        return merged_df, all_vars, groundtruth
    
    else:
        folder = options["folder"]
        folderpath = f"./data/distributed/{dataname}/{folder}"
        groundtruth = np.loadtxt(f"./data/distributed/{dataname}/adj.txt")

        silos = []
        if not Path(folderpath).exists():
            print("Folder", folderpath, "not exist!")
        else:
            for file in sorted(os.listdir(folderpath)):
                filename = os.path.join(folderpath, file)
                silo_data = pd.read_csv(filename)
                silos.append(silo_data)
                # print("Loaded file:", filename)

        merged_df = pd.concat(silos, axis=0)
        merged_df = merged_df.reindex(sorted(merged_df.columns, key=lambda item: int(item[1:])), axis=1)
        all_vars = list(merged_df.columns)
        
        if not Path(options['output']).exists():
            f = open(options["output"], "w")
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                'dataname', 'folder', 'baseline',
                'etrue', 'espur', 'emiss', 'efals', 
                'shd', 'tpr','time'))
            f.close()
        
        return merged_df, all_vars, groundtruth


if __name__ == "__main__":
    options = read_opts()
    print("Running:", options)
    data, all_vars, groundtruth = load_data(options)
    
    for _ in range(options['repeat']):
        start = time.time()
        
        if options['baseline'] == "PC":
            model = bn.structure_learning.fit(data, methodtype='cs', verbose=0)
            adj_mtx = np.zeros([len(all_vars), len(all_vars)])
            for edge in model['dag_edges']:     # type:ignore
                source, target = edge
                source_id = int(source[1:]) - 1
                target_id = int(target[1:]) - 1
                adj_mtx[source_id][target_id] = 1
                if adj_mtx[target_id][source_id] == 1:
                    adj_mtx[source_id][target_id] = 0
                    adj_mtx[target_id][source_id] = 0
        
        
        if options['baseline'] == "Chow-Liu":
            basis_index = [i for i in range(groundtruth.shape[0]) if np.sum(groundtruth[:,i]) == 0]
            sources = np.array(all_vars)[np.array(basis_index)].tolist()
            dfhot, dfnum = bn.df2onehot(data)
            model = bn.structure_learning.fit(dfnum, methodtype='cl', verbose=0, root_node=sources[0])
            adj_mtx = model['adjmat'].to_numpy() * 1.0      # type:ignore
        
        
        elif options['baseline'] == "CDNOD":  # Do not have proper datasets for this baseline
            num_var = len(data.columns)
            num_env = 10
            num_sample = int(len(data)/num_env)
            c_indx = np.repeat(range(1, num_env + 1), num_sample).reshape(-1, 1).astype(float)

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
            adj_mtx = np.zeros([len(all_vars), len(all_vars)])
            for edge in output.edges:     # type:ignore
                source, target = edge
                source_id = int(source[1:]) - 1
                target_id = int(target[1:]) - 1
                adj_mtx[source_id][target_id] = 1
                if adj_mtx[target_id][source_id] == 1:
                    adj_mtx[source_id][target_id] = 0
                    adj_mtx[target_id][source_id] = 0
        
        
        elif options['baseline'] == "Notears":
            if options['dataname'] == "notears":
                W_est = notears_linear(data.to_numpy(), lambda1=0.1, loss_type='l2')
            else:
                W_est = notears_linear(data.to_numpy(), lambda1=0.1, loss_type='logistic')
            adj_mtx = (W_est > 0) * 1.
        
        
        elif options['baseline'] == "MLPNotears":
            model = NotearsMLP(dims=[len(all_vars), 10, 1], bias=True)
            W_est = notears_nonlinear(model, data.to_numpy(dtype=np.float32), lambda1=0.01, lambda2=0.01)
            adj_mtx = (W_est > 0) * 1.
        
        
        elif options['baseline'] == "DAS":
            algorithm = DAS(data.sample(500).to_numpy(), kwargs={'d': len(all_vars), 'eta_G': 0.001, 'eta_H': 0.001, 
                                                    'cam_cutoff': 0.001, 'pruning': 'DAS', 'threshold': 0.05, 'K': 10})
            adj_mtx = algorithm.inference()
            adj_mtx = (adj_mtx > 0) * 1.
            
        
        elif options['baseline'] == "SCORE":
            algorithm = SCORE(data.sample(500).to_numpy(), kwargs={'d': len(all_vars), 'eta_G': 0.001, 'eta_H': 0.001, 
                                                    'cam_cutoff': 0.001, 'pruning': 'CAM', 'threshold': 0.05, 'pns': 10})
            adj_mtx = algorithm.inference()
            adj_mtx = (adj_mtx > 0) * 1.
        
        
        finish = time.time()
        
        etrue = len(true_edge(groundtruth, adj_mtx))
        espur = len(spur_edge(groundtruth, adj_mtx))
        efals = len(fals_edge(groundtruth, adj_mtx))
        emiss = len(miss_edge(groundtruth, adj_mtx))

        f = open(options["output"], "a")
        if options['dataname'] == "notears":
            f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                options['dataname'], options['d'], options['s'] if options['s'] else options['d'], options['baseline'],
                etrue, espur, emiss, efals, espur+emiss+efals, round(etrue/(etrue + espur + efals), 2) if etrue + espur + efals > 0 else 0, finish - start
            ))
            
        else:
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
                options['dataname'], options['folder'], options['baseline'],
                etrue, espur, emiss, efals, espur+emiss+efals, round(etrue/(etrue + espur + efals), 2) if etrue + espur + efals > 0 else 0, finish - start
            ))
        f.close()
        print("Writting results done!")