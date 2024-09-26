import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
from utils.plot_utils import true_edge, spur_edge, fals_edge, miss_edge
import time
from causallearn.utils.cit import CIT
from copy import deepcopy
from sklearn.cluster import KMeans
from multiprocessing import Pool
from typing import List, Tuple
from random import shuffle


def read_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="asia")
    parser.add_argument("--folder", type=str, default="m3_d1_n10")
    parser.add_argument("--output", type=str, default="res.csv")
    parser.add_argument("--confidence", type=float, default='0.05')
    parser.add_argument("--TMB", type=int, default=1)
    parser.add_argument("--hardcap", type=float, default='0.001') # do not change
    parser.add_argument("--gamma2", type=float, default='0.5')
    parser.add_argument("--num_env", type=int, default=10)
    parser.add_argument("--mode", type=str, choices=['aS', 'aL', 'aL-Re', 'n'], default='n') # do not change
    parser.add_argument("--exp_repeat", type=int, default=1)
    
    parser.add_argument("--d", type=int, default=20, help="Only used for notears dataset, the number of nodes")
    parser.add_argument("--s", type=int, default=None, help="Only used for notears dataset, the number of edges")
    parser.add_argument("--b", type=int, default=4, help="Only used for notears dataset, the number of discretization bins") # do not change
    parser.add_argument("--ntype", type=str, default="linear", choices=["linear", "nonlinear", 
                                                                        "sf_linear", "sf_nonlinear",
                                                                        "bp_linear", "bp_nonlinear"])
    options = vars(parser.parse_args())
    return options


def load_data(options):
    dataname = options["dataname"]
    
    if dataname == 'notears':
        d, b, ntype = options['d'], options['b'], options['ntype']
        s = options['s'] if options['s'] is not None else d
        folderpath = f"./data/{dataname}/{ntype}Gaussian/processed/X_{d}_{s}_{b}.csv"
        merged_df = pd.read_csv(folderpath, index_col=0)
        groundtruth = np.loadtxt(f"./data/{dataname}/{ntype}Gaussian/W_true_{d}_{s}.csv", delimiter=',')
        all_vars = list(merged_df.columns)
        
        if not Path(options['output']).exists():
            f = open(options["output"], "w")
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                'dataname', 'd', 's', 'b', 'num_env','gamma2', 'TMB', 'mode',
                'etrue', 'espur', 'emiss', 'efals', 'shd', 'tpr','time'))
            f.close()
            
        return merged_df, all_vars, groundtruth
    
    else:
        folder = options["folder"]
        folderpath = f"./data/categorical/{dataname}/{folder}"
        groundtruth = np.loadtxt(f"./data/categorical/{dataname}/adj.txt")

        silos = []
        if not Path(folderpath).exists():
            print("Folder", folderpath, "not exist!")
        else:
            for file in sorted(os.listdir(folderpath)):
                filename = os.path.join(folderpath, file)
                silo_data = pd.read_csv(filename)
                silos.append(silo_data)
                # print("Loaded file:", filename)

        merged_df = pd.concat(silos[:-1], axis=0)
        merged_df = merged_df.reindex(sorted(merged_df.columns, key=lambda item: int(item[1:])), axis=1)
        all_vars = list(merged_df.columns)
        
        if not Path(options['output']).exists():
            f = open(options["output"], "w")
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                'dataname', 'folder', 'num_env','gamma2', 'TMB', 'mode',
                'etrue', 'espur', 'emiss', 'efals', 'shd', 'tpr','time'))
            f.close()
        
        return merged_df, all_vars, groundtruth


def find_connectivity(df: pd.DataFrame, all_vars: list, confidence=0.05):
    data = df[all_vars]
    connectivity = {var: [] for var in all_vars}
    chisq_obj = CIT(data, "chisq")

    for X in connectivity.keys():
        other_vars = list(set(all_vars) - set(connectivity[X]) - set([X]))
        for Y in other_vars:
            pval = chisq_obj(all_vars.index(X), all_vars.index(Y), []) # type: ignore
            if pval <= confidence: # type: ignore
                connectivity[X] = list(set(connectivity[X]) | set([Y]))
                connectivity[Y] = list(set(connectivity[Y]) | set([X]))
    return connectivity


def find_basis(connectivity: dict, bounded_set = None):
    """
    Find the maximum set of inter-independent variables
    given the connectivity dictionary
    {
        var: [list of all variables that are dependent on var]
    }
    """
    basis = []
    if bounded_set:
        ordering = sorted(bounded_set, key=lambda item: len(connectivity[item]), reverse=False)
    else:
        random_vars = deepcopy(list(connectivity.keys()))
        shuffle(random_vars)
        ordering = sorted(random_vars, key=lambda item: len(connectivity[item]), reverse=False)
        
    while len(ordering):
        x = ordering.pop(0)
        discard_vars = connectivity[x]
        ordering = sorted(list(set(ordering) - set(discard_vars)), 
                        key=lambda item: len(list(set(connectivity[item]) - set(discard_vars))), reverse=False)
        basis.append(x)
    return basis


def GSMB(df: pd.DataFrame, indexes, confidence=0.01):
    data = df.iloc[indexes].reset_index().drop(columns=['index'])
    chisq_obj = CIT(data, "chisq") # construct a CIT instance with data and method name
    all_var_idx = [i for i in range(len(data.columns))]
    markov_blankets_idx = {i: [] for i in range(len(data.columns))}

    for X in all_var_idx:
        S = []
        prev_length = 0
        count = 0
        while True:
            count += 1
            # print("==============New cycle==================")
            for Y in list(set(all_var_idx) - set(S) - set([X])):
                if Y != X:
                    pval = chisq_obj(X, Y, S) # type:ignore
                    if pval <= confidence: # type:ignore
                        S.append(Y)
            
            for Y in deepcopy(S):
                pval = chisq_obj(X, Y, list(set(S) - set([Y]))) # type:ignore
                if pval > confidence: # type:ignore
                    S.remove(Y)
            
            if (len(S) - prev_length == 0) or (count > 2):
                break
            else:
                prev_length = len(S)
        markov_blankets_idx[X] = list(set(markov_blankets_idx[X])|set(S))
    
    all_vars = df.columns.to_list()
    markov_blankets = {var: [] for var in all_vars}
    for idx, mb_idxes in markov_blankets_idx.items():
        var = all_vars[idx]
        markov_blankets[var] = [all_vars[i] for i in mb_idxes]
    
    return markov_blankets


def removes_irrelevant(df, var, plausible_set, confidence=0.01):
    subdata = df[[var, *plausible_set]]
    all_var = list(subdata.columns)
    all_var_idx = [i for i in range(len(all_var))]
    chisq_obj = CIT(subdata, 'chisq')
    
    X = all_var.index(var)
    S = []
    prev_length = 0
    count = 0
    while True:
        count += 1
        for Y in deepcopy(S):
            pval = chisq_obj(X, Y, list(set(S) - set([Y]))) # type:ignore
            if pval > confidence: # type:ignore
                S.remove(Y)
                
        for Y in list(set(all_var_idx) - set(S) - set([X])):
            if Y != X:
                pval = chisq_obj(X, Y, S) # type:ignore
                if pval <= confidence: # type:ignore
                    S.append(Y)
                    
        if (len(S) - prev_length == 0) or (count > 10):
            break
        else:
            prev_length = len(S)
        
    return [all_var[i] for i in S]


def true_markov_blanket(adj_matrix, var_idx):
    parents = np.where(adj_matrix[:, var_idx])[0].tolist()
    children = np.where(adj_matrix[var_idx])[0].tolist()
    
    spouses = set()
    for c in children:
        for sp in np.where(adj_matrix[:, c])[0]:
            spouses.add(sp)
    
    pa_sp = list(set(parents)&spouses - set(parents) - set([var_idx]))
    ch_sp = list(set(children)&spouses - set(children) - set([var_idx]))
    spouses = list(spouses - set(pa_sp) - set(ch_sp) - set([var_idx]))
    
    return parents, pa_sp, spouses, ch_sp, children


def to_list(all_vars, mb_idx_list):
        return [all_vars[i] for i in mb_idx_list]


def generate_uniform_distributions(P0: np.ndarray, num_gen=100, gamma2=0.8):
    Ulist = list(np.eye(P0.shape[0]))
    # Compute the boundary points
    boundaries = []
    for i in range(len(Ulist)):
        if P0[i]/gamma2 < 1:
            alpha_i = 1/(1 - P0[i]) * (1 - P0[i]/(gamma2 + 0.001))
            boundary_i = alpha_i * P0 + (1 - alpha_i) * Ulist[i]
        else:
            boundary_i = Ulist[i]
        boundaries.append(boundary_i)
    
    boundaries = np.stack(boundaries)
    w = np.concatenate([np.random.dirichlet([alpha/2] * len(Ulist), size=num_gen) for alpha in range(1, 10)])
    
    kmeans = KMeans(n_clusters=num_gen, n_init="auto")
    kmeans.fit(w @ boundaries)
    res = kmeans.cluster_centers_
    
    return res


def univariate_sampling(data: pd.DataFrame, variable: str, sample_dis: dict):
    """
    This function create a new data frame from the input data frame
    By sampling single variable following the input sample distribution
    
    Arguments:
        variable:   str
        sample_dis: dict {'value': prob}
    
    Return:
        new_data: pd.DataFrame
    """
    coc = data[variable].to_numpy().flatten()             # Column of Concern (variable column)
    vals = [val for val in sample_dis.keys()]
    counts = np.array([np.sum(coc == val) for val in sample_dis.keys()])
    probs = np.array([p for p in sample_dis.values()])
    num_selects = np.floor(min(counts/probs) * probs).flatten()
        
    all_index = []
    for val, num_select in zip(vals, num_selects):
        all_index += list(np.random.choice(list(np.where(coc==val)[0]), size=int(num_select), replace=False))
    
    res = data.iloc[all_index].reset_index()
    return res.drop(columns=['index']), all_index


def multivariate_sampling(data: pd.DataFrame, variables: list, sample_dis: dict, instance_index):
    remains = deepcopy(variables)
    while len(remains):
        sampling_var = remains.pop(0)
        distribution = sample_dis[sampling_var][instance_index]
        _, all_index = univariate_sampling(data, sampling_var, {i: distribution[i] for i in range(distribution.shape[0])})
    return all_index

### Build the tree
class node:
    def __init__(self, name, bound_set):
        self.name = name
        if name == 'X0':
            self.search_space = bound_set
        else:
            self.search_space = set(markov_blankets[name])&bound_set
        self.path = []

from copy import deepcopy

leaves = []
def build_tree(root: node):
    for leaf in leaves:
        if len((set(root.path) | root.search_space) - set(leaf.path)) == 0:
            return
        
    search_space = sorted((deepcopy(root.search_space)), key=lambda i: -len(root.search_space&set(markov_blankets[i])))
    nest_visited = []
    if len(search_space):
        while len(search_space):
            child_name = search_space.pop()
            child_node = node(child_name, set(root.search_space) - set(nest_visited))
            child_node.path = root.path + [child_name]
            build_tree(child_node)
            nest_visited.append(child_name)
    else:
        leaves.append(root)

# Function to execute F in parallel
def execute_in_parallel(func, args_list: List[Tuple]):
    with Pool() as pool:
        # Map the function F to the arguments in parallel
        results = pool.starmap(func, args_list)
    return results


def evaluate(groundtruth, adj_mtx):
    etrue = true_edge(groundtruth, adj_mtx)
    espur = spur_edge(groundtruth, adj_mtx)
    efals = fals_edge(groundtruth, adj_mtx)
    emiss = miss_edge(groundtruth, adj_mtx)

    return len(etrue), len(espur), len(emiss), len(efals)


def compute_mll(summary_with_ch: pd.DataFrame, potential_parent: list, num_env):
    if len(potential_parent):
        marginalized_ch = summary_with_ch.groupby(potential_parent)['count'].sum().reset_index()
        output = summary_with_ch.merge(marginalized_ch, on=potential_parent, how='left')            
        output.insert(0, f'probs_{num_env}', output['count_x']/output['count_y'])
        output.insert(0, f'joint_{num_env}', output['count_x']/output['count_x'].sum())
        mll = np.array(output['count_x']).dot(np.log(output[f'probs_{num_env}'])).item()
        output = output.drop(['count_x', 'count_y'], axis=1)
        return mll, output
    else:
        output = summary_with_ch.copy()
        output.insert(0, f'probs_{num_env}', output['count']/output['count'].sum())
        mll = np.array(output['count']).dot(np.log(output[f'probs_{num_env}'])).item()
        output = output.drop(['count'], axis=1)
        return mll, output


def get_potential_parents(all_vars, markov_blankets):
    # recursive_outputs = {}
    # for anchor_var in all_vars:
    #     visited.clear()
    #     recursive_outputs[anchor_var] = recursive_conn(deepcopy(markov_blankets[anchor_var]), [])
    leaves.clear()
    root = node('X0', set(all_vars))
    build_tree(root)
    
    recursive_outputs = {var: [] for var in all_vars}
    for leaf in leaves:
        for var in leaf.path:
            recursive_outputs[var].append(list(set(leaf.path) - set([var])))
            
    return recursive_outputs


def marginal_prob(df: pd.DataFrame, variables: list):
    """
    This function compute the marginal distribution of variables
    in a dataset
        
    Arguments:
        df: pd.DataFrame - The input data, whose columns are variables
    
    Return:
        marginal distribution (np.ndarray)
    """
    vars = df.columns
    subdata = df.copy()
    subdata['count'] = [1] * len(subdata)
    res_var = subdata[[*variables, 'count']].groupby(variables).sum().reset_index()
    output = res_var['count']/res_var['count'].sum()
    return output.to_numpy()


def res2mtx(results: dict, all_vars: list):
    weighted_mtx = np.ones([len(all_vars), len(all_vars)])
    for var in all_vars: #type:ignore
        if len(results[var].items()):
            var_id = all_vars.index(var)
            best_comb, best_variance = min(results[var].items(), key=lambda item: item[1])        
            for parent in best_comb:
                pa_id = all_vars.index(parent)
                if best_variance < weighted_mtx[var_id][pa_id]:
                    weighted_mtx[pa_id][var_id] = best_variance
                    weighted_mtx[var_id][pa_id] = 1
    return weighted_mtx
    


if __name__ == "__main__":
    options = read_opts()
    df, all_vars, groundtruth = load_data(options)
    
    print("Running settings:", options)
    for r in range(options['exp_repeat']):
        print(f"Run {r+1}/{options['exp_repeat']}... ", end="")
        start = time.time()
        connectivity = find_connectivity(df, all_vars, 0.05)
        markov_blankets = {var: [] for var in all_vars}
        if options['TMB']:
            for var in markov_blankets.keys():
                pa, pa_sp, sp, ch_sp, ch = true_markov_blanket(groundtruth, int(var[1:]) - 1)
                markov_blankets[var] = list(set(to_list(all_vars, pa + pa_sp + sp + ch_sp + ch)) - set([var]))
        else:
            markov_blankets = GSMB(df, [i for i in range(len(df))])
            
        def compute_variance_via_index(indexes: list, variable: str, parents: list):
            conditional_probs_record = df[parents + [variable]].groupby(parents + [variable]).count().reset_index()
            mll_list = []
            env = 0
            # sample_volumes = []
            for index in indexes:
                vertical_sampled_data = df.iloc[index].reset_index()
                vertical_sampled_data = vertical_sampled_data.drop(columns=['index'])
                vertical_sampled_data.insert(0, 'count', [1] * len(vertical_sampled_data))
                
                summary_with_ch = vertical_sampled_data.groupby(parents + [variable])['count'].sum().reset_index()
                # sample_volumes.append(np.mean(summary_with_ch['count']))
                mll, output = compute_mll(summary_with_ch, parents, env)
                conditional_probs_record = conditional_probs_record.merge(output, on=parents + [variable], how='left')
                mll_list.append(mll)
                env += 1
            
            mean_mll = np.mean(mll_list)
            var_avg = conditional_probs_record.iloc[:, len(parents) + 1:].var(axis=1, skipna=True).mean()
            # mean_sample_volumes = np.mean(sample_volumes)
            return var_avg, mean_mll, conditional_probs_record, True

        def compute_weighted_variance_via_index(indexes: list, variable: str, parents: list):
            _, _, df, sufficient = compute_variance_via_index(indexes, variable, parents)
            if len(parents) and sufficient:
                joint_mat = np.array([df[f'joint_{i}'] for i in range(len(indexes))]).T
                probs_mat = np.array([df[f'probs_{i}'] for i in range(len(indexes))]).T
                probs_mean = []
                for i in range(probs_mat.shape[0]):
                    if len(probs_mat[i][~np.isnan(probs_mat[i])]):
                        probs_mean.append(np.mean(probs_mat[i][~np.isnan(probs_mat[i])]).item())
                    else:
                        probs_mean.append(0)
                        
                probs_mean = np.expand_dims(np.array(probs_mean), 1)
                prod = joint_mat * (probs_mat - probs_mean)**2
                return np.power(np.mean(prod[~np.isnan(prod)]), 0.5), parents
            else:
                return 1, parents

        def individual_causal_search_forward(var, potential_parents_for_var, silos_index):
            buffers = {}
            for group in potential_parents_for_var:
                conn_group = list(set(connectivity[var])&set(group))
                cleaned_group = removes_irrelevant(df, var, conn_group)
                if len(cleaned_group) and (tuple(sorted(cleaned_group)) not in buffers.keys()):
                    variance, _ = compute_weighted_variance_via_index(silos_index, var, cleaned_group)
                    buffers[tuple(sorted(cleaned_group))] = variance
            return {var: buffers}

        def procedure_for_sources(sources):
            potential_parents = get_potential_parents(all_vars, markov_blankets)
            sample_dis = {x: generate_uniform_distributions(P0=marginal_prob(df, [x]),
                                                            num_gen=options['num_env'], 
                                                            gamma2=np.power(options['gamma2'], 1./len(sources))) for x in sources}
            silos_index = [multivariate_sampling(df, sources, sample_dis, i) for i in range(options['num_env'])]

            inputs = [(var, potential_parents[var], silos_index) for var in list(set(all_vars) - set(sources))]
            
            outputs = execute_in_parallel(individual_causal_search_forward, inputs)
            results = tuple()
            for out_dict in outputs:
                results += tuple(out_dict.items())
            results = dict(results)
            for s in sources:
                results[s] = {}
            
            weighted_mtx = res2mtx(results, all_vars)
            hardcap_invariance = options['hardcap']
            weighted_mtx[weighted_mtx > hardcap_invariance] = 0
            adj_mtx = (weighted_mtx > 0) * 1
            return adj_mtx

        basis = find_basis(connectivity)
        adj_mtx = procedure_for_sources(basis)

        finish = time.time()
        print("Done!", end=" ")
        etrue, espur, emiss, efals = evaluate(groundtruth, adj_mtx)
        
        f = open(options["output"], "a")
        if options['dataname'] == "notears":
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                options['dataname'], options['d'],  options['s'] if options['s'] else options['d'], 
                options['b'], options['num_env'], options['gamma2'], options['TMB'], options['mode'],
                etrue, espur, emiss, efals, espur+emiss+efals, round(etrue/(etrue + espur + efals), 2), finish - start
            ))
            
        else:
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                options['dataname'], options['folder'], options['num_env'], 
                options['gamma2'], options['TMB'], options['mode'],
                etrue, espur, emiss, efals, espur+emiss+efals, round(etrue/(etrue + espur + efals), 2), finish - start
            ))
            
        f.close()
        print("Writing results done!")



