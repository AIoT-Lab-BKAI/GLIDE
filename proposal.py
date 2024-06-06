import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
from itertools import combinations
from plot_utils import true_edge, spur_edge, fals_edge, miss_edge
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
    parser.add_argument("--confidence", type=float, default='0.01')
    parser.add_argument("--TMB", type=int, default=0)
    parser.add_argument("--hardcap", type=float, default='0.001')
    parser.add_argument("--gamma2", type=float, default='0.5')
    parser.add_argument("--num_env", type=int, default=10)
    parser.add_argument("--capsize", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=['aS', 'aL', 'aL-Re', 'n'], default='aS')
    parser.add_argument("--exp_repeat", type=int, default=1)
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
    else:
        for file in sorted(os.listdir(folderpath)):
            filename = os.path.join(folderpath, file)
            silo_data = pd.read_csv(filename)
            silos.append(silo_data)
            # print("Loaded file:", filename)

    merged_df = pd.concat(silos[:-1], axis=0)
    merged_df = merged_df.reindex(sorted(merged_df.columns, key=lambda item: int(item[1:])), axis=1)
    all_vars = list(merged_df.columns)
    return merged_df, all_vars, groundtruth


def find_basis(df: pd.DataFrame, all_vars: list, confidence=0.01):
    """
    Find the maximum set of inter-independent variables
    given the data and the list of variables
    Args:
        df: data
        all_vars: set of variables of interest
    """
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
    
    basis = []
    random_vars = deepcopy(all_vars)
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


def true_markov_blanket(adj_matrix, var_idx):
    parents = np.where(adj_matrix[:, var_idx])[0].tolist()
    children = np.where(adj_matrix[var_idx])[0].tolist()
    
    spouses = set()
    for c in children:
        for sp in np.where(adj_matrix[:, c])[0]:
            spouses.add(sp)
    
    pa_sp = list(set(parents)&spouses)
    ch_sp = list(set(children)&spouses)
    spouses = list(spouses - set(pa_sp) - set(ch_sp))
    
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


def recursive_conn(markov_blankets, neighbors):
    output = []
    if len(neighbors) <= 1:
      output = [neighbors]
    else:
      for i in neighbors:
        res_i = [i] + recursive_conn(markov_blankets, list(set(neighbors)&set(markov_blankets[i])))
        output.append(res_i)
    return output


def unfold(input):
    """
    Arguments:
      input: [var, var, ..., [var, ...], [var, ...]]

    that has a number of non-list element and a number of list element
    """
    cut_index = 0
    while cut_index < len(input):
      cut_index += 1
      if isinstance(input[cut_index], list):
        break

    out = []
    for i in range(cut_index, len(input)):
      out.append([*input[:cut_index], *input[i]])
    return out


def individual_causal_search_backward(var, silos_index):
    record = {}
    for mb_var in markov_blankets[var]:
        variance, _ = compute_weighted_variance_viaindexesv2(silos_index, var, [mb_var])
        record[tuple([mb_var])] = variance
    return {var: record}


def individual_causal_search_forward(var, potential_parents, silos_index):
    buffers = {}
    for group in sorted(potential_parents[var], key=lambda item: len(item)):
        for l in range(1, min(len(group)+1, options['capsize'])).__reversed__():
            for comb in combinations(group, l):
                comb = tuple(sorted(comb))
                if comb not in buffers.keys():
                    variance, _ = compute_weighted_variance_viaindexesv2(silos_index, var, list(comb))
                    buffers[comb] = variance
    return {var: buffers}
  
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


def get_potential_parents(markov_blankets):
    recursive_outputs = {}
    for anchor_var in markov_blankets.keys():
        recursive_outputs[anchor_var] = recursive_conn(markov_blankets, deepcopy(markov_blankets[anchor_var]))

    potential_parents = {}
    for anchor_var in markov_blankets.keys():
        recursive_output = recursive_outputs[anchor_var]
        final_output = set()
        for i in range(len(recursive_output)):
            test_case = deepcopy(recursive_output[i])
            unique_elements = set()
            if len(test_case) <= 1:
                unique_elements.add(tuple(test_case))
            else:
                first_element = test_case.pop(0)
                while len(test_case):
                    examine_group = test_case.pop(0)
                    if len(examine_group) and not isinstance(examine_group[0], list) and isinstance(examine_group[-1], list):
                        test_case += [*unfold(examine_group)]
                    else:
                        unique_elements.add(tuple(sorted(examine_group + [first_element])))
                    
            final_output = final_output|unique_elements
        potential_parents[anchor_var] = [j for j in final_output]
    return potential_parents


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
    if not Path(options['output']).exists():
        f = open(options["output"], "w")
        f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            'dataname', 'folder', 'num_env','gamma2', 'TMB', 'mode', 'capsize',
            'etrue', 'espur', 'emiss', 'efals', 'time'))
        f.close()
    
    for _ in range(options['exp_repeat']):
        start = time.time()
        markov_blankets = {var: [] for var in all_vars}
        if options['TMB']:
            for var in markov_blankets.keys():
                pa, pa_sp, sp, ch_sp, ch = true_markov_blanket(groundtruth, int(var[1:]) - 1)
                markov_blankets[var] = list(set(to_list(all_vars, pa + pa_sp + sp + ch_sp + ch)) - set([var]))
        else:
            markov_blankets = GSMB(df, [i for i in range(len(df))])
        

        def compute_variance_viaindexesv2(indexes: list, variable: str, parents: list):
            conditional_probs_record = df[parents + [variable]].groupby(parents + [variable]).count().reset_index()
            mll_list = []
            env = 0
            for index in indexes:
                vertical_sampled_data = df.iloc[index].reset_index()
                vertical_sampled_data = vertical_sampled_data.drop(columns=['index'])
                vertical_sampled_data.insert(0, 'count', [1] * len(vertical_sampled_data))
                
                summary_with_ch = vertical_sampled_data.groupby(parents + [variable])['count'].sum().reset_index()
                mll, output = compute_mll(summary_with_ch, parents, env)
                conditional_probs_record = conditional_probs_record.merge(output, on=parents + [variable], how='left')
                mll_list.append(mll)
                env += 1
            
            mean_mll = np.mean(mll_list)
            var_avg = conditional_probs_record.iloc[:, len(parents) + 1:].var(axis=1, skipna=True).mean()
            return var_avg, mean_mll, conditional_probs_record

        def compute_weighted_variance_viaindexesv2(indexes: list, variable: str, parents: list):
            variance, _, df = compute_variance_viaindexesv2(indexes, variable, parents)
            if len(parents):
                joint_mat = np.array([df[f'joint_{i}'] for i in range(len(indexes))]).T
                probs_mat = np.array([df[f'probs_{i}'] for i in range(len(indexes))]).T
                probs_mean = []
                for i in range(probs_mat.shape[0]):
                    if len(probs_mat[i][~np.isnan(probs_mat[i])]):
                        probs_mean.append(np.mean(probs_mat[i][~np.isnan(probs_mat[i])]).item())
                    else:
                        probs_mean.append(0)
                        
                probs_mean = np.expand_dims(np.array(probs_mean), 1)
                # joint_mat = joint_mat.shape[1] * joint_mat/joint_mat.sum(axis=1, keepdims=True)
                prod = joint_mat * (probs_mat - probs_mean)**2
                return np.power(np.mean(prod[~np.isnan(prod)]), 0.5), parents
            else:
                return variance, parents

        def procedure_for_sources(sources):
            potential_parents = get_potential_parents(markov_blankets)
            sample_dis = {x: generate_uniform_distributions(P0=marginal_prob(df, [x]),
                                                            num_gen=options['num_env'], 
                                                            gamma2=np.power(options['gamma2'], 1./len(sources))) for x in sources}
            silos_index = [multivariate_sampling(df, sources, sample_dis, i) for i in range(options['num_env'])]
            inputs = [(var, potential_parents, silos_index) for var in markov_blankets.keys()]
            outputs = execute_in_parallel(individual_causal_search_forward, inputs)

            results = tuple()
            for out_dict in outputs:
                results += tuple(out_dict.items())
            results = dict(results)
            
            weighted_mtx = res2mtx(results, all_vars)
            hardcap_invariance = options['hardcap']
            weighted_mtx[weighted_mtx > hardcap_invariance] = 0
            adj_mtx = (weighted_mtx > 0) * 1
            return adj_mtx

        def procedure_for_leaves(leaves):
            sample_dis = {x: generate_uniform_distributions(P0=marginal_prob(df, [x]),
                                                    num_gen=options['num_env'], 
                                                    gamma2=np.power(options['gamma2'], 1./len(leaves))) for x in leaves}
            silos_index = [multivariate_sampling(df, leaves, sample_dis, i) for i in range(options['num_env'])]
            inputs = [(var, silos_index) for var in markov_blankets.keys()]
            outputs = execute_in_parallel(individual_causal_search_backward, inputs)

            results = tuple()
            for out_dict in outputs:
                results += tuple(out_dict.items())
            results = dict(results)
            
            weighted_mtx = res2mtx(results, all_vars)
            hardcap_invariance = options['hardcap']
            weighted_mtx[weighted_mtx > hardcap_invariance] = 0
            adj_mtx = (weighted_mtx > 0) * 1
            return adj_mtx.T
        
        
        mode = options['mode']
        if (mode.upper() == "AS") or (mode.upper() == "MS"):
            basis_index = [i for i in range(groundtruth.shape[0]) if np.sum(groundtruth[:,i]) == 0]
            sources = np.array(all_vars)[np.array(basis_index)].tolist()
            adj_mtx = procedure_for_sources(sources)
            
            
        elif (mode.upper() == "AL") or (mode.upper() == "AL-RE"):
            basis_index = [i for i in range(groundtruth.shape[0]) if np.sum(groundtruth[i]) == 0]
            leaves = np.array(all_vars)[np.array(basis_index)].tolist()
            leaves = find_basis(df, leaves)
            adj_mtx = procedure_for_leaves(leaves)
            
            if mode.upper() == "AL-RE":
                sources_idx = np.array([i for i in range(len(all_vars)) if np.sum(adj_mtx[:, i]) == 0])
                sources = np.array(all_vars)[sources_idx].tolist()
                sources = list(set(sources) - set(leaves))
                adj_mtx = procedure_for_sources(sources)
        
        
        else: # mode = 'n'
            basis = find_basis(df, all_vars)
            adj_mtx = procedure_for_sources(basis)



        finish = time.time()
        etrue, espur, emiss, efals = evaluate(groundtruth, adj_mtx)
        f = open(options["output"], "a")
        f.write("{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            options['dataname'], options['folder'], options['num_env'], options['gamma2'], options['TMB'], options['mode'], options['capsize'],
            etrue, espur, emiss, efals, finish - start
        ))
        f.close()
        print("Writting results done!")



