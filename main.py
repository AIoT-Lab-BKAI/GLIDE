import argparse
import pandas as pd
from upgrade import *
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from itertools import combinations
from plot_utils import true_edge, spur_edge, fals_edge, miss_edge
import time
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

options = read_opts()
data, all_vars, groundtruth = load_data(options)


def run_in_parallel(func, silos, max_parallel_executions):
    with ProcessPoolExecutor(max_workers=max_parallel_executions) as executor:
        future_to_arg = {executor.submit(func, arg): arg for arg in silos}
        for future in concurrent.futures.as_completed(future_to_arg):
            arg = future_to_arg[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'{arg} generated an exception: {exc}')
    return results


def GSMB(indexes, confidence=0.01):
    subdata = data.iloc[indexes].reset_index().drop(columns=['index'])
    markov_blankets = {}
    chisq_obj = CIT(subdata, "chisq") # construct a CIT instance with subdata and method name
    all_var_idx = [i for i in range(len(subdata.columns))]

    for X in all_var_idx:
        S = []
        # X = 6
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
            
            if (len(S) - prev_length == 0) or (count >= 10):
                break
            else:
                prev_length = len(S)

        markov_blankets[subdata.columns[X]] = [subdata.columns[i] for i in S]
    return markov_blankets


def compute_variance_viaindexesv2(indexes: list, variable: str, parents: list, verbose=False):
    conditional_probs_record = data[parents + [variable]].groupby(parents + [variable]).count().reset_index()
    mll_list = []
    env = 0
    for index in indexes:
        vertical_sampled_data = data.iloc[index].reset_index()
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


def compute_weighted_variance_viaindexesv2(indexes: list, variable: str, parents: list, verbose=False):
    variance, _, df = compute_variance_viaindexesv2(indexes, variable, parents, verbose=verbose)
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
        prod = joint_mat * (probs_mat - probs_mean)**2
        return np.mean(prod[~np.isnan(prod)])
    else:
        return variance


if __name__ == "__main__":
    start = time.time()
    confidence = options["confidence"]
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
    ordering = sorted(list(connectivity.keys()), key=lambda item: len(connectivity[item]), reverse=False)
    while len(ordering):
        var = ordering.pop(0)
        ordering = list(set(ordering) - set(connectivity[var]))
        basis.append(var)

    ### ===================== Learning the Markov Blanket =========================== ###
    markov_blankets = {var: [] for var in all_vars}
    num_env = options["num_env"]
    gamma2 = options["gamma2"]
    sample_dis = {var: generate_uniform_distributions(P0=marginal_prob(data, [var]),
                                                    num_gen=num_env, 
                                                    gamma2=np.power(gamma2, 1./len(basis))) for var in basis}
    # Number of parallel executions
    num_parallel_executions = num_env
    silos_index = [multivariate_sampling(data, basis, sample_dis, i) for i in range(num_parallel_executions)]
    # List to store results
    results = []
    # Running F in parallel and storing results
    results = run_in_parallel(GSMB, silos_index, max_parallel_executions=64)
    for res in results:
        for var, blanket in res.items(): #type:ignore
            markov_blankets[var] += blanket
    
    mk_with_freq = {var: [] for var in all_vars}
    max_size = options["max_markov_size"]
    for var in markov_blankets.keys():
        mk_with_freq[var] = []
        vals, freqs = np.unique(markov_blankets[var], return_counts=True)
        for val, freq in zip(vals, freqs):
            if freq >= int(0.8*num_env):
                if val not in mk_with_freq[var]:
                    mk_with_freq[var].append(val)
        mk_with_freq[var] = sorted(mk_with_freq[var], key=lambda item: item[1], reverse=False)[:max_size]
    
    
    ### ===================== Finding the causal relations =========================== ###
    potential_parents = {var: [] for var in all_vars}
    children = {var: [] for var in all_vars}
    invariance_hardcap = options["hardcap"]
    repeat = options["causal_search_repeat"]
    adj_record = []
    for _ in tqdm(range(repeat)):
        for anchor_var in mk_with_freq.keys():
            markov_variables = list(set(mk_with_freq[anchor_var]) - set(children[anchor_var]))
            if len(markov_variables) < 1:
                continue
            
            lowest_variance = 1e2
            best_comb = []
            
            for l in range(1, len(markov_variables) + 1):
                for comb in list(combinations(markov_variables, l)):
                    comb_variance = compute_weighted_variance_viaindexesv2(silos_index, anchor_var, list(comb))
                    
                    if comb_variance < lowest_variance and comb_variance < invariance_hardcap:
                        lowest_variance = comb_variance
                        best_comb = list(comb)

            potential_parents[anchor_var] = (best_comb, lowest_variance) # type:ignore
            
        adj_mtx = np.zeros([len(all_vars), len(all_vars)])
        for var in potential_parents.keys():
            if len(potential_parents[var]):
                parents, invariance = potential_parents[var]
                var_id = int(var[1:]) - 1
                for pa in parents:
                    pa_id = int(pa[1:]) - 1
                    if adj_mtx[var_id][pa_id] == 0:
                        adj_mtx[pa_id][var_id] = invariance
                    elif adj_mtx[var_id][pa_id] > adj_mtx[pa_id][var_id]:
                        adj_mtx[pa_id][var_id] = invariance
                        adj_mtx[var_id][pa_id] = 0

        for i in range(len(all_vars)):
            children[f'X{i+1}'] = []
            for j in range(len(all_vars)):
                if adj_mtx[i][j] > 0:
                    children[f'X{i+1}'].append(f'X{j+1}')
                    
        adj_record.append(adj_mtx)
        if len(adj_record) >= 2:
            if np.sum(adj_record[-1] - adj_record[-2]) == 0:
                break
    
    finish = time.time()
    print("Causal search done!")
    
    
    ### ===================== Evaluate the results =========================== ###
    res_record = []
    for fin_adjmtx in adj_record:
        etrue = true_edge(groundtruth, fin_adjmtx)
        espur = spur_edge(groundtruth, fin_adjmtx)
        efals = fals_edge(groundtruth, fin_adjmtx)
        emiss = miss_edge(groundtruth, fin_adjmtx)
        res_record.append([len(etrue), len(espur), len(emiss), len(efals)])

    ranks = sorted(res_record, key=lambda item: item[1] + item[2] + item[3])
        
    f = open(options["output"], "a")
    ### Best
    etrue_best, espur_best, emiss_best, efals_best = ranks[0]
    f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
        options['dataname'], options['folder'], options['num_env'], options['gamma2'], "best", etrue_best, espur_best, emiss_best, efals_best, finish - start
    ))
    
    ### Worst
    etrue_worst, espur_worst, emiss_worst, efals_worst = ranks[-1]
    f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
        options['dataname'], options['folder'], options['num_env'], options['gamma2'], "worst", etrue_worst, espur_worst, emiss_worst, efals_worst, finish - start
    ))
    
    ### First
    etrue_firest, espur_firest, emiss_firest, efals_firest = res_record[0]
    f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
        options['dataname'], options['folder'], options['num_env'], options['gamma2'], "first", etrue_firest, espur_firest, emiss_firest, efals_firest, finish - start
    ))
    
    ### Final
    etrue_fin, espur_fin, emiss_fin, efals_fin = res_record[-1]
    f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
        options['dataname'], options['folder'], options['num_env'], options['gamma2'], "final", etrue_fin, espur_fin, emiss_fin, efals_fin, finish - start
    ))
    
    f.close()
    print("Writting results done!")