from upgrade import GSMB, read_opts, load_data, true_markov_blanket, to_list, unfold, \
    generate_uniform_distributions, multivariate_sampling, marginal_prob
    
from causallearn.utils.cit import CIT
import numpy as np
from tqdm import tqdm
from plot_utils import true_edge, spur_edge, fals_edge, miss_edge
import time
from copy import deepcopy
from itertools import combinations
from multiprocessing import Pool
from typing import List, Tuple




options = read_opts()
data, all_vars, groundtruth = load_data(options)

start = time.time()
# ===================================== BASIS ======================================== #
confidence = options['confidence']
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

# ================================= MARKOV BLANKET =================================== #
markov_blankets = {var: [] for var in all_vars}
if options['TMB_activated']:
    for var in markov_blankets.keys():
        pa, ch, sp = true_markov_blanket(groundtruth, int(var[1:]) - 1)
        markov_blankets[var] = list(set(to_list(all_vars, list(set(pa)|set(ch)|set(sp)))) - set([var]))
else:
    markov_blankets = GSMB(data, confidence)


# ================================ PLAUSIBLE CAUSES ================================== #
def recursive_conn(neighbors):
    output = []
    if len(neighbors) <= 1:
        output = [neighbors]
    else:
        for i in neighbors:
            res_i = [i] + recursive_conn(list(set(neighbors)&set(markov_blankets[i])))
        output.append(res_i)
    return output

recursive_outputs = {}
for anchor_var in tqdm(markov_blankets.keys()):
    recursive_outputs[anchor_var] = recursive_conn(deepcopy(markov_blankets[anchor_var]))

potential_parents = {}
for anchor_var in tqdm(markov_blankets.keys(), leave=False):
    recursive_output = recursive_outputs[anchor_var]
    final_output = set()
    for i in range(len(recursive_output)):
        test_case = deepcopy(recursive_output[i])
        unique_elements = set()
        first_element = test_case.pop(0)
        while len(test_case):
            examine_group = test_case.pop(0)
            if len(examine_group) and not isinstance(examine_group[0], list) and isinstance(examine_group[-1], list):
                unfolded = unfold(examine_group)
                test_case += [*unfold(examine_group)]
            else:
                unique_elements.add(tuple(sorted(examine_group + [first_element])))
        final_output = final_output|unique_elements
    potential_parents[anchor_var] = [j for j in final_output]


# ================================ SAMPLING DATASETS ================================== #
num_env = options['num_env']
gamma2 = options['gamma2']

sample_dis = {var: generate_uniform_distributions(P0=marginal_prob(data, [var]),
                                                num_gen=num_env, 
                                                gamma2=np.power(gamma2, 1./len(basis))) for var in basis}

silos_index = [multivariate_sampling(data, basis, sample_dis, i) for i in range(num_env)]


# =============================== CAUSAL IDENTIFICATION =============================== #

def compute_variance_viaindexesv2(indexes: list, variable: str, parents: list):
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
        prod = joint_mat * (probs_mat - probs_mean)**2
        return np.mean(prod[~np.isnan(prod)]), parents
    else:
        return variance, parents
    
def individual_causal_search(var, all_test_cases, silos_index, invariance_hardcap=1e-3):
    print("Here in", var)
    best_variance = invariance_hardcap
    best_comb = ['initiated']
    buffers = {}

    for test_case in all_test_cases:
        for l in range(1, len(test_case) + 1):
            results = []
            for comb in list(combinations(test_case, l)):
                key = tuple(sorted(comb))
                if key in buffers.keys():
                    variance = buffers[key]
                else:
                    variance, _ = compute_weighted_variance_viaindexesv2(silos_index, var, list(key))
                    buffers[key] = variance
                results.append([variance, key])
                
            lowest_variance, cor_comb = sorted(results, key=lambda item: item[0])[0]
            if lowest_variance < best_variance:
                best_variance = lowest_variance
                best_comb = cor_comb

    print("Done with", var)
    return {var: (best_comb, best_variance)}

def execute_in_parallel(args_list: List[Tuple]):
    with Pool() as pool:
        # Map the function F to the arguments in parallel
        results = pool.starmap(individual_causal_search, args_list)
    return results


true_causal_parents = {var: [] for var in all_vars}
invariance_hardcap = options['hardcap']

outputs = []
inputs = [(var, potential_parents[var], silos_index, invariance_hardcap) for var in ['X1', 'X2']] #true_causal_parents
outputs = execute_in_parallel(inputs)

finish = time.time()


### ===================== Evaluate the results =========================== ###
answer_dict = ()
for item in outputs:
    answer_dict += tuple(item.items())
answer_dict = dict(answer_dict)
adj_mtx = np.zeros([len(all_vars), len(all_vars)])
for var in answer_dict.keys():
    if len(answer_dict[var]):
        parents, invariance = answer_dict[var]
        var_id = int(var[1:]) - 1
        for pa in parents:
            pa_id = int(pa[1:]) - 1
            if adj_mtx[var_id][pa_id] == 0:
                adj_mtx[pa_id][var_id] = invariance
            elif adj_mtx[var_id][pa_id] > adj_mtx[pa_id][var_id]:
                adj_mtx[pa_id][var_id] = invariance
                adj_mtx[var_id][pa_id] = 0
                

etrue = len(true_edge(groundtruth, adj_mtx))
espur = len(spur_edge(groundtruth, adj_mtx))
efals = len(fals_edge(groundtruth, adj_mtx))
emiss = len(miss_edge(groundtruth, adj_mtx))
    
f = open(options["output"], "a")
f.write("{},{},{},{},{},{},{},{},{},{}\n".format(
    options['dataname'], options['folder'], options['num_env'], options['gamma2'], etrue, espur, emiss, efals, finish - start
))

f.close()
print("Writting results done!")