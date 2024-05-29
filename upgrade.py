import numpy as np
import pandas as pd
from copy import deepcopy
from causallearn.utils.cit import CIT
from sklearn.cluster import KMeans
import argparse
from pathlib import Path
from tqdm import tqdm
import os

def inP(p0: np.ndarray, px: np.ndarray, gamma):
    return min(p0/px) >= gamma

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
    w = np.random.uniform(0, 1, (num_gen * 30, len(Ulist)))
    w = w/w.sum(axis=1, keepdims=True)
    
    kmeans = KMeans(n_clusters=num_gen, n_init="auto")
    kmeans.fit(w @ boundaries)
    res = kmeans.cluster_centers_
    return res


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
    # res[var] = res_var['prob'].to_numpy()
    return output.to_numpy()

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
        data, all_index = univariate_sampling(data, sampling_var, {i: distribution[i] for i in range(distribution.shape[0])})
    return data, all_index

# def GSMB(data: pd.DataFrame, confidence=0.01):
#     markov_blankets = {}
#     chisq_obj = CIT(data, "chisq") # construct a CIT instance with data and method name
#     all_var_idx = [i for i in range(len(data.columns))]

#     for X in all_var_idx:
#         S = []
#         # X = 6
#         prev_length = 0
#         count = 0
#         while True:
#             count += 1
#             # print("==============New cycle==================")
#             for Y in list(set(all_var_idx) - set(S) - set([X])):
#                 if Y != X:
#                     pval = chisq_obj(X, Y, S) # type:ignore
#                     if pval <= confidence: # type:ignore
#                         S.append(Y)
            
#             for Y in deepcopy(S):
#                 pval = chisq_obj(X, Y, list(set(S) - set([Y]))) # type:ignore
#                 if pval > confidence: # type:ignore
#                     S.remove(Y)
            
#             if (len(S) - prev_length == 0) or (count >= 10):
#                 break
#             else:
#                 prev_length = len(S)

#         markov_blankets[data.columns[X]] = [data.columns[i] for i in S]
#     return markov_blankets


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
    

def read_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="asia")
    parser.add_argument("--folder", type=str, default="m3_d1_n10")
    parser.add_argument("--output", type=str, default="res.csv")
    parser.add_argument("--confidence", type=float, default='0.01')
    parser.add_argument("--hardcap", type=float, default='0.001')
    parser.add_argument("--gamma2", type=float, default='0.8')
    parser.add_argument("--num_env", type=int, default=100)
    parser.add_argument("--max_markov_size", type=int, default=8)
    parser.add_argument("--causal_search_repeat", type=int, default=5)
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
    merged_df = pd.concat(silos, axis=0)
    print("Loading data done! -- Full data:", len(merged_df))
    return merged_df, all_vars, groundtruth