import numpy as np
import pandas as pd
import torch
from typing import List

from typing import List

def compute_mll(summary_with_ch: pd.DataFrame, potential_parent: list, num_env):
    if len(potential_parent):
        marginalized_ch = summary_with_ch.groupby(potential_parent)['count'].sum().reset_index()
        output = summary_with_ch.merge(marginalized_ch, on=potential_parent, how='left')            
        output.insert(0, f'probs_{num_env}', output['count_x']/output['count_y'])
        output.insert(0, f'joint_{num_env}', output['count_x']/output['count_x'].sum())
        mll = np.array(output['count_x']).dot(np.log(output[f'probs_{num_env}'])).item()
        # output = output.rename(columns={"count_x": f"count_{num_env}"})
        # output = output.drop(['count_y'], axis=1)
        output = output.drop(['count_x', 'count_y'], axis=1)
        return mll, output
    else:
        output = summary_with_ch.copy()
        output.insert(0, f'probs_{num_env}', output['count']/output['count'].sum())
        mll = np.array(output['count']).dot(np.log(output[f'probs_{num_env}'])).item()
        output = output.drop(['count'], axis=1)
        return mll, output


def compute_variance_viasilos(silos: List[pd.DataFrame], variable: str, parents: list, verbose=False):
    conditional_probs_record = silos[0][parents + [variable]].groupby(parents + [variable]).count().reset_index()
    mll_list = []
    env = 0
    for data in silos:
        vertical_sampled_data = data[parents + [variable]]
        vertical_sampled_data.insert(0, 'count', [1] * len(vertical_sampled_data))
        
        summary_with_ch = vertical_sampled_data.groupby(parents + [variable])['count'].sum().reset_index()
        mll, output = compute_mll(summary_with_ch, parents, env)
        conditional_probs_record = conditional_probs_record.merge(output, on=parents + [variable], how='left')
        mll_list.append(mll)
        env += 1
            
    mean_mll = np.mean(mll_list)
    var_avg = conditional_probs_record.iloc[:, len(parents) + 1:].var(axis=1, skipna=True).mean()
    if verbose:
        print(conditional_probs_record)
    return var_avg, mean_mll, conditional_probs_record


def compute_weighted_variance_viasilos(silos: List[pd.DataFrame], variable: str, parents: list, verbose=False):
    variance, _, df = compute_variance_viasilos(silos, variable, parents, verbose=verbose)
    if len(parents):
        joint_mat = np.array([df[f'joint_{i}'] for i in range(len(silos))]).T
        probs_mat = np.array([df[f'probs_{i}'] for i in range(len(silos))]).T
        probs_mean = np.array([np.mean(probs_mat[i][~np.isnan(probs_mat[i])], keepdims=True) for i in range(probs_mat.shape[0])])
        prod = joint_mat * (probs_mat - probs_mean)**2
        return np.mean(prod[~np.isnan(prod)])
    else:
        return variance
    



# Conditional probability variance
def compute_variance(data: pd.DataFrame, variable: list, parents: list, num_envs=100, frac=0.5):
    vertical_sampled_data = data[parents + variable]
    conditional_probs_record = vertical_sampled_data.groupby(parents + variable).count().reset_index()
    vertical_sampled_data.insert(0, 'count', [1] * len(vertical_sampled_data))

    mll_list = []
    env = 0
    while env < num_envs:
        horizontal_sampled_data = vertical_sampled_data.sample(frac=frac)
        summary_with_ch = horizontal_sampled_data.groupby(parents + variable)['count'].sum().reset_index()
        
        if len (summary_with_ch) < len(conditional_probs_record):
            frac = min(frac + 0.02, 1.0)
            continue
        
        if len(parents):
            mll, output = compute_mll(summary_with_ch, parents, env)
            conditional_probs_record = conditional_probs_record.merge(output, on=parents + variable, how='left')
            mll_list.append(mll)
            env += 1
            frac = max(frac - 0.01, 0.1)
    
    mean_mll = np.mean(mll_list)
    var_avg = conditional_probs_record.iloc[:, len(parents) + 1:].var(axis=1).mean()
    return var_avg, mean_mll



def compute_variance_v2(data: pd.DataFrame, variable: list, parents: list, num_test=5):
    vertical_sampled_data = data[parents + variable]
    conditional_probs_record = vertical_sampled_data.groupby(parents + variable).count().reset_index()
    vertical_sampled_data.insert(0, 'count', [1] * len(vertical_sampled_data))

    mll_list = []
    indexing = np.array([i for i in range(len(vertical_sampled_data))])

    env = 0
    for pa in parents:
        pa_vals = np.unique(vertical_sampled_data[pa]).tolist()
        pa_vals_count = np.array([len(indexing[vertical_sampled_data[pa] == val]) for val in pa_vals])
        
        for _ in range(num_test):
            ran_num = [np.random.randint(10, count) for count in pa_vals_count]

            chosen_index = []
            for val, num in zip(pa_vals, ran_num):
                chosen_index += np.random.choice(indexing[vertical_sampled_data[pa] == val], num).tolist()
            
            horizontal_sampled_data = vertical_sampled_data.iloc[chosen_index]
            summary_with_ch = horizontal_sampled_data.groupby(parents + variable)['count'].sum().reset_index()
            mll, output = compute_mll(summary_with_ch, parents, env)
            conditional_probs_record = conditional_probs_record.merge(output, on=parents + variable, how='left')
            mll_list.append(mll)
            env += 1
    
    mean_mll = np.mean(mll_list)
    var_avg = conditional_probs_record.iloc[:, len(parents) + 1:].var(axis=1).mean()
    return var_avg, mean_mll



# Entropy
def entropy(df: pd.DataFrame, covariate: list):
    """ Compute entropy
    Arguments:
        covariate: list of elements - the variable X
    Returns:
        H(X)
    Confirmed correct using scipy.stats.entropy
    """
    sub_df = df[covariate + ['count']].groupby(covariate)['count'].sum().reset_index()
    sub_df['prob'] = sub_df['count'] / sub_df['count'].sum()
    return -sub_df['prob'].dot(np.log(sub_df['prob']))


# Conditional Entropy
def conditional_entropy(df: pd.DataFrame, covariate: list, condition_sets: list):
    """ Compute the conditional entropy
    Arguments:
        covariate: list of ONE element - the variable X
        condition_sets: list of conditional variable  Y
    Returns:
        H(X|Y)
    
    Example:
        covariate       = ['D']
        condition_sets  = ['A', 'B', 'C']
        => Return H(D|A,B,C)
    Confirmed correct using H(X) - H(X|Y) = I(X;Y) and double checked the definition
    """
    if len(condition_sets):
        all_var = covariate + condition_sets

        all_var_df = df[all_var + ['count']].groupby(all_var)['count'].sum().reset_index()
        condition_df = all_var_df.groupby(condition_sets)['count'].sum().reset_index()

        all_var_df['prob'] = all_var_df['count'] / all_var_df['count'].sum()
        condition_df['prob'] = condition_df['count'] / condition_df['count'].sum()

        output = all_var_df.merge(condition_df, on=condition_sets, how='left')
        conditional_enp = - output['prob_x'].dot(np.log(output['prob_x'] / output['prob_y']))

        return conditional_enp
    else:
        return entropy(df, covariate)
    

# Mutual Information
def mutual_information(df: pd.DataFrame, covariate1: list, covariate2: list):
    """
    Compute I(covariate1, covariate2)
    Confirmed correct using sklearn.metrics.mutual_info_score
    """
    joint_df = df[covariate1 + covariate2 + ['count']].groupby(covariate1 + covariate2)['count'].sum().reset_index()
    joint_df['prob_join'] = joint_df['count'] / np.sum(joint_df['count'])

    var1_df = joint_df[covariate1 + ['count']].groupby(covariate1)['count'].sum().reset_index()
    var1_df['var1_prob'] = var1_df['count'] / np.sum(var1_df['count'])

    var2_df = joint_df[covariate2 + ['count']].groupby(covariate2)['count'].sum().reset_index()
    var2_df['var2_prob'] = var2_df['count'] / np.sum(var2_df['count'])

    output = joint_df.merge(var1_df, on=covariate1, how='left').merge(var2_df, on=covariate2, how='left')
    mi = output['prob_join'].dot(np.log(output['prob_join']/(output['var1_prob'] * output['var2_prob'])))
    return mi


# Conditional Mutual Information
def conditional_mutual_information(df: pd.DataFrame, covariate1: list, covariate2: list, condition_sets: list):
    """ Compute the conditional mutual information 
        of covariate1 and covariate2, given the condition_sets
    
    Arguments:
        condition_sets: list of conditional variable   Z
        covariate1: list of ONE element - the variable X
        covariate1: list of ONE element - the variable Y
    Returns:
        I(X;Y|Z)
    
    Notice that:
        I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
    therefore, generally:
        I(X;Y|Z) != I(Y;X|Z)
    
    Since Conditional entropy confirmed correct
    and Mutual information confirmed correct
    This function is, by nature, correct without checking
    """
    if len(condition_sets):
        interest_df = df[condition_sets + covariate1 + covariate2 + ['count']]
        return conditional_entropy(interest_df, covariate1, condition_sets) - conditional_entropy(interest_df, covariate1, condition_sets + covariate2)
    else:
        return mutual_information(df, covariate1, covariate2)


def is_acyclic(adj):
    return torch.trace(torch.linalg.matrix_exp(torch.from_numpy(adj))).item() - adj.shape[0] == 0


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            B_est = (B_est > 0) * 1.0
        if not is_acyclic(B_est):
            raise ValueError('B_est should be a DAG')
        
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}