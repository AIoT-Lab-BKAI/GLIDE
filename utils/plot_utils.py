def swap_pos(pos, key1, key2):
    tmp = pos[key1]
    pos[key1] = pos[key2]
    pos[key2] = tmp
    return pos


def true_edge(true_adj, pred_adj):
    d, _ = pred_adj.shape
    res = []
    for i in range(d):
        for j in range(d):
            if pred_adj[i][j] != 0 and true_adj[i][j] != 0:
                if ([f'X{i+1}', f'X{j+1}'] not in res) and ([f'X{j+1}', f'X{i+1}'] not in res):
                    res.append([f'X{i+1}', f'X{j+1}'])
    return res


def spur_edge(true_adj, pred_adj):
    d, _ = pred_adj.shape
    res = []
    for i in range(d):
        for j in range(d):
            if pred_adj[i][j] != 0 and true_adj[i][j] == 0 and true_adj[j][i] == 0:
                if ([f'X{i+1}', f'X{j+1}'] not in res) and ([f'X{j+1}', f'X{i+1}'] not in res):
                    res.append([f'X{i+1}', f'X{j+1}'])
    return res


def fals_edge(true_adj, pred_adj):
    d, _ = pred_adj.shape
    res = []
    for i in range(d):
        for j in range(d):
            if pred_adj[i][j] != 0 and true_adj[i][j] == 0 and true_adj[j][i] != 0:
                if ([f'X{i+1}', f'X{j+1}'] not in res) and ([f'X{j+1}', f'X{i+1}'] not in res):
                    res.append([f'X{i+1}', f'X{j+1}'])
    return res


def miss_edge(true_adj, pred_adj):
    d, _ = pred_adj.shape
    res = []
    for i in range(d):
        for j in range(d):
            if true_adj[i][j] != 0 and pred_adj[i][j] == 0 and pred_adj[j][i] == 0:
                if ([f'X{i+1}', f'X{j+1}'] not in res) and ([f'X{j+1}', f'X{i+1}'] not in res):
                    res.append([f'X{i+1}', f'X{j+1}'])
    return res