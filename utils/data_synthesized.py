from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import random

class Node:
    def __init__(self, num_vals, id, alpha=1.) -> None:
        """
            num_vals: Int
            id: Int
            alpha: Float (dirichlet's alpha)
        """
        self.id = id
        self.num_vals = num_vals
        self.local_marginal_probs = pd.DataFrame({f"X{self.id}": [i for i in range(num_vals)]})
        self.parents = []
        self.cur_val = -1
        self.alpha = alpha
        self.local_conditional_probs = None
        
    def set_parents(self, parents):
        """
            parents: List[Node]
        """
        # print(f"Node {self.id} have parents: {[pa.id for pa in parents]}")
        self.parents = parents
        return
        
    def get_level(self):
        if len(self.parents):
            return max([pa.get_level() for pa in self.parents]) + 1
        else:
            return 0
    
    def set_condprob(self):
        if len(self.parents):
            self.local_conditional_probs = self.local_marginal_probs.copy()
            pa_comb = 1
            for pa in self.parents:
                pa_comb *= pa.num_vals
                self.local_conditional_probs = self.local_conditional_probs.merge(pa.local_marginal_probs, how="cross")
            
            # Generating conditional probability, constraints: sum (P(X=x|Z=z)) = 1 for all x
            self.local_conditional_probs['prob'] = [1] * len(self.local_conditional_probs)
            self.local_conditional_probs = self.local_conditional_probs.sort_values(by=[f'X{pa.id}' for pa in self.parents])
            for i in range(pa_comb):
                self.local_conditional_probs['prob'].iloc[self.num_vals*i:self.num_vals*(i+1)] = np.random.dirichlet([self.alpha] * self.num_vals).flatten()
        return
    
    def set_margprob(self, dirichlet_alpha=1):
        if not len(self.parents):
            probs = np.random.dirichlet([dirichlet_alpha] * self.num_vals).flatten()
            try:
                old_probs = self.local_marginal_probs['prob']
                while np.sqrt((old_probs - probs).pow(2).sum()) < 0.25:
                    probs = np.random.dirichlet([dirichlet_alpha] * self.num_vals).flatten()
            except:
                pass
                
            self.local_marginal_probs['prob'] = probs.flatten()
        return
    
    def sample(self):
        if len(self.parents):
            conditions = np.array([True for _ in range(len(self.local_conditional_probs))]) # type:ignore
            for pa in self.parents:
                select = (self.local_conditional_probs[f'X{pa.id}'] == pa.cur_val).to_numpy()   # type:ignore
                conditions = np.logical_and(conditions,select)
            cond_df = self.local_conditional_probs[conditions]  # type:ignore
            self.cur_val = np.random.choice(cond_df[f'X{self.id}'], size=1, p=cond_df[f'prob']).item()
        else:
            self.cur_val = np.random.choice(self.local_marginal_probs[f'X{self.id}'], size=1, p=self.local_marginal_probs[f'prob']).item()
        return
        

class DAG:
    def __init__(self, adj_mtx, max_numvals, alpha) -> None:
        self.adj_mtx = adj_mtx
        self.nodes = [Node(np.random.randint(2, max_numvals), i + 1, alpha) for i in range(adj_mtx.shape[0])]
        self.__endogeneous_nodes = []
        self.__render_adjmtx()
        self.__order_nodes()
        self.__init_condprobs()
        self.__init_margprob()
        
    def __render_adjmtx(self):
        for i in range(self.adj_mtx.shape[0]):
            parents = [self.nodes[j] for j in range(self.adj_mtx.shape[0]) if self.adj_mtx[j][i] == 1]
            self.nodes[i].set_parents(parents)  # type:ignore
            if len(parents) == 0:
                self.__endogeneous_nodes.append(self.nodes[i])
        return
    
    def __order_nodes(self):
        self.nodes.sort(key=lambda item: item.get_level())
        return
    
    def __init_condprobs(self):
        for node in self.nodes:
            node.set_condprob()    # type:ignore
        return
    
    def __init_margprob(self):
        for node in self.nodes:
            node.set_margprob(1.0)    # type:ignore
        return
    
    def reinit_endoprob(self, dirichlet_alpha):
        chosen_id = np.random.choice([i for i in range(len(self.__endogeneous_nodes))], size=1).item()
        self.__endogeneous_nodes[chosen_id].set_margprob(dirichlet_alpha)
        return
    
    def disseminate(self, n):
        df = pd.DataFrame(columns=[f'X{node.id}' for node in self.nodes])
        for i in tqdm(range(n), leave=False):
            res = []
            for node in self.nodes:
                # print(f"X{node.id}-->", end="")
                node.sample()
                res.append(node.cur_val)
            # print("|")
            df.loc[len(df),:] = res  # type:ignore
        return df
    
    
import os, argparse
from pathlib import Path
from utils import is_acyclic

def gen_data(dag, n=10000, savepath="./data", filename="output.csv"):
    df = dag.disseminate(n)
    
    if filename is not None:
        res_path = os.path.join(savepath)
        if not Path(res_path).exists():
            os.makedirs(res_path)
        
        df.to_csv(os.path.join(res_path, filename), index=False)
    
    return df


import networkx as nx
import numpy as np
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj_path", type=str, default=None)
    parser.add_argument("--num_node", type=int, default=10)
    parser.add_argument("--mi", type=int, default=3)
    parser.add_argument("--di", type=float, default='1')
    parser.add_argument("--p", type=float, default='0.1')
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--s", type=int, default=5000)
    options = vars(parser.parse_args())
    
    num_node = options['num_node']
    p = options['p']
    mi = options['mi']
    di = options['di']
    n = options['n']
    s = options['s']
    
    if options['adj_path'] is None:
        graph = nx.erdos_renyi_graph(n=num_node, p=p, seed=0, directed=True)
        try:
            while True:
                cycle_list = nx.find_cycle(graph, orientation='original')
                if len(cycle_list):
                    random.shuffle(cycle_list)
                    a, b, type = cycle_list[0]
                    graph.remove_edge(a, b)
                else:
                    break
        except:
            pass

        adj_mtx = np.zeros([num_node, num_node])
        for edge in graph.edges:
            a, b = edge
            adj_mtx[a][b] = 1
            
        outdegrees = np.sum(adj_mtx, axis=0, keepdims=True)
        indegrees = np.sum(adj_mtx, axis=1, keepdims=True)

        if outdegrees.max() < indegrees.max():
            adj_mtx = adj_mtx.T
    else:
        adj_mtx = np.loadtxt(options['adj_path'], delimiter=' ')
        print("Loaded adj matrix")

    dag = DAG(adj_mtx, max_numvals=mi, alpha=di)
    
    for i in range(n):
        dag.reinit_endoprob(dirichlet_alpha=1.)
        df = gen_data(dag, s, savepath=f"../data/categorical/erdos_renyi/d{num_node}_p{p}/m{mi}_d{di}_n{n}", filename=f"silo-{i}.csv") # f"silo-{i}.csv"

    with open(f"../data/categorical/erdos_renyi/d{num_node}_p{p}/adj.txt", "w") as f:
        np.savetxt(f, adj_mtx)