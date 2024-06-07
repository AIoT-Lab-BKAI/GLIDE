import numpy as np
from itertools import combinations

class Point:
    def __init__(self, coord, bys, h=0):
        self.coord = coord
        self.parents = bys
        self.h = h
        self.greatest_ancestors = []
        if self.parents is not None:
            for pa in self.parents:
                if len(pa.greatest_ancestors):
                    for ga in pa.greatest_ancestors:
                        if ga not in self.greatest_ancestors:
                            self.greatest_ancestors.append(ga)
                else:
                    self.greatest_ancestors.append(pa)
    
    def distance_to(self, other):
        return np.sqrt(np.power(self.coord - other.coord, 2).sum())

class Edge:
    def __init__(self, endpoint1: Point, endpoint2: Point):
        self.A = endpoint1
        self.B = endpoint2
        self.length = self.A.distance_to(self.B)
        self.mid_point = Point(0.5 * (self.A.coord + self.B.coord), bys=[endpoint1, endpoint2], h=max(self.A.h, self.B.h)+1)


def generate_distributions_singlevar(original_dis: np.ndarray, gamma1=0.1, gamma2=0.8):
    P0 = Point(original_dis, bys=None)
    Ulist = list(np.eye(P0.coord.shape[0]))
    
    # Compute the boundary points
    boundaries = []
    for i in range(len(Ulist)):
        if P0.coord[i]/gamma2 < 1:
            alpha_i = 1/(1 - P0.coord[i]) * (1 - P0.coord[i]/gamma2)
            boundary_i = Point(alpha_i * P0.coord + (1 - alpha_i) * Ulist[i], bys=None)
        else:
            boundary_i = Point(Ulist[i], bys=None)
        boundaries.append(boundary_i)

    # Find bounding edges
    edges = [] # A queue
    if len(boundaries) >= 3:
        for comb in list(combinations(boundaries + [P0], 2)):
            A, B = comb
            e = Edge(A,B)
            if e.length > 2*gamma1:
                edges.append(e)
    else:
        for point in boundaries:
            edges.append(Edge(P0, point))

    # Find intermediate points
    points = []
    while len(edges):
        edge = edges.pop(0)
        m_point = edge.mid_point
        points.append(m_point)
        
        half_edgeA = Edge(m_point, edge.A)
        if half_edgeA.length > 2*gamma1:
            edges.append(half_edgeA)
            
        half_edgeB = Edge(m_point, edge.B)
        if half_edgeB.length > 2*gamma1:
            edges.append(half_edgeB)




