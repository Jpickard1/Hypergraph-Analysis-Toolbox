import sys
sys.path.append('../')
from HAT import Hypergraph
import numpy as np

def constructor_1():
    print('constructor_1 start')
    edge_list = [[0,1,2],
                 [0,1,3]]
    HG = Hypergraph(edge_list=edge_list)
    print(HG.nodes)
    print(HG.edges)
    print(f"{HG.order=}")
    print(f"{HG.uniform=}")
    print(f"{HG.directed=}")
    print(f"{HG.incidence_matrix=}")
    print(f"{HG.adjacency_tensor=}")
    print('constructor_1 complete')

def constructor_2():
    print('constructor_2 start')
    A = np.array([[[0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0]],

       [[0, 0, 1, 1],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]],

       [[0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],

       [[0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]]
    )
    HG = Hypergraph(adjacency_tensor=A)
    print(HG.nodes)
    print(HG.edges)
    print(f"{HG.order=}")
    print(f"{HG.uniform=}")
    print(f"{HG.directed=}")
    print(f"{HG.incidence_matrix=}")
    print(f"{HG.adjacency_tensor=}")
    print('constructor_2 complete')

# def main():
constructor_1()
constructor_2()
