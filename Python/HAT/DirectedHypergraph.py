import numpy as np
import scipy as sp
import scipy.linalg
import scipy.stats
from itertools import permutations
import networkx as nx

import HAT.multilinalg as mla
import HAT.draw
import HAT.HAT

class DirectedHypergraph:
    """This class represents directed hypergraphs and implements directed hypergraph walks, graph expansions,
    and several other features.

    A directed hypergraph :math:`D=(V,E)` is a set of vertices :math:`V` and a set of edges :math:`E`
    where each edge :math:`e\in E` is :math:`e\subseteq V`and may be partitioned into a head and tail set indicating
    the flow or direction along the hyperedge i.e. :math:`e=e^h\cup e^t`. 
    
    We focus on tail-uniform directed hypergraphs where the tails of all hyperedges are the same size. Tail uniform
    directed hypergraphs admit uniform tensor representation and have dynamics described as homogeneous polynomials.

    :param im: Incidence matrix
    :param es: Hyperedge set
    """
    def __init__(self, im=None, es=None):
        # Auth: Joshua Pickard
        #       jpic@umich.edu
        # Date: Oct 31, 2023
        # if type(im) == type(None) and type(es) == type(None)
        self.IM = im
        self.edgeSet = es

    def Hypergraph(self):
        """
        Constructs an undirected `Hypergraph` object. All vertices in the head and tail are treated as belonging to the
        same hyperedge with no distinciton.
        """
        # Converts the hypergraph to an undirected hypergraph
        UIM = (self.IM != 0).astype(int)
        HG = HAT.Hypergraph(IM)
        return HG

    def isStructurallyControllable(self, driver_nodes):
        """
        This function determines if a directed hypergraph is structurally controllable by 1) applying a hypergraph cascade beginning
        at the driver_nodes and 2) solving the star graph maximum matching problem.
        """
        # Hypergraph cascade
        accessibleHEdges, remainingHyperedges = self.getAccessibleHyperedges(accessibleVxc)
        i = 0
        while len(accessibleVxc) < n and len(accessibleHEdges) > 0:
            print("Itr: " + str(i) + ", Num Accessible: " + str(len(accessibleVxc)))
            numAccessible = len(accessibleVxc)
            for e in accessibleHEdges:
                accessibleVxc.append(e[1])
            accessibleVxc = list(set(accessibleVxc))
            if numAccessible == len(set(accessibleVxc)):
                break
            accessibleHEdges, remainingHyperedges = self.getAccessibleHyperedges(accessibleVxc, HEdges=remainingHyperedges)
            i += 1
        if len(accessibleVxc) < self.IM.shape[0]:
            return False

        # Star graph maximum matching
        S = HG.starGraph()
        matching = sorted(nx.maximal_matching(S))
        matchingSize = len(matching)
        if matchingSize < self.IM.shape[0]:
            return False
        return True
    
    def controlNodeSelection(self):
        """
        This function selects optimal control nodes on a directed hypergraph using XXX
        """
        return None
        
    def controllableSpace(self, driver_nodes):
        """
        This function determines which states/nodes can be controlled given a set of driver nodes.
        """
        return None
    
    def starGraph():
        """
        Constructs a directed star graph.
        """
        n = len(self.IM) + len(self.IM[0])
        A = np.zeros((n,n))
        A[len(A) - len(self.IM):len(A),0:len(self.IM[0])] = self.IM
        A[0:len(self.IM[0]),len(A) - len(self.IM):len(A)] = self.IM.T
        num_nodes = A.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if A[i, j] > 0:
                    G.add_edge(i, j)
                elif A[i, j] < 0:
                    G.add_edge(j, i)
        return G
    
    def cliqueGraph():
        """
        Constructs a directed clique graph. (note there are some difficulties here)
        """
        edgeOrder = np.sum(self.IM, axis=0)
        A = self.IM @ self.IM.T
        np.fill_diagonal(A,0) # Omit self loops
        num_nodes = A.shape[0]
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if A[i, j] > 0:
                    G.add_edge(i, j)
                elif A[i, j] < 0:
                    G.add_edge(j, i)
        return G
    
    def getAccessibleHyperedges(self, aVxc, HEdges=None):
        """
        This function finds the set of hyperedges whose tails are accessible from the accessible vertices aVxc
        """
        # this function supposes that all vertices are individually accessible
        if type(HEdges) == type(None):
            HEdges = self.getHyperedgeSet
        aVxc = set(aVxc)
        aHEdges, rHEdges = [], []
        for e in HEdges:
            if set(e[0]).issubset(aVxc):
                aHEdges.append(e)
            else:
                rHEdges.append(e)
        return aHEdges, rHEdges
    
    def adjTensors(self):
        if self.maxHeadSize == 1:
            HG = self.Hypergraph()
            return HG.adjTensor()
        return None