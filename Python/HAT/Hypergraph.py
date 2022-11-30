import numpy as np
import scipy as sp
import scipy.linalg
# import scipy.stats
from itertools import permutations
import networkx as nx

import HAT.multilinalg as mla
import HAT.draw

class Hypergraph:
    """This is the base class representing a Hypergraph object. It is the primary entry point and
    provides an interface to functions implemented in HAT's other modules. The underlying data
    structure of this class is an incidence matrix, but many methods exploit tensor representation
    of uniform hypergraphs.

    Formally, a Hypergraph :math:`H=(V,E)` is a set of vertices :math:`V` and a set of edges :math:`E`
    where each edge :math:`e\in E` is defined :math:`e\subseteq V.` In contrast to a graph, a hypergraph
    edge :math:`e` can contain any number of vertices, which allows for efficient representation of multi-way
    relationships.
    
    In a uniform Hypergraph, all edges contain the same number of vertices. Uniform hypergraphs are represnted
    as tensors, which precisely model multi-way interactions.

    :param im: Incidence matrix
    :param ew: Edge weight vector
    :param nw: Node weight vector
    """
    def __init__(self, im, ew=None, nw=None):
        self.IM = im
        self.edgeWeights = ew
        self.nodeWeights = nw

        if self.edgeWeights is None:
            self.edgeWeights = np.ones(self.IM.shape[1])
        if self.nodeWeights is None:
            self.nodeWeights = np.ones(self.IM.shape[0])
    
    def draw(self, shadeRows=True, connectNodes=True, dpi=200, edgeColors=None):
        """ This function draws the incidence matrix of the hypergraph object. It calls the function
        ``HAT.draw.incidencePlot``, but is provided to generate the plot directly from the object.

        :param shadeRows: shade rows (bool)
        :param connectNodes: connect nodes in each hyperedge (bool)
        :param dpi: the resolution of the image (int)
        :param edgeColors: The colors of edges represented in the incidence matrix. This is random by default
        
        :return: matplotlib axes with figure drawn on to it
        """
        return HAT.draw.incidencePlot(self, shadeRows=shadeRows, connectNodes=connectNodes, dpi=dpi, edgeColors=edgeColors)
        
    def dual(self):
        """The dual hypergraph is constructed.

        Let :math:`H=(V,E)` be a hypergraph. In the dual hypergraph each original edge :math:`e\in E`
        is represented as a vertex and each original vertex :math:`v\in E` is represented as an edge. Numerically, the
        transpose of the incidence matrix of a hypergraph is the incidence matrix of the dual hypergraph.

        :return: Hypergraph object
        
        References
        ----------
        .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        IM = self.IM.T
        return Hypergraph(IM)

    def cliqueGraph(self):
        """The clique expansion graph is constructed.

        The clique expansion algorithm constructs a *graph* on the same set of vertices as the hypergraph by defining an
        edge set where every pair of vertices contained within the same edge in the hypergraph have an edge between them
        in the graph. Given a hypergraph :math:`H=(V,E_h)`, then the corresponding clique graph is :math:`C=(V,E_c)` where

        .. math:

            E_c = \{(v_i, v_j) |\ \exists\  e\in E_h \text{ where }v_i, v_j\in e\}.

        This is called clique expansion because the vertices contained in each :math:`h\in E_h` forms a clique in :math:`C`.
        While the map from :math:`H` to :math:`C` is well-defined, the transformation to a clique graph is a lossy process,
        so the hypergraph structure of :math:`H` cannot be uniquely recovered from the clique graph :math:`C` alone [1].

        References
        ----------
        .. [1] Amit Surana, Can Chen, and Indika Rajapakse. Hypergraph similarity measures. IEEE Transactions on Network Science and Engineering, pages 1-16, 2022.
        .. [2] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        edgeOrder = np.sum(self.IM, axis=0)
        M = np.zeros((len(edgeOrder),len(edgeOrder)))
        np.fill_diagonal(M, self.edgeWeights)
        A = self.IM @ M @ self.IM.T
        np.fill_diagonal(A,0) # Omit self loops
        return nx.from_numpy_matrix(A)

    def lineGraph(self):
        """The line graph, which is the clique expansion of the dual graph, is constructed.
        
        References
        ----------
        .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        D = self.dual()
        return D.cliqueGraph()

    def starGraph(self):
        """The star graph representation is constructed.
        
        References
        ----------
        .. [1] Yang, Chaoqi, et al. "Hypergraph learning with line expansion." arXiv preprint arXiv:2005.04843 (2020).
        """
        
        n = len(self.IM) + len(self.IM[0])
        A = np.zeros((n,n))
        A[len(A) - len(self.IM):len(A),0:len(self.IM[0])] = self.IM
        A[0:len(self.IM[0]),len(A) - len(self.IM):len(A)] = self.IM.T
        return nx.from_numpy_matrix(A)
    
    def laplacianMatrix(self, type):
        if type == 'Bolla':
            return self.bolla()
        elif type == 'Rodriguez':
            return self.rodriguez()
        elif type == 'Zhou':
            return self.zhou()
        
    def bolla(self):
        """
        References
        ----------
        .. [1] Bolla, M. (1993). Spectra, euclidean representations and clusterings of hypergraphs. Discrete Mathematics, 117.
               https://www.sciencedirect.com/science/article/pii/0012365X9390322K
        """
        dv = np.sum(self.IM, axis=1)
        Dv = np.zeros((len(dv), len(dv)))
        np.fill_diagonal(Dv, dv)
        de = np.sum(self.IM, axis=0)
        De = np.zeros((len(de), len(de)))
        np.fill_diagonal(De, de)
        DeInv = sp.linalg.inv(De, overwrite_a=True)
        L = Dv - (self.IM @ DeInv @ self.IM.T)
        return L
        
    def rodriguez(self):
        """
        References
        ----------
        .. [1] Rodriguez, J. A. (2002). On the Laplacian eigenvalues and metric parameters of hypergraphs. Linear and Multilinear Algebra, 50(1), 1-14.
               https://www.tandfonline.com/doi/abs/10.1080/03081080290011692
        .. [2] Rodriguez, J. A. (2003). On the Laplacian spectrum and walk-regular hypergraphs. Linear and Multilinear Algebra, 51, 285–297.
               https://www.tandfonline.com/doi/abs/10.1080/0308108031000084374
        """
        A = self.IM @ self.IM.T
        np.fill_diagonal(A, 0)
        L = np.diag(sum(A)) - A
        return L
    
    def zhou(self):
        """
        References
        ----------
        .. [1] Zhou, D., Huang, J., & Schölkopf, B. (2005). Beyond pairwise classification and clustering using hypergraphs. (Equation 3.3)
               https://dennyzhou.github.io/papers/hyper_tech.pdf
        """
        Dvinv = np.diag(1/np.sqrt(self.IM @ self.edgeWeights))
        Deinv = np.diag(1/np.sum(self.IM, axis=0))
        W = np.diag(self.edgeWeights)
        L = np.eye(len(self.IM)) - Dvinv @ self.IM @ W @ Deinv @ self.IM.T @ Dvinv
        return L        
        
    def adjTensor(self):
        """This constructs the adjacency tensor for uniform hypergraphs.
        """
        order = int(sum(self.IM[:,0]))
        denom = np.math.factorial(order - 1)
        modes = len(self.IM) * np.ones(order)
        A = np.zeros(modes.astype(int))
        for e in range(len(self.IM[0])):
            vxc = np.where(self.IM[:, e] == 1)[0]
            vxc = list(permutations(vxc))
            for p in vxc:
                A[p] = 1 / denom
        return A
    
    def degreeTensor(self):
        """This constructs the degree tensor for uniform hypergraphs.
        """
        order = int(sum(self.IM[:,0]))
        modes = len(self.IM) * np.ones(order)
        D = np.zeros(modes.astype(int))
        for vx in range(len(self.IM)):
            D[tuple((np.ones(order) * vx).astype(int))] = sum(self.IM[vx])
        return D
    
    def laplacianTensor(self):
        """This constructs the Laplacian tensor for uniform hypergraphs.
        """
        D = self.degreeTensor()
        A = self.adjTensor()
        L = D - A
        return L
        
    def tensorEntropy(self):
        """Computes hypergraph entropy based on the singular values of the Laplacian tensor. This comes from definition 7 of [1] and
        is implemented with Algorithm 1 from [1].
        
        References
        ----------
        .. [1] C. Chen and I. Rajapakse, Tensor Entropy for Uniform Hypergraphs, IEEE TRANSACTIONS ON NETWORK SCIENCE AND ENGINEERING (2020)
        """
        
        L = self.laplacianTensor()
        _, S, _ = mla.hosvd(L, M=False, uniform=True)
        return sp.stats.entropy(S)
    
    def matrixEntropy(self, Ltype='Rodriguez'):
        """Computes hypergraph entropy based on the eigenvalues values of the Laplacian matrix.
        """
        L = self.laplacianMatrix(Ltype)
        U, V = np.linalg.eig(L)
        return sp.stats.entropy(U)
    
    def avgDistance(self):
        """Computes the average pairwise distance between any 2 vertices in the hypergraph.

        The hypergraph is clique expanded to a graph object, and the average shortest path on
        the clique expanded graph is returned.

        :return: avgDist
        :rtype: float
        """
        G = self.cliqueGraph()
        avgDist = nx.average_shortest_path_length(G)
        return avgDist
    
    def ctrbk(self, inputVxc):
        A = self.adjTensor()
        modes = A.shape
        n = modes[0]
        order = len(modes)
        Aflat = np.reshape(A, (n, n**(order-1)))
        # print(Aflat.shape)
        ctrbMatrix = self.bMatrix(inputVxc)
        print(ctrbMatrix.shape)
        j = 0
        while j < n and np.linalg.matrix_rank(ctrbMatrix) < n:
            kprod = mla.kronExponentiation(ctrbMatrix, len(modes)-1)
            # print(kprod.shape)
            nextCtrbMatrix = Aflat @ kprod;
            # print(nextCtrbMatrix.shape)
            ctrbMatrix = np.concatenate((ctrbMatrix, nextCtrbMatrix), axis=1)
            print(ctrbMatrix.shape)
            r = np.linalg.matrix_rank(ctrbMatrix)
            U, _, _ = sp.linalg.svd(ctrbMatrix)
            ctrbMatrix = U[:,0:r]
            print(ctrbMatrix.shape)
            j += 1
        return ctrbMatrix
        
    def bMatrix(self, inputVxc):
        """Constructs controllability :math:`B` matrix commonly used in the linear control system
        
        .. :math:
            \frac{dx}{dt} = Ax+Bu

        :param inputVxc: a list of input control nodes
        :type inputVxc: ``np.array``
        :return: control matrix
        :rtype: ``np.array``

        References
        ==========
        .. [1] Can Chen, Amit Surana, Anthony M Bloch, and Indika Rajapakse. Controllability of hypergraphs. IEEE Transactions
        on Network Science and Engineering, 8(2):1646–1657, 2021. https://drive.google.com/file/d/12aReE7mE4MVbycZUxUYdtICgrAYlzg8o/view
        """
        B = np.zeros((len(self.nodeWeights), len(inputVxc)))
        for i in range(len(inputVxc)):
            B[inputVxc[i], i] = 1
        return B
    
    def clusteringCoef(self):
        """Computes clustering average clustering coefficient of the hypergraph.

        For a uniform hypergraph, the clustering coefficient of a vertex :math:`v_i`
        is defined as the number of edges the vertex participates in (i.e. :math:`deg(v_i)`) divided
        by the number of :math:`k-`way edges that could exist among vertex :math:`v_i` and its neighbors 
        (See equation 31 in [1]). This is written

        .. :math:
            C_i = \frac{deg(v_i)}{\binom{|N_i|}{k}}

        where :math:`N_i` is the set of neighbors or vertices adjacent to :math:`v_i`. The hypergraph
        clustering coefficient computed here is the average clustering coefficient for all vertices,
        written

        .. :math:
            C=\sum_{i=1}^nC_i

        :return: Average clustering coefficient
        :rtype: float

        References
        ==========
        .. [1] Surana, Amit, Can Chen, and Indika Rajapakse. "Hypergraph Similarity Measures."
        IEEE Transactions on Network Science and Engineering (2022).
        """
        n, e = self.IM.shape
        order = sum(self.IM[:,0])
        avgClusterCoef = 0
        for v in range(n):
            edges = np.where(self.IM[v,:] > 0)[0]
            neighbors = np.where(np.sum(self.IM[:, edges], axis=1) > 0)[0]
            avgClusterCoef += (len(edges) / sp.special.comb(len(neighbors), order))
        avgClusterCoef /= n
        return avgClusterCoef
                                   
    def centrality(self, tol=1e-4, maxIter=3000, model='LogExp', alpha=10):
        """Computes node and edge centralities.

        :param tol: threshold tolerance for the convergence of the centrality measures, defaults to 1e-4
        :type tol: _type_, optional
        :param maxIter: maximum number of iterations for the centrality measures to converge in, defaults to 3000
        :type maxIter: int, optional
        :param model: the set of functions used to compute centrality. This defaults to 'LogExp', and other choices include
        'Linear', 'Max' or a list of 4 custom function handles (See [1]).
        :type model: str, optional
        :param alpha: Hyperparameter used for computing centrality (See [1]), defaults to 10
        :type alpha: int, optional

        :return: vxcCentrality
        :rtype: *ndarray* containing centrality scores for each vertex in the hypergraph
        :return: edgeCentrality
        :rtype: *ndarray* containing centrality scores for each edge in the hypergraph
        
        References
        ----------
        .. [1] % Tudisco, F., Higham, D.J. Node and edge nonlinear eigenvector centrality for hypergraphs. Commun Phys 4, 201 (2021). https://doi.org/10.1038/s42005-021-00704-2
        """
        if model=='Linear':
            f = lambda x : x
            g = lambda x : x
            phi = lambda x : x
            psi = lambda x : x
        elif model=='LogExp':
            f = lambda x : x
            g = lambda x : np.sqrt(x)
            phi = lambda x : np.log(x)
            psi = lambda x : np.exp(x)
        elif model=='Max':
            f = lambda x : x
            g = lambda x : x
            phi = lambda x : x**alpha
            psi = lambda x : x**(1/alpha)
        elif isinstance(model, list):
            f = model[0]
            g = model[1]
            phi = model[2]
            psi = model[3]
        else:
            print('Enter a valid centrality model')
            
        B = self.IM
        W = self.edgeWeights
        N = self.nodeWeights
        W = np.diag(W)
        N = np.diag(N)
        n, m = B.shape        
        x0 = np.ones(n,) / n
        y0 = np.ones(m,) / m
        
        for i in range(maxIter):
            u = np.sqrt(np.multiply(np.array(x0),np.array(g(B @ W @ f(y0)))))
            v = np.sqrt(np.multiply(np.array(y0),np.array(psi(B.T @ N @ phi(x0)))))
            x = u / sum(u)
            y = v / sum(v)
            
            check = sum(x - x0) + sum(y - y0)
            if check < tol:
                break
            else:
                x0 = x
                y0 = y
                
        vxcCentrality = x
        edgeCentrality = y
        
        return vxcCentrality, edgeCentrality