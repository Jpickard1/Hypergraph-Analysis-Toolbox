classdef Hypergraph
    %HYPERGRAPH Class to store a hypergraph, summarize it, make
    %   computations on it, and plot it. 
    
    % This class treats rows of the incidence matrix as nodes and the
    % columns as hyperedges. 
    properties
        IM (:,:) % incidence matrix 
        edgeWeights 
        nodeWeights
        edgeNames
        nodeNames
    end
    
    methods
        function obj = Hypergraph(nameValueArgs)
            %HYPERGRAPH Construct an instance of this class.
            %   Takes in any 2D array representing an incidence matrix and
            %   stores it in sparse format. Also can store the hyperedge
            %   set for uniform hypergraphs.
            arguments
                nameValueArgs.IM = sparse(1);
                nameValueArgs.edgeWeights = 0;
                nameValueArgs.nodeWeights = 0;
                nameValueArgs.edgeNames = 0;
                nameValueArgs.nodeNames = 0;
            end
            obj.IM = sparse(nameValueArgs.IM);
            if nameValueArgs.edgeWeights == 0
                nameValueArgs.edgeWeights = ones(size(obj.IM, 2), 1);
            end
            if nameValueArgs.nodeWeights== 0
                nameValueArgs.nodeWeights = ones(size(obj.IM, 1), 1); 
            end
            if nameValueArgs.edgeNames == 0
                nameValueArgs.edgeNames = 1:size(obj.IM, 2);
            end
            if nameValueArgs.nodeNames== 0
                nameValueArgs.nodeNames = 1:size(obj.IM, 1); 
            end
            obj.edgeWeights = nameValueArgs.edgeWeights;
            obj.nodeWeights = nameValueArgs.nodeWeights;
            obj.edgeNames = nameValueArgs.edgeNames;
            obj.nodeNames = nameValueArgs.nodeNames;
        end
        
        %% Summarization
        function m = numRows(obj)
            %NUMROWS Get the number of rows in the incidence matrix.
            m = size(obj.IM, 1);
        end

        function n = numCols(obj)
            %NUMCOLS Get the number of columns in the incidence matrix.
            n = size(obj.IM, 2);
        end

        function d = density(obj)
            %DENSITY Gets the density of the underlying incidence matrix.
            %If this is less than (m*(n-1)-1)/2, then the matrix is so
            %dense that storing it in CSC format takes up more memory than
            %dense format.
            d = nnz(obj.IM)/(obj.numCols * obj.numRows);
        end

        function t = CSCThreshold(obj)
            %CSCTHRESHOLD Gets the density threshold over which it saves
            %space to store the matrix in dense format.
            m = obj.numRows;
            n = obj.numCols;
            t = (m*(n-1)-1)/(2*m*n);
        end
        
        function sz = edgeSizes(obj)
            %EDGESIZES Get the number of nodes in each edge.
            sz = sum(obj.IM, 1);
        end

        function dg = nodeDegrees(obj)
            %NODEDEGREES Get the degree of each node.
            dg = sum(obj.IM, 2);
        end

        % fun decls
        
        % Returns the s-connected components of the hypergraph.
        % s: the minimum connecting edge size. When s=1, this function
        %   returns the connected components of the clique expansion.
        % outputForm {"vector" (default), "cell"}: arg to MATLAB's 
        %   conncomp function. Specifies the output form of the connected 
        %   components. 
        [bins, binSize]  = sConnectedComponents(obj, s, outputForm)

        r = sRadius(obj, s);

        d = sDiameter(obj, s);

        %% Representation
        function A = adjTensor(obj)
            A = HAT.TensorRepresentation.adjacencyTensor(obj);
        end

        function D = degreeTensor(obj)
            D = HAT.TensorRepresentation.degreeTensor(obj);
        end

        function L = laplacianTensor(obj)
            L = obj.degreeTensor - obj.adjTensor;
        end

        function C = cliqueGraph(obj)
            C = HAT.GraphRepresentation.cliqueGraph(obj);
        end

        function S = starGraph(obj)
            S = HAT.GraphRepresentation.starGraph(obj);
        end

        function L = lineGraph(obj)
            L = HAT.GraphRepresentation.lineGraph(obj);
        end

        function HG = dual(obj)
            HG = Hypergraph(obj.IM');
        end

        function [A, L] = laplacianMatrix(obj, type)
            if nargin == 1
                warning("Enter Matrix Laplacian Type: Bolla, Rodriguez or Zhou");
                return
            end
            if strcmp(type, "Bolla")
                [A, L] = HAT.GraphRepresentation.BollaLaplacian(obj);
            elseif strcmp(type, "Rodriguez")
                [A, L] = HAT.GraphRepresentation.RodriguezLaplacian(obj);
            elseif strcmp(type, "Zhou")
                [A, L] = HAT.GraphRepresentation.ZhouLaplacian(obj);
            end
        end

        %% Computation
        function E = tensorEntropy(obj)
            E = HAT.tensorEntropy(obj);
        end

        function M = matrixEntropy(obj)
            M = HAT.matrixEntropy(obj);
        end

        function B = ctrbk(obj, driverNodes)
            B = HAT.ctrbk(obj, driverNodes);
        end

        function A = avgDistance(obj)
            A = HAT.averageDistance(obj);
        end

        function c = clusteringCoef(obj)
            c = HAT.clusteringCoefficient(obj);
        end

        %{
        function c = centrality(obj, NameValueArgs)
            c = HAT.centrality(obj, NameValueArgs);
        end
        %}

        %% Visualization
        function ax = plot(obj)
            ax = HAT.plotIncidenceMatrix(obj);
        end

    end
end

