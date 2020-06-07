using LightGraphs, SimpleWeightedGraphs, LinearAlgebra, SparseArrays

function SunFlowerGraph_weighted(;n = 600)
"""
weighted SunFlowerGraph
require: n > 26, Plots.jl, LightGraphs.jl, SimpleWeightedGraphs.jl and gplot.jl
argument: n, number of vertices;
outputs: simple weighted graph G; weighted Laplacian matrix L; vertices location info. xy
"""
    c=1.0/n; θ=(sqrt(5.0)-1)*π;
    xy = zeros(n,2);
    for k=1:n xy[k,:]=c*(k-1)*[cos((k-1)*θ) sin((k-1)*θ)]; end

    G = SimpleWeightedGraph(n)
    for k = 1:13
        G.weights[k,k+8] = 1/norm(xy[k,:] - xy[k+8,:])
        G.weights[k+8,k] = 1/norm(xy[k,:] - xy[k+8,:])
        G.weights[k,k+13] = 1/norm(xy[k,:] - xy[k+13,:])
        G.weights[k+13,k] = 1/norm(xy[k,:] - xy[k+13,:])
    end
    for k = 14:n
        G.weights[k,k-8] = 1/norm(xy[k,:] - xy[k-8,:])
        G.weights[k-8,k] = 1/norm(xy[k,:] - xy[k-8,:])
        G.weights[k,k-13] = 1/norm(xy[k,:] - xy[k-13,:])
        G.weights[k-13,k] = 1/norm(xy[k,:] - xy[k-13,:])
    end
    W = weights(G) #weighted adjacency_matrix
    Lw = Diagonal(sum(W, dims = 2)[:]) - W #weighted laplacian_matrix
    return G, Lw, xy
end
