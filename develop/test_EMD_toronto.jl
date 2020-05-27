using Plots, LightGraphs, JLD, Distances, MultivariateStats
include(joinpath("..", "src", "func_includer.jl"))

## Build weighted RGC#100 graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "new_toronto_graph.lgz")); N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"xy")
A = 1.0 .* adjacency_matrix(G)
dist_X = pairwise(Euclidean(),X; dims = 1)
Weight = A .* dualGraph(dist_X; method = "inverse") # weighted adjacence matrix
L = Matrix(Diagonal(sum(Weight;dims = 1)[:]) - Weight)
lamb, V = eigen(L)
sgn = (maximum(V, dims = 1)[:] .> -minimum(V, dims = 1)[:]) .* 2 .- 1
V = (V' .* sgn)'; V = Matrix(V);
Q = incidence_matrix(G; oriented = true)
edge_lengths = sqrt.(sum((Q' * X).^2, dims = 2)[:])
## Try out OptimalTransport.jl
using OptimalTransport, SimpleWeightedGraphs
sources, destinations =Int64[], Int64[]; for e in collect(edges(G)); push!(sources, e.src); push!(destinations, e.dst); end
wG = SimpleWeightedGraph(sources, destinations, edge_lengths) #edge weights are the Euclidean length of the edge
costmx = floyd_warshall_shortest_paths(G, weights(wG)).dists

P = V.^2

p, q = rand(N), rand(N); p, q = p/norm(p,1), q/norm(q,1);
print("=================\n")
@time _, d = ROT_Distance(p, q, Q; le = edge_lengths)
@time emdcost = emd2(p, q, costmx)
