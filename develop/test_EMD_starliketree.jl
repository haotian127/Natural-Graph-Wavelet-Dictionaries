using Plots, LightGraphs
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
G, X = StarLikeTree(70,16); N = nv(G)
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
V = (V' .* sign.(V[1,:]))'
Q = incidence_matrix(G; oriented = true)
# W = 1.0 * adjacency_matrix(G)

# Visualize the star-like tree graph
# gplot(W, X; shape = :circle, mwidth = 10)

## Test ROT distance
p, q = rand(N), rand(N); p, q = p/norm(p,1), q/norm(q,1);
u, v = zeros(N), zeros(N);
u = (p-q .> 0) .* (p-q); v =  - (p-q .< 0) .* (p-q);
# @time wt, d = ROT_Distance(p, q, Q)
@time wt2, d2 = ROT_Distance(u, v, Q)

## try OptimalTransport.jl
using OptimalTransport
distmx = floyd_warshall_shortest_paths(G).dists
# @time emdcost = emd2(p, q, 1.0.*distmx)
@time emdcost2 = emd2(u, v, 1.0.*distmx)

print("\n Relative difference between results = ", norm(emdcost - d) / norm(d))

# runtime of EMD on u,v is much faster to p,q
