using Plots, LightGraphs
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
N1, N2 = 11, 7; G = LightGraphs.grid([N1,N2]); N = nv(G)
X = zeros(N1, N2, 2); for i in 1:N1; for j in 1:N2; X[i,j,1] = i; X[i,j,2] = j; end; end; X = reshape(X, (N,2))
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
V = (V' .* sign.(V[1,:]))'
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

## Test ROT distance
# @time distROT = eigROT_Distance(V.^2, Q)
p, q = [rand(20); zeros(N-20)], [zeros(N-20); rand(20)]; p, q = p/norm(p,1), q/norm(q,1);
@time wt, d = ROT_Distance(p, q, Q)
#
# E = collect(edges(G))
# m = length(E)
# selectedEdgeIdx = findall(wt .> 1e-4)
#
# gplot(W, X); scatter_gplot!(X; marker = p, ms = 200 .* abs.(p)); scatter_gplot!(X; marker = q, ms = 200 .* abs.(q))
# for i in selectedEdgeIdx
#     if i > m
#         plot!([X[E[i-m].dst,1] X[E[i-m].src,1]]', [X[E[i-m].dst,2] X[E[i-m].src,2]]', linecolor = :red, linewidth = 3 * wt[i] / maximum(wt), arrow = 0.4, aspect_ratio = 1, legend = false)
#     else
#         plot!([X[E[i].src,1] X[E[i].dst,1]]', [X[E[i].src,2] X[E[i].dst,2]]', linecolor = :red, linewidth = 3 * wt[i] / maximum(wt), arrow = 0.4, aspect_ratio = 1, legend = false)
#     end
# end
# current()

## try OptimalTransport.jl
using OptimalTransport
distmx = floyd_warshall_shortest_paths(G).dists
@time emdcost = emd2(p, q, 1.0.*distmx)
