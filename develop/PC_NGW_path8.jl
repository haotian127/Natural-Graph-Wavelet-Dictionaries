## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
N = 8; G = path_graph(N)
X = zeros(N,2); X[:,1] = 1:N
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
V = (V' .* sign.(V[1,:]))'
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

## Build Dual Graph
dist_DAG = eigDAG_Distance(V,Q,N)
# W_dual = sparse(dualGraph(dist_DAG)) #sparse dual weighted adjacence matrix
W_dual = 1.0 * adjacency_matrix(path_graph(N))

## Verify the partition results
ht_elist_dual, ht_vlist_dual = HTree_EVlist(V,W_dual)
