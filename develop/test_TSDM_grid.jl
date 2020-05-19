using Plots, LightGraphs
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
N1, N2 = 11, 7; G = LightGraphs.grid([N1,N2]); N = nv(G)
X = zeros(N1, N2, 2); for i in 1:N1; for j in 1:N2; X[i,j,1] = i; X[i,j,2] = j; end; end; X = reshape(X, (N,2))
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
V = Matrix((V' .* sign.(V[1,:]))')
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

# create a mutable struct
grid_graph = GraphStruct(G,X,N,L,lamb,V)

## Test TSDM distance and compare with ROT distance
p, q = [rand(20); zeros(N-20)], [zeros(N-20); rand(20)]; p, q = p/norm(p,1), q/norm(q,1);
dTSD = TSD_Distance(p,q,V,lamb,Q,L;m = "Inf",dt = 0.1,tol = 1e-5)
_, dROT = ROT_Distance(p, q, Q)
print("d_TSD(p, q; T = ∞) = ", dTSD, "\n", "d_ROT(p,q; α = 1) = ", dROT)
