using Plots, LightGraphs
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
N1, N2 = 11, 7; G = LightGraphs.grid([N1,N2]); N = nv(G)
X = zeros(N1, N2, 2); for i in 1:N1; for j in 1:N2; X[i,j,1] = i; X[i,j,2] = j; end; end; X = reshape(X, (N,2))
L = Matrix(laplacian_matrix(G))
lamb, 𝛷 = eigen(L)
𝛷 = (𝛷' .* sign.(𝛷[1,:]))'
Q = incidence_matrix(G; oriented = true)

## non-trivial eigenvector metric
distDAG = eigDAG_Distance(V, Q, N)

## test Soft Clustering NGW frame
Ψ = SC_NGW_frame(distROT, 𝛷; σ = 0.3, β = 4)
scatter_gplot(X; marker = Ψ[50,30,:], ms = 10)
