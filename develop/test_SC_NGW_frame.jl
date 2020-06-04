## Include all the necessary packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
N1, N2 = 11, 7; G = LightGraphs.grid([N1,N2]); N = nv(G)
X = zeros(N1, N2, 2); for i in 1:N1; for j in 1:N2; X[i,j,1] = i; X[i,j,2] = j; end; end; X = reshape(X, (N,2))
L = Matrix(laplacian_matrix(G))
lamb, 𝛷 = eigen(L)
𝛷 = (𝛷' .* sign.(𝛷[1,:]))'
Q = incidence_matrix(G; oriented = true)

## non-trivial eigenvector metric
distDAG = eigDAG_Distance(𝛷, Q, N)
# aHAD = eigHAD_Affinity(𝛷,lamb)
distHAD = eigHAD_Distance(𝛷,lamb)

## test Soft Clustering NGW frame
Ψ = SC_NGW_frame(distDAG, 𝛷; σ = 0.3, β = 4)
scatter_gplot(X; marker = Ψ[6,30,:], ms = 10)

## test TFSC_NGW_frame
M = 3
graphClusters = spectral_clustering(𝛷, M)
activeEigenVecs = find_active_eigenvectors(𝛷, M, graphClusters)
partial_dist_ls = partialEig_Distance(graphClusters, activeEigenVecs, lamb, 𝛷, Q, L; eigen_metric = :TSD)

TF_Ψ = TFSC_NGW_frame(partial_dist_ls, 𝛷, M, graphClusters, activeEigenVecs; σ = 0.3, β = 4)[2]
scatter_gplot(X; marker = TF_Ψ[6,30,:], ms = 10)
