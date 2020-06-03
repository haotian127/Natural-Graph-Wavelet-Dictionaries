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

aHAD = eigHAD_Affinity(𝛷,lamb,N)
distHAD = eigHAD_Distance(𝛷,lamb,N)

## test Soft Clustering NGW frame
Ψ = SC_NGW_frame(distDAG, 𝛷; σ = 0.3, β = 4)
scatter_gplot(X; marker = Ψ[6,30,:], ms = 10)

## test TFSC_NGW_frame
M = 3
graphClusters = spectral_clustering(𝛷, M)
activeEigenVecs = find_active_eigenvectors(𝛷, M, graphClusters)
partial_dist_ls = []
for k in 1:M
    # write a function for this
    J = length(activeEigenVecs[k])
    tmp_dist = zeros(N,N); for i in 1:N, j in 1:N; if i != j; tmp_dist[i,j] = Inf; end; end;
    tmp_dist[activeEigenVecs[k],activeEigenVecs[k]] = eigDAG_Distance(𝛷[graphClusters[k], activeEigenVecs[k]], Q[graphClusters[k],:], J)
    push!(partial_dist_ls, tmp_dist)
end
TF_Ψ = TFSC_NGW_frame(partial_dist_ls, 𝛷, M, graphClusters, activeEigenVecs; σ = 0.3, β = 4)[2]
scatter_gplot(X; marker = TF_Ψ[6,30,:], ms = 10)
