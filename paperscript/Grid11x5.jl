## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
N1, N2 = 11, 5; G = LightGraphs.grid([N1,N2]); N = nv(G)
X = zeros(N1, N2, 2); for i in 1:N1; for j in 1:N2; X[i,j,1] = i; X[i,j,2] = j; end; end; X = reshape(X, (N,2))
L = Matrix(laplacian_matrix(G))
lamb, 𝛷 = eigen(L); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')
Q = incidence_matrix(G; oriented = true)

## Build SGWT frame vectors
W = 1.0 * adjacency_matrix(G)
Ψ_SGWT = pSGWT.sgwt_transform(28, Matrix(W); nf = 6) # centered location vertex = 28, number of filters = 6.

## Generate figure 1
gr(dpi = 400)
for i in 2:5
    gplot(W, X; width = 1); scatter_gplot!(X; marker = Ψ_SGWT[:,i], ms = 14); Grid_SC_plt = plot!(framestyle = :none, xlim = [0.5, 11], ylim = [0.5, 5.5])
    savefig(Grid_SC_plt, "paperfigs/Grid_SGWT_MexicanHat_wavelet_scale$(i).png")
end

## Non-trivial eigenvector metric
distHAD = eigHAD_Distance(𝛷,lamb)

## Build Soft Clustering NGW frame
Ψ = SC_NGW_frame(distHAD, 𝛷; σ = 0.3, β = 4)

## Generate figure 2
gr(dpi = 400) # paper standard 400 dpi
focusEigenVecInd = [6, 9, 10, 19]

for i in focusEigenVecInd
    gplot(W, X; width = 1); scatter_gplot!(X; marker = 𝛷[:,i], ms = 14); Grid_SC_plt = plot!(framestyle = :none, xlim = [0.5, 11], ylim = [0.5, 5.5])
    savefig(Grid_SC_plt, "paperfigs/Grid_EigenVec$(i-1).png")
end

for i in focusEigenVecInd
    gplot(W, X; width = 1); scatter_gplot!(X; marker = Ψ[i,28,:], ms = 14); Grid_SC_plt = plot!(framestyle = :none, xlim = [0.5, 11], ylim = [0.5, 5.5])
    savefig(Grid_SC_plt, "paperfigs/Grid_SC_HAD_wavelet_focusEigenVec$(i-1).png")
end

## Non-trivial eigenvector metric
distDAG = eigDAG_Distance(𝛷, Q, N)

## Build Soft Clustering NGW frame
Ψ = SC_NGW_frame(distDAG, 𝛷; σ = 0.3, β = 4)

## Generate figure 2
gr(dpi = 400) # paper standard 400 dpi
focusEigenVecInd = [6, 9, 10, 19]

for i in focusEigenVecInd
    gplot(W, X; width = 1); scatter_gplot!(X; marker = Ψ[i,28,:], ms = 14); Grid_SC_plt = plot!(framestyle = :none, xlim = [0.5, 11], ylim = [0.5, 5.5])
    savefig(Grid_SC_plt, "paperfigs/Grid_SC_DAG_wavelet_focusEigenVec$(i-1).png")
end
