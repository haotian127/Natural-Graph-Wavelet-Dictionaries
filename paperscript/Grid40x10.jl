## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
N1, N2 = 40, 10; G = LightGraphs.grid([N1,N2]); N = nv(G)
X = zeros(N1, N2, 2); for i in 1:N1; for j in 1:N2; X[i,j,1] = i; X[i,j,2] = j; end; end; X = reshape(X, (N,2))
L = Matrix(laplacian_matrix(G))
𝛷 = dct2d_basis(N1, N2)
lamb = sqrt.(sum((L*𝛷).^2, dims = 1))[:]
Q = incidence_matrix(G; oriented = true)
W = 1.0*adjacency_matrix(G)

## Build SGWT frame vectors
Ψ_SGWT = pSGWT.sgwt_transform(100, Matrix(W); nf = 6) # centered location vertex = 28, number of filters = 6.

## Generate SGWT Mexican Hat wavelets
gr(dpi = 400)
for i in 2:5
    heatmap(transpose(reshape(Ψ_SGWT[:,i], N1, N2)), c = :viridis, aspect_ratio = 1); Grid_SC_plt = plot!(framestyle = :none)
    savefig(Grid_SC_plt, "paperfigs/Grid40x10_SGWT_MexicanHat_wavelet_scale$(i).png")
end

## Non-trivial eigenvector metric
distHAD = eigHAD_Distance(𝛷,lamb)

## Build Soft Clustering NGW frame
Ψ_HAD = SC_NGW_frame(distHAD, 𝛷; σ = 0.3, β = 4)

## Generate figure
gr(dpi = 400) # paper standard 400 dpi
focusEigenVecInd = [4,8,31,61,41,5]

for i in focusEigenVecInd
    heatmap(transpose(reshape(𝛷[:,i], N1, N2)), c = :viridis, aspect_ratio = 1); Grid_SC_plt = plot!(framestyle = :none)
    savefig(Grid_SC_plt, "paperfigs/Grid40x10_EigenVec$(i-1).png")
end

for i in focusEigenVecInd
    heatmap(transpose(reshape(Ψ_HAD[i,100,:], N1, N2)), c = :viridis, aspect_ratio = 1); Grid_SC_plt = plot!(framestyle = :none)
    savefig(Grid_SC_plt, "paperfigs/Grid40x10_SC_HAD_wavelet_focusEigenVec$(i-1).png")
end

## Non-trivial eigenvector metric
distDAG = eigDAG_Distance(𝛷, Q, N)

## Build Soft Clustering NGW frame
Ψ_DAG = SC_NGW_frame(distDAG, 𝛷; σ = 0.3, β = 4)

## Generate figure
gr(dpi = 400) # paper standard 400 dpi
focusEigenVecInd = [4,8,31,61,41,5]

for i in focusEigenVecInd
    heatmap(transpose(reshape(Ψ_DAG[i,100,:], N1, N2)), c = :viridis, aspect_ratio = 1); Grid_SC_plt = plot!(framestyle = :none)
    savefig(Grid_SC_plt, "paperfigs/Grid40x10_SC_DAG_wavelet_focusEigenVec$(i-1).png")
end
