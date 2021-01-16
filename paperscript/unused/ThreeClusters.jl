## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build graph
G, L, X = threeClustersGraph(); N = nv(G)
lamb, 𝛷 = eigen(L); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

## Build partial dual graphs
M = 3
# graphClusters = spectral_clustering(𝛷, M)
graphClusters = JLD.load(joinpath(@__DIR__, "..", "datasets", "ThreeClusters_HAD_TFPsi.jld"), "graphClusters")
# activeEigenVecs = find_active_eigenvectors(𝛷, M, graphClusters)
activeEigenVecs = JLD.load(joinpath(@__DIR__, "..", "datasets", "ThreeClusters_HAD_TFPsi.jld"), "activeEigenVecs")
# partial_dist_ls = partialEig_Distance(graphClusters, activeEigenVecs, lamb, 𝛷, Q, L; eigen_metric = :HAD)
partial_dist_ls = JLD.load(joinpath(@__DIR__, "..", "datasets", "ThreeClusters_HAD_TFPsi.jld"), "partial_dist_ls")

## Build time-frequency adapted NGW frame
# TF_Ψ = TFSC_NGW_frame(partial_dist_ls, 𝛷, M, graphClusters, activeEigenVecs; σ = 0.3, β = 4)
# JLD.save(joinpath(@__DIR__, "..", "datasets", "ThreeClusters_HAD_TFPsi.jld"), "TF_Ψ", TF_Ψ, "graphClusters", graphClusters, "activeEigenVecs", activeEigenVecs, "partial_dist_ls", partial_dist_ls)
TF_Ψ = JLD.load(joinpath(@__DIR__, "..", "datasets", "ThreeClusters_HAD_TFPsi.jld"), "TF_Ψ")

## Generate figures
gr(dpi = 400)

gplot(W, X; width = 1); scatter_gplot!(X; marker = TF_Ψ[1][46,251,:]); ThreeClusters_TFSC_Plt = plot!(framestyle = :none, ylim = [-0.1, 0.3])
# savefig(ThreeClusters_TFSC_Plt, "paperfigs/ThreeClusters_TFSC_bigCluster_wavelet.png")
gplot(W, X; width = 1); scatter_gplot!(X; marker = TF_Ψ[2][9,357,:]); ThreeClusters_TFSC_Plt = plot!(framestyle = :none, ylim = [-0.1, 0.3])
# savefig(ThreeClusters_TFSC_Plt, "paperfigs/ThreeClusters_TFSC_smallClusterRight_wavelet.png")
gplot(W, X; width = 1); scatter_gplot!(X; marker = TF_Ψ[3][9,21,:]); ThreeClusters_TFSC_Plt = plot!(framestyle = :none, ylim = [-0.1, 0.3])
# savefig(ThreeClusters_TFSC_Plt, "paperfigs/ThreeClusters_TFSC_smallClusterLeft_wavelet.png")

gplot(W, X; width = 1); scatter_gplot!(X; marker = pSGWT.sgwt_transform(251, 6, Matrix(W))[:,3]); ThreeClusters_TFSC_Plt = plot!(framestyle = :none, ylim = [-0.1, 0.3])
