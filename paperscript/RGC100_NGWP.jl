## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "RGC100.lgz"))
N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,1:2]
L = Matrix(laplacian_matrix(G))
lamb, 𝛷 = eigen(L); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

## Build Dual Graph
distROT = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_unweighted_alp1.jld"), "distROT")
W_dual = sparse(dualGraph(distROT))

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = HTree_wavelet_packet_varimax(𝛷,ht_elist_varimax)

# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_unweighted_alp1_wavelet_packet_varimax.jld"), "wavelet_packet_varimax", wavelet_packet_varimax)
## Display dual graph and its bi-partitions in 3-dim MDS embedding space
using MultivariateStats
gr(dpi=300)
X_dual = Matrix(transpose(transform(fit(MDS, distROT, maxoutdim = 3, distances = true))))
scatter_gplot(X_dual; ms = 4, c = :blue); plt = plot!(aspect_ratio = 1,  title = "MDS embedding of the dual graph's nodes")
savefig(plt, "figs/RGC100_unweighted_ROT1_dual_partition_lvl0")

for l in 1:8
    lvl_dual_partition = zeros(N)
    color_scheme = alternating_numbers(length(ht_elist_dual[l]))
    for i in 1:length(ht_elist_dual[l])
        lvl_dual_partition .+= color_scheme[i] .* characteristic(ht_elist_dual[l][i], N)
    end
    scatter_gplot(X_dual; marker = lvl_dual_partition, ms = 4); plt = plot!(cbar = false, aspect_ratio = 1, title = "dual graph partition lvl = $(l)")
    savefig(plt, "figs/RGC100_unweighted_ROT1_dual_partition_lvl$(l)")
end

## Display graph and its bi-partitions in 2D plane
gr(dpi=300)
scatter_gplot(X; ms = 4, c = :blue); plt = plot!(aspect_ratio = 1,  title = "the graph's nodes in 2D projection")
savefig(plt, "figs/RGC100_unweighted_ROT1_partition_lvl0")

for l in 1:8
    lvl_partition = zeros(N)
    color_scheme = alternating_numbers(length(ht_vlist_dual[l]))
    for i in 1:length(ht_vlist_dual[l])
        lvl_partition .+= color_scheme[i] .* characteristic(ht_vlist_dual[l][i], N)
    end
    scatter_gplot(X; marker = lvl_partition, ms = 4); plt = plot!(cbar = false, aspect_ratio = 1, title = "graph partition lvl = $(l)")
    savefig(plt, "figs/RGC100_unweighted_ROT1_partition_lvl$(l)")
end

## Show some PC NGW basis
using LaTeXStrings

gr(dpi=400)
lvl = 7; i = 4; k = 5; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{", lvl-1, "}"))
; savefig(plt, "figs/RGC100_unweighted_ROT1_PC_NGW_psi_lvl$(lvl-1)_i$(i-1)_k$(k-1)")



## Show some varimax NGW basis
gr(dpi=400)
lvl = 9; i = 4; k = 5; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{", lvl-1, "}"))
; savefig(plt, "figs/RGC100_unweighted_ROT1_varimax_NGW_psi_lvl$(lvl-1)_i$(i-1)_k$(k-1)")



###########################################################################################
### DAG metric
###########################################################################################
## Build Dual Graph
distDAG = eigDAG_Distance(𝛷,Q,N)
W_dual = sparse(dualGraph(distDAG))

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = HTree_wavelet_packet_varimax(𝛷,ht_elist_varimax)

# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distDAG_unweighted_wavelet_packet_varimax.jld"), "wavelet_packet_varimax", wavelet_packet_varimax)
## Display dual graph and its bi-partitions in 3-dim MDS embedding space
using MultivariateStats
gr(dpi=300)
X_dual = Matrix(transpose(transform(fit(MDS, distDAG, maxoutdim = 3, distances = true))))
scatter_gplot(X_dual; ms = 4, c = :blue); plt = plot!(aspect_ratio = 1,  title = "MDS embedding of the dual graph's nodes")
savefig(plt, "figs/RGC100_unweighted_DAG_dual_partition_lvl0")

for l in 1:4
    lvl_dual_partition = zeros(N)
    color_scheme = 1:length(ht_elist_dual[l])
    for i in 1:length(ht_elist_dual[l])
        lvl_dual_partition .+= color_scheme[i] .* characteristic(ht_elist_dual[l][i], N)
    end
    scatter_gplot(X_dual; marker = lvl_dual_partition, ms = 4); plt = plot!(cbar = false, aspect_ratio = 1, title = "dual graph partition lvl = $(l)")
    savefig(plt, "figs/RGC100_unweighted_DAG_dual_partition_lvl$(l)")
end

## Display graph and its bi-partitions in 2D plane
gr(dpi=300)
scatter_gplot(X; ms = 3, c = :blue); plt = plot!(aspect_ratio = 1,  title = "the graph's nodes in 2D projection")
savefig(plt, "figs/RGC100_unweighted_DAG_partition_lvl0")

for l in 1:4
    lvl_partition = zeros(N)
    color_scheme = 1:length(ht_vlist_dual[l])
    for i in 1:length(ht_vlist_dual[l])
        lvl_partition .+= color_scheme[i] .* characteristic(ht_vlist_dual[l][i], N)
    end
    scatter_gplot(X; marker = lvl_partition, ms = 3); plt = plot!(cbar = false, aspect_ratio = 1, title = "graph partition lvl = $(l)")
    savefig(plt, "figs/RGC100_unweighted_DAG_partition_lvl$(l)")
end

## Show some PC NGW basis
using LaTeXStrings

gr(dpi=400)
lvl = 5; i = 9; k = 8; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{", lvl-1, "}"))
; savefig(plt, "figs/RGC100_unweighted_DAG_PC_NGW_psi_lvl$(lvl-1)_i$(i-1)_k$(k-1)")



## Show some varimax NGW basis
gr(dpi=400)
lvl = 5; i = 8; k = 10; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{", lvl-1, "}"))
; savefig(plt, "figs/RGC100_unweighted_DAG_varimax_NGW_psi_lvl$(lvl-1)_i$(i-1)_k$(k-1)")





###########################################################################################
### HAD metric
###########################################################################################
## Build Dual Graph
aHAD = eigHAD_Affinity(𝛷, lamb)
W_dual = sparse(aHAD)
JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_aHAD_unweighted.jld"), "aHAD", aHAD)
