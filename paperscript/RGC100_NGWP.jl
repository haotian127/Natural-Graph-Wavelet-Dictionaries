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

# ## Build Dual Graph
# distROT = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_unweighted_alp1.jld"), "distROT")
# W_dual = sparse(dualGraph(distROT))
#
# ## Assemble wavelet packets
# ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
# wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)
#
# ht_elist_varimax = ht_elist_dual
# wavelet_packet_varimax = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_unweighted_alp1_wavelet_packet_varimax.jld"), "wavelet_packet_varimax")
# # JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_unweighted_alp1_wavelet_packet_varimax.jld"), "wavelet_packet_varimax", wavelet_packet_varimax)
#
# ## Display dual graph and its bi-partitions in 3-dim MDS embedding space
# using MultivariateStats
# gr(dpi=300)
# X_dual = Matrix(transpose(transform(fit(MDS, distROT, maxoutdim = 3, distances = true))))
# scatter_gplot(X_dual; ms = 4, c = :blue); plt = plot!(aspect_ratio = 1,  title = "MDS embedding of the dual graph's nodes")
# savefig(plt, "figs/RGC100_unweighted_ROT1_dual_partition_lvl0")
#
# for l in 1:8
#     lvl_dual_partition = zeros(N)
#     color_scheme = alternating_numbers(length(ht_elist_dual[l]))
#     for i in 1:length(ht_elist_dual[l])
#         lvl_dual_partition .+= color_scheme[i] .* characteristic(ht_elist_dual[l][i], N)
#     end
#     scatter_gplot(X_dual; marker = lvl_dual_partition, ms = 4); plt = plot!(cbar = false, aspect_ratio = 1, title = "dual graph partition lvl = $(l)")
#     savefig(plt, "figs/RGC100_unweighted_ROT1_dual_partition_lvl$(l)")
# end
#
# ## Display graph and its bi-partitions in 2D plane
# gr(dpi=300)
# scatter_gplot(X; ms = 4, c = :blue); plt = plot!(aspect_ratio = 1,  title = "the graph's nodes in 2D projection")
# savefig(plt, "figs/RGC100_unweighted_ROT1_partition_lvl0")
#
# for l in 1:8
#     lvl_partition = zeros(N)
#     color_scheme = alternating_numbers(length(ht_vlist_dual[l]))
#     for i in 1:length(ht_vlist_dual[l])
#         lvl_partition .+= color_scheme[i] .* characteristic(ht_vlist_dual[l][i], N)
#     end
#     scatter_gplot(X; marker = lvl_partition, ms = 4); plt = plot!(cbar = false, aspect_ratio = 1, title = "graph partition lvl = $(l)")
#     savefig(plt, "figs/RGC100_unweighted_ROT1_partition_lvl$(l)")
# end
#
# ## Show some PC NGW basis
# using LaTeXStrings
#
# gr(dpi=400)
# lvl = 7; i = 4; k = 5; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{", lvl-1, "}"))
# ; savefig(plt, "figs/RGC100_unweighted_ROT1_PC_NGW_psi_lvl$(lvl-1)_i$(i-1)_k$(k-1)")
#
#
#
# ## Show some varimax NGW basis
# gr(dpi=400)
# lvl = 9; i = 4; k = 5; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{", lvl-1, "}"))
# ; savefig(plt, "figs/RGC100_unweighted_ROT1_varimax_NGW_psi_lvl$(lvl-1)_i$(i-1)_k$(k-1)")
#


###########################################################################################
### nDAG metric
###########################################################################################
## Build Dual Graph
distDAG = eigDAG_Distance_normalized(𝛷,Q,N)
W_dual = sparse(dualGraph(distDAG))

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_distDAG_normalized_unweighted_wavelet_packet_varimax.jld"), "wavelet_packet_varimax")

# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distDAG_unweighted_wavelet_packet_varimax.jld"), "wavelet_packet_varimax", wavelet_packet_varimax)


# ## Display dual graph and its bi-partitions in 3-dim MDS embedding space
# using MultivariateStats
# gr(dpi=300)
# X_dual = Matrix(transpose(transform(fit(MDS, distDAG[1:100, 1:100], maxoutdim = 3, distances = true))))
# scatter_gplot(X_dual[1:100,:]; ms = 4, c = :blue); plt = plot!(aspect_ratio = 1,  title = "MDS embedding of the dual graph's nodes")
# savefig(plt, "figs/RGC100_unweighted_nDAG_dual_partition_lvl0")

# for l in 1:4
#     lvl_dual_partition = zeros(N)
#     color_scheme = 1:length(ht_elist_dual[l])
#     for i in 1:length(ht_elist_dual[l])
#         lvl_dual_partition .+= color_scheme[i] .* characteristic(ht_elist_dual[l][i], N)
#     end
#     scatter_gplot(X_dual; marker = lvl_dual_partition, ms = 4); plt = plot!(cbar = false, aspect_ratio = 1, title = "dual graph partition lvl = $(l)")
#     savefig(plt, "figs/RGC100_unweighted_DAG_dual_partition_lvl$(l)")
# end

# ## Display graph and its bi-partitions in 2D plane
# gr(dpi=300)
# scatter_gplot(X; ms = 3, c = :blue); plt = plot!(aspect_ratio = 1,  title = "the graph's nodes in 2D projection")
# savefig(plt, "figs/RGC100_unweighted_DAG_partition_lvl0")
#
# for l in 1:4
#     lvl_partition = zeros(N)
#     color_scheme = 1:length(ht_vlist_dual[l])
#     for i in 1:length(ht_vlist_dual[l])
#         lvl_partition .+= color_scheme[i] .* characteristic(ht_vlist_dual[l][i], N)
#     end
#     scatter_gplot(X; marker = lvl_partition, ms = 3); plt = plot!(cbar = false, aspect_ratio = 1, title = "graph partition lvl = $(l)")
#     savefig(plt, "figs/RGC100_unweighted_DAG_partition_lvl$(l)")
# end

## Show some PC NGW basis
using LaTeXStrings

gr(dpi=400)

# Level 3
# Father wavelet
# print("location: ", findmax(wavelet_packet_dual[lvl][i][node,:].^2)[2])
lvl = 4; i = 1; k = 98; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_father_wavelet")

# Mother wavelet
lvl = 4; i = 2; k = 64; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_mother_wavelet")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))

# Packet wavelet
lvl = 4; i = 3; k = 159; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_wavelet_packet_vec")



# Level 4
# Father wavelet
lvl = 5; i = 1; k = 1; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_father_wavelet")
node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node) # node = 698
# scatter_gplot(X; marker = spike(node, N))

# Mother wavelet
lvl = 5; i = 2; k = 20; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_mother_wavelet")
node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node) # node = 784
# scatter_gplot(X; marker = spike(node, N))

# Packet wavelet
lvl = 5; i = 5; k = 7; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_wavelet_packet_vec")
node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node) # node = 977
# scatter_gplot(X; marker = spike(node, N))


# Level 5
# Father wavelet
lvl = 6; i = 1; k = 3; gplot(W, X; width=2); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_father_wavelet")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))

# Mother wavelet
lvl = 6; i = 2; k = 11; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_mother_wavelet")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))

# Packet wavelet
lvl = 6; i = 10; k = 5; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_wavelet_packet_vec")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))





## Show some varimax NGW basis
gr(dpi=400)

# Level 3
# Father wavelet
lvl = 4; i = 1; k = 4; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_father_wavelet")

# Mother wavelet
lvl = 4; i = 2; k = 57; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_mother_wavelet")

# Packet wavelet
lvl = 4; i = 3; k = 29; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_wavelet_packet_vec")


# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# print("location: ", sortperm(wavelet_packet_varimax[lvl][i][node,:].^2, rev = true)[1:5])
# scatter_gplot(X; marker = spike(node, N))


# Level 4
# Father wavelet
lvl = 5; i = 1; k = 4; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_father_wavelet")

# Mother wavelet
lvl = 5; i = 2; k = 50; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_mother_wavelet")

# Packet wavelet
lvl = 5; i = 5; k = 29; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_wavelet_packet_vec")


# node = 977
# print("location: ", sortperm(wavelet_packet_varimax[lvl][i][node,:].^2, rev = true)[1:5])
# scatter_gplot(X; marker = spike(node, N))


# Level 5
# Father wavelet
lvl = 6; i = 1; k = 4; gplot(W, X; width=2); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_father_wavelet")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))

# Mother wavelet
lvl = 6; i = 2; k = 4; gplot(W, X; width=1); scatter_gplot!(X; marker = -wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_mother_wavelet")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))

# Packet wavelet
lvl = 6; i = 10; k = 13; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "figs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_wavelet_packet_vec")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))


# julia> ht_elist_dual[6]
# 60-element Array{Array{Int64,1},1}:
#  [1, 3, 11]
#  [7, 8]
#  [16, 21, 25, 34, 36, 42, 49]
#  [23, 29, 31, 33, 46, 53]
#  [57, 64, 71, 80, 99, 105, 118, 128, 135, 156  …  184, 192, 210, 218, 230, 232, 238, 242, 248, 273]
#  ⋮
#  [358, 691]
#  [733, 871, 1011]
#  [1144]
#  [1153]

# node_lists = [698,784,977]
# for node in node_lists
#     plt = scatter_gplot(X; marker = spike(node, N))
#     savefig(plt, "figs/RGC100_interest_node_$(node)")
# end
