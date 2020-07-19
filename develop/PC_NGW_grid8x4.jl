## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))


## Build Graph
N1, N2 = 8, 4; G = LightGraphs.grid([N1,N2]); N = nv(G)
X = zeros(N1, N2, 2); for i in 1:N1; for j in 1:N2; X[i,j,1] = i; X[i,j,2] = j; end; end; X = reshape(X, (N,2))
# L = Matrix(laplacian_matrix(G))
# lamb, 𝛷 = eigen(L)
# 𝛷[:,3:4] = varimax(𝛷[:,3:4]); 𝛷[:,9:10] = varimax(𝛷[:,9:10]); 𝛷[:,12:13] = varimax(𝛷[:,12:13]); 𝛷[:,17:18] = varimax(𝛷[:,17:18]); 𝛷[:,21:23] = varimax(𝛷[:,21:23]); 𝛷[:,27:28] = varimax(𝛷[:,27:28]);
# sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')
𝛷 = dct2d_basis(N1, N2)
lamb = sqrt.(sum((L*𝛷).^2, dims = 1))[:]
Q = incidence_matrix(G; oriented = true)
W = 1.0*adjacency_matrix(G)

## Build Dual Graph
distDAG = eigDAG_Distance(𝛷,Q,N)
aHAD = eigHAD_Affinity(𝛷, lamb)
# distROT = eigROT_Distance(𝛷.^2, Q)
W_dual = sparse(aHAD) #required: sparse dual weighted adjacence matrix

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
# wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

## Display dual graph in 2-dim MDS embedding space
using MultivariateStats
gr(dpi=300)
X_dual = Matrix(transpose(transform(fit(MDS, aHAD, maxoutdim = 2, distances = true))))
scatter_gplot(X_dual; ms = 10, c = :blue); plt = plot!(aspect_ratio = 1,  title = "MDS embedding of the dual graph's nodes")
savefig(plt, "figs/Grid8x4_HAD_dual_partition_lvl0")

for l in 1:3
    lvl_dual_partition = zeros(N)
    for i in 1:length(ht_elist_dual[l])
        lvl_dual_partition .+= i .* characteristic(ht_elist_dual[l][i], N)
    end
    scatter_gplot(X_dual; marker = lvl_dual_partition, ms = 10); plt = plot!(cbar = false, aspect_ratio = 1, title = "dual graph partition lvl = $(l)")
    savefig(plt, "figs/Grid8x4_HAD_dual_partition_lvl$(l)")
end

# ## Display graph
# gplot(W, X; width = 1); scatter_gplot!(X; ms = 12, c = :blue); plt = plot!(title = "grid graph")
# savefig(plt, "figs/Grid_partition_lvl0")
#
# for l in 1:3
#     lvl_partition = zeros(N)
#     for i in 1:length(ht_vlist_dual[l])
#         lvl_partition .+= i .* characteristic(ht_vlist_dual[l][i], N)
#     end
#     scatter_gplot(X; marker = lvl_partition, ms = 12); plt = plot!(cbar = false, title = "graph partition lvl = $(l)")
#     savefig(plt, "figs/Grid_partition_lvl$(l)")
# end

# ## Show some PC NGW basis
# using LaTeXStrings
# lvl = 4; i = 2; k = 1; heatmap(transpose(reshape(wavelet_packet_dual[lvl][i][:,k], N1, N2)), c = :viridis, aspect_ratio = 1); plt = plot!(title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{", lvl-1, "}")); savefig(plt, "figs/Grid_PC_NGW_psi_lvl$(lvl-1)_i$(i-1)_k$(k-1)")
# lvl = 4; i = 8; k = 1; heatmap(transpose(reshape(-wavelet_packet_dual[lvl][i][:,k], N1, N2)), c = :viridis, aspect_ratio = 1); plt = plot!(title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{", lvl-1, "}")); savefig(plt, "figs/Grid_PC_NGW_psi_lvl$(lvl-1)_i$(i-1)_k$(k-1)")
# lvl = 3; i = 4; k = 1; heatmap(transpose(reshape(wavelet_packet_dual[lvl][i][:,k], N1, N2)), c = :viridis, aspect_ratio = 1); plt = plot!(title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{", lvl-1, "}")); savefig(plt, "figs/Grid_PC_NGW_psi_lvl$(lvl-1)_i$(i-1)_k$(k-1)")
# lvl = 2; i = 2; k = 7; heatmap(transpose(reshape(wavelet_packet_dual[lvl][i][:,k], N1, N2)), c = :viridis, aspect_ratio = 1); plt = plot!(title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{", lvl-1, "}")); savefig(plt, "figs/Grid_PC_NGW_psi_lvl$(lvl-1)_i$(i-1)_k$(k-1)")
