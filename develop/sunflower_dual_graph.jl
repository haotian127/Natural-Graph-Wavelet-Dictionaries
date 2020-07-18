## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build weighted graph
G, L, X = SunFlowerGraph(); N = nv(G)
lamb, 𝛷 = eigen(Matrix(L)); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)
edge_weight = [e.weight for e in edges(G)]

## Build Dual Graph
distDAG = eigDAG_Distance(𝛷,Q,N; edge_weight = edge_weight)
W_dual = sparse(dualGraph(distDAG)) #required: sparse dual weighted adjacence matrix

## Display dual graph in 2-dim MDS embedding space
using MultivariateStats
gr(dpi=300)
X_dual = Matrix(transpose(transform(fit(MDS, distDAG, maxoutdim = 2, distances = true))))
scatter_gplot(X_dual; ms = 3, c = :blue); plt = plot!(aspect_ratio = 1, title = "MDS embedding of the dual graph's nodes")
savefig(plt, "figs/sunflower_DAG_dual_partition_lvl0")


## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

for l in 1:3
    lvl_dual_partition = zeros(N)
    for i in 1:length(ht_elist_dual[l])
        lvl_dual_partition .+= i .* characteristic(ht_elist_dual[l][i], N)
    end
    scatter_gplot(X_dual; marker = lvl_dual_partition, ms = 3); plt = plot!(cbar = false, aspect_ratio = 1, title = "dual graph partition lvl = $(l)")
    savefig(plt, "figs/Sunflower_DAG_dual_partition_lvl$(l)")
end


## Display some GL eigenvectors
idx = ht_elist_dual[1][1][end]; scatter_gplot(X; marker = 𝛷[:,idx], ms = LinRange(3.0, 7.0, N), smallValFirst = false); plt = plot!(framestyle = :none, title = latexstring("\\phi_{", idx-1, "}")); savefig(plt, "figs/Sunflower_GL_eigenvector_$(idx-1).png")
idx = ht_elist_dual[1][2][1]; scatter_gplot(X; marker = 𝛷[:,idx], ms = LinRange(3.0, 7.0, N), smallValFirst = false); plt = plot!(framestyle = :none, title = latexstring("\\phi_{", idx-1, "}")); savefig(plt, "figs/Sunflower_GL_eigenvector_$(idx-1).png")
idx = ht_elist_dual[2][1][end]; scatter_gplot(X; marker = 𝛷[:,idx], ms = LinRange(3.0, 7.0, N), smallValFirst = false); plt = plot!(framestyle = :none, title = latexstring("\\phi_{", idx-1, "}")); savefig(plt, "figs/Sunflower_GL_eigenvector_$(idx-1).png")
idx = ht_elist_dual[2][2][1]; scatter_gplot(X; marker = 𝛷[:,idx], ms = LinRange(3.0, 7.0, N), smallValFirst = false); plt = plot!(framestyle = :none, title = latexstring("\\phi_{", idx-1, "}")); savefig(plt, "figs/Sunflower_GL_eigenvector_$(idx-1).png")
idx = ht_elist_dual[2][3][end]; scatter_gplot(X; marker = 𝛷[:,idx], ms = LinRange(3.0, 7.0, N), smallValFirst = false); plt = plot!(framestyle = :none, title = latexstring("\\phi_{", idx-1, "}")); savefig(plt, "figs/Sunflower_GL_eigenvector_$(idx-1).png")
idx = ht_elist_dual[2][4][1]; scatter_gplot(X; marker = 𝛷[:,idx], ms = LinRange(3.0, 7.0, N), smallValFirst = false); plt = plot!(framestyle = :none, title = latexstring("\\phi_{", idx-1, "}")); savefig(plt, "figs/Sunflower_GL_eigenvector_$(idx-1).png")

idx = 23; scatter_gplot(X; marker = 𝛷[:,idx], ms = LinRange(3.0, 7.0, N), smallValFirst = false); plt = plot!(framestyle = :none, title = latexstring("\\phi_{", idx-1, "}")); savefig(plt, "figs/Sunflower_GL_eigenvector_$(idx-1).png")

idx = 110; scatter_gplot(X; marker = 𝛷[:,idx], ms = LinRange(3.0, 7.0, N), smallValFirst = false); plt = plot!(framestyle = :none, title = latexstring("\\phi_{", idx-1, "}")); savefig(plt, "figs/Sunflower_GL_eigenvector_$(idx-1).png")

idx = 228; scatter_gplot(X; marker = 𝛷[:,idx], ms = LinRange(3.0, 7.0, N), smallValFirst = false); plt = plot!(framestyle = :none, title = latexstring("\\phi_{", idx-1, "}")); savefig(plt, "figs/Sunflower_GL_eigenvector_$(idx-1).png")

idx = 343; scatter_gplot(X; marker = 𝛷[:,idx], ms = LinRange(3.0, 7.0, N), smallValFirst = false); plt = plot!(framestyle = :none, title = latexstring("\\phi_{", idx-1, "}")); savefig(plt, "figs/Sunflower_GL_eigenvector_$(idx-1).png")
