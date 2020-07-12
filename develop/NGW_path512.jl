## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
N = 512; G = path_graph(N)
X = zeros(N,2); X[:,1] = 1:N
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
V = (V' .* sign.(V[1,:]))'
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

## Build Dual Graph
dist_DAG = eigDAG_Distance(V,Q,N)
# W_dual = sparse(dualGraph(dist_DAG)) #sparse dual weighted adjacence matrix
W_dual = 1.0 * adjacency_matrix(path_graph(N)) #ground truth

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(V,W_dual)
ht_elist_varimax = ht_elist_dual

parent_dual = HTree_findParent(ht_vlist_dual)
wavelet_packet_dual = HTree_wavelet_packet(V,ht_vlist_dual,ht_elist_dual)

parent_varimax = HTree_findParent(ht_elist_varimax)
wavelet_packet_varimax = HTree_wavelet_packet_varimax(V,ht_elist_varimax)

## generate gif files

# Pair Clustering NGW
# anim = @animate for i=1:128
#     WW = wavelet_packet_dual[3][1]
#     sgn = (maximum(WW, dims = 1)[:] .> -minimum(WW, dims = 1)[:]) .* 2 .- 1
#     WW = (WW' .* sgn)'
#     ord = findmax(abs.(WW), dims = 1)[2][:]
#     idx = sortperm([j[1] for j in ord])
#     plot(WW[:,idx[i]], legend = false, ylim = [-0.3,0.7])
# end
# gif(anim, "gif/anim_Path512_PC_NGW.gif", fps = 30)
#
# # Varimax NGW
# anim = @animate for i=1:128
#     WW = wavelet_packet_varimax[3][1]
#     sgn = (maximum(WW, dims = 1)[:] .> -minimum(WW, dims = 1)[:]) .* 2 .- 1
#     WW = (WW' .* sgn)'
#     ord = findmax(abs.(WW), dims = 1)[2][:]
#     idx = sortperm([j[1] for j in ord])
#     plot(WW[:,idx[i]], legend = false, ylim = [-0.3,0.7])
# end
# gif(anim, "gif/anim_Path512_Varimax_NGW.gif", fps = 30)

# Mother Wavelet
anim = @animate for i=1:128
    WW = wavelet_packet_dual[3][2]
    sgn = (maximum(WW, dims = 1)[:] .> -minimum(WW, dims = 1)[:]) .* 2 .- 1
    WW = (WW' .* sgn)'
    ord = findmax(abs.(WW), dims = 1)[2][:]
    idx = sortperm([j[1] for j in ord])
    plot(WW[:,idx[i]], legend = false, ylim = [-0.3,0.7])
end
gif(anim, "gif/anim_Path512_PC_NGW_mother.gif", fps = 30)

# Varimax NGW
anim = @animate for i=1:128
    WW = wavelet_packet_varimax[3][2]
    sgn = (maximum(WW, dims = 1)[:] .> -minimum(WW, dims = 1)[:]) .* 2 .- 1
    WW = (WW' .* sgn)'
    ord = findmax(abs.(WW), dims = 1)[2][:]
    idx = sortperm([j[1] for j in ord])
    plot(WW[:,idx[i]], legend = false, ylim = [-0.3,0.7])
end
gif(anim, "gif/anim_Path512_Varimax_NGW_mother.gif", fps = 30)
