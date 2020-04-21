## load packages
using Plots, LightGraphs, JLD, LaTeXStrings, Distances
include(joinpath("..", "src", "func_includer.jl"))

## load graph info
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "new_toronto_graph.lgz"))
N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"xy")
W = 1.0 .* adjacency_matrix(G)
dist_X = pairwise(Euclidean(),X; dims = 1)
Weight = W .* dualGraph(dist_X; method = "inverse") # weighted adjacence matrix
L = Matrix(Diagonal(sum(Weight;dims = 1)[:]) - Weight)
# L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
sgn = (maximum(V, dims = 1)[:] .> -minimum(V, dims = 1)[:]) .* 2 .- 1
V = (V' .* sgn)'
Q = incidence_matrix(G; oriented = true)
edge_weights = 1 ./ sqrt.(sum((Q' * X).^2, dims = 2)[:])

## build dual graph by non-trivial Laplacian eigenvectors metric
dist_DAG = eigDAG_Distance(V,Q,N; edge_weight = edge_weights)
W_dual = sparse(dualGraph(dist_DAG))

## build two versions of wavelet packets: vertex_prioritized and spectral_prioritized (dual)
ht_vlist, ht_elist = HTree_VElist(V,W)
ht_elist_dual, ht_vlist_dual = HTree_EVlist(V,W_dual)

parent_vertex = HTree_findParent(ht_vlist)
wavelet_packet = HTree_wavelet_packet(V,ht_vlist,ht_elist)
# wavelet_packet_varimax = HTree_wavelet_packet_varimax(V,ht_elist)
# wavelet_packet_varimax = HTree_wavelet_packet_varimax(V,ht_elist_dual)

parent_dual = HTree_findParent(ht_vlist_dual)
wavelet_packet_dual = HTree_wavelet_packet(V,ht_vlist_dual,ht_elist_dual)

## load graph signals
# f = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fv")
# f = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fp")
f6 = load(joinpath(@__DIR__, "..", "datasets", "toronto_systhetic_signal.jld"), "f6")
f12 = load(joinpath(@__DIR__, "..", "datasets", "toronto_systhetic_signal.jld"), "f12")
f18 = load(joinpath(@__DIR__, "..", "datasets", "toronto_systhetic_signal.jld"), "f18")
f24 = load(joinpath(@__DIR__, "..", "datasets", "toronto_systhetic_signal.jld"), "f24")
f30 = load(joinpath(@__DIR__, "..", "datasets", "toronto_systhetic_signal.jld"), "f30")

f = f6
# plt = scatter_gplot(X[sortperm(f30),:]; marker = sort(f30))
# savefig(plt, "figs\\toronto_f30.png")


## build wavelet basis by best basis selection algorithm
ht_coeff, ht_coeff_L1 = HTree_coeff_wavelet_packet(f,wavelet_packet)
# C = HTree_coeff2mat(ht_coeff,N)
dvec = best_basis_algorithm2(ht_vlist,parent_vertex,ht_coeff_L1)
Wav = assemble_wavelet_basis(dvec,wavelet_packet)

# ht_coeff_varimax, ht_coeff_L1_varimax = HTree_coeff_wavelet_packet(f,wavelet_packet_varimax)
# # C_varimax = HTree_coeff2mat(ht_coeff_varimax,N)
# dvec_varimax = best_basis_algorithm(ht_vlist_dual,parent_dual,ht_coeff_L1_varimax)
# Wav_varimax = assemble_wavelet_basis(dvec_varimax,wavelet_packet_varimax)

ht_coeff_dual, ht_coeff_L1_dual = HTree_coeff_wavelet_packet(f,wavelet_packet_dual)
# C_dual = HTree_coeff2mat(ht_coeff_dual,N)
dvec_dual = best_basis_algorithm2(ht_vlist_dual,parent_dual,ht_coeff_L1_dual)
Wav_dual = assemble_wavelet_basis(dvec_dual,wavelet_packet_dual)

## order wavelet by locations
# ord = findmax(abs.(Wav), dims = 1)[2][:]
# idx = sortperm([i[1] for i in ord])
# heatmap(Wav[:,idx])
# plot(Wav[:,idx[end]], legend = false)

# ord = findmax(abs.(Wav_varimax), dims = 1)[2][:]
# idx = sortperm([i[1] for i in ord])
# heatmap(Wav_varimax[:,idx])
# plot(Wav_varimax[:,idx[20]], legend = false)

# ord = findmax(abs.(Wav_dual), dims = 1)[2][:]
# idx = sortperm([i[1] for i in ord])
# heatmap(Wav_dual[:,idx])
# plot(Wav_dual[:,idx[30]], legend = false)




## draw approx. error figure w.r.t. fraction of kept coefficients

error_Wavelet = [1.0]
# error_Wavelet_varimax = [1.0]
error_Wavelet_dual = [1.0]
error_Laplacian = [1.0]

coeff_Wavelet = Wav' * f
coeff_Wavelet_dual = Wav_dual' * f
coeff_Laplacian = V' * f

for frac = 0.01:0.01:0.3
    numKept = Int(ceil(frac * N))
    ## wavelet reconstruction
    ind = sortperm(coeff_Wavelet.^2, rev = true)
    ind = ind[numKept+1:end]
    push!(error_Wavelet, norm(coeff_Wavelet[ind])/norm(f))

    # ## wavelet varimax reconstruction
    # coeff_Wavelet_varimax = Wav_varimax' * f
    # ind = sortperm(coeff_Wavelet_varimax.^2, rev = true)
    # ind = ind[numKept+1:end]
    # push!(error_Wavelet_varimax, norm(coeff_Wavelet_varimax[ind])/norm(f))

    ## wavelet dual reconstruction
    ind = sortperm(coeff_Wavelet_dual.^2, rev = true)
    ind = ind[numKept+1:end]
    push!(error_Wavelet_dual, norm(coeff_Wavelet_dual[ind])/norm(f))

    ## Laplacian reconstruction
    ind = sortperm(coeff_Laplacian.^2, rev = true)
    ind = ind[numKept+1:end]
    # rc_f = V[:,ind] * coeff_Laplacian[ind]
    # push!(error_Laplacian, norm(f - rc_f) / norm(f))
    push!(error_Laplacian, norm(coeff_Laplacian[ind])/norm(f))
end

gr(dpi = 300)
fraction = 0:0.01:0.3
plt = plot(fraction,[error_Wavelet error_Wavelet_dual error_Laplacian], yaxis=:log, lab = ["WB_vertex" "WB_spectral" "Laplacian"], linestyle = [:dashdot :solid :dot], linewidth = 3)
# savefig(plt,"figs/signal_approx_toronto_f6_signless_Laplacian_affinity_MGS.png")



######################################
### wavelet visualization
######################################
# lvl = 9; WB = assemble_wavelet_basis_at_certain_layer(wavelet_packet_dual; layer = lvl); ord = findmax(abs.(WB), dims = 1)[2][:]; idx = sortperm([i[1] for i in ord]); WB = WB[:,idx]

# ind = findall((X[:,1] .> -79.4) .& (X[:,1] .< -79.35) .& (X[:,2] .> 43.62) .& (X[:,2] .< 43.66));
# scatter_gplot(X; marker = WB[:,1]); plt = plot!(framestyle = :none)
# savefig(plt, "figs\\RGC100_wavelet_spectral_layer$(lvl-1)_zoomin.png")
