## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "RGC100.lgz"))
N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,1:2]
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
sgn = (maximum(V, dims = 1)[:] .> -minimum(V, dims = 1)[:]) .* 2 .- 1
V = (V' .* sgn)'
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

## Build Dual Graph
dist_DAG = eigDAG_Distance(V,Q,N)
W_dual = sparse(dualGraph(dist_DAG)) #sparse dual weighted adjacence matrix
# W_dual = 1.0 * adjacency_matrix(path_graph(N))

## Assemble wavelet packets
ht_vlist, ht_elist = HTree_VElist(V,W)
ht_elist_dual, ht_vlist_dual = HTree_EVlist(V,W_dual)
ht_elist_varimax = ht_elist_dual

parent_vertex = HTree_findParent(ht_vlist)
wavelet_packet = HTree_wavelet_packet(V,ht_vlist,ht_elist)

parent_varimax = HTree_findParent(ht_elist_varimax)
wavelet_packet_varimax = HTree_wavelet_packet_varimax(V,ht_elist_varimax)

parent_dual = HTree_findParent(ht_vlist_dual)
wavelet_packet_dual = HTree_wavelet_packet(V,ht_vlist_dual,ht_elist_dual)


## Graph signal
# f = [exp(-(k-N/3)^2/10)+0.5*exp(-(k-2*N/3)^2/30) for k = 1:N] .+ 0.1*randn(N); f ./= norm(f)
f = zeros(N); ind = findall((X[:,1] .< 0) .& (X[:,2] .> 20)); f[ind] .= sin.(X[ind,2] .* 0.1); f[1] = 1; ind2 = findall((X[:,1] .> 90) .& (X[:,2] .< 20)); f[ind2] .= sin.(X[ind2,1] .* 0.07)
# plt = scatter_gplot(X; marker = f)
# savefig(plt, "figs\\RGC100_SinSpikeSin.png")


## Best basis selection algorithm
ht_coeff, ht_coeff_L1 = HTree_coeff_wavelet_packet(f,wavelet_packet)
dvec = best_basis_algorithm(ht_coeff_L1, parent_vertex)
Wav = assemble_wavelet_basis(dvec,wavelet_packet)

ht_coeff_varimax, ht_coeff_L1_varimax = HTree_coeff_wavelet_packet(f,wavelet_packet_varimax)
dvec_varimax = best_basis_algorithm(ht_coeff_L1_varimax, parent_varimax)
Wav_varimax = assemble_wavelet_basis(dvec_varimax,wavelet_packet_varimax)

ht_coeff_dual, ht_coeff_L1_dual = HTree_coeff_wavelet_packet(f,wavelet_packet_dual)
dvec_dual = best_basis_algorithm(ht_coeff_L1_dual, parent_dual)
Wav_dual = assemble_wavelet_basis(dvec_dual,wavelet_packet_dual)

## sort wavelets by centered locations
# heatmap(sortWaveletsByCenteredLocations(Wav_dual))

## plot approx. error figure w.r.t. fraction of kept coefficients
approx_error_plot([Wav, Wav_varimax, Wav_dual, V, I], f; label = ["WB_vertex" "WB_varimax" "WB_spectral" "Laplacian" "Standard Basis"])
current()
# plt = plot(fraction,[error_Wavelet error_Wavelet_dual error_Laplacian], yaxis=:log, lab = ["WB_vertex" "WB_spectral" "Laplacian"], linestyle = [:dashdot :solid :dot], linewidth = 3)


## wavelet visualization

# lvl = 3; WB = assemble_wavelet_basis_at_certain_layer(wavelet_packet_dual; layer = lvl); ord = findmax(abs.(WB), dims = 1)[2][:]; idx = sortperm([i[1] for i in ord]); WB = WB[:,idx]
#
# ind = findall((X[:,1] .> -50) .& (X[:,1] .< 50) .& (X[:,2] .> -30) .& (X[:,2] .< 50)); plt = scatter_gplot(X[ind,:]; marker = WB[:,3], ms = 8)
# # savefig(plt, "figs\\RGC100_wavelet_spectral_layer$(lvl-1)_zoomin.png")
