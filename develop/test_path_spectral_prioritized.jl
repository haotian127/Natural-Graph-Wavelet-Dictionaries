using Plots, LightGraphs, JLD, LaTeXStrings
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
N = 128; G = path_graph(N)
X = zeros(N,2); X[:,1] = 1:N
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
V = (V' .* sign.(V[1,:]))'
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
# f = [exp(-(k-N/3)^2/10)+0.5*exp(-(k-2*N/3)^2/30) for k = 1:N] .+ 0.05*randn(N); f ./= norm(f)
f = V[:,10] + [V[1:25,20]; zeros(N-25)] + [zeros(50);V[51:end,40]]  + V[:,75]
# f = spike(10,N)
# f = rand(N)
# f = [1 .- [(abs(k-24.9))^.5 for k = 1:50] ./ 5; zeros(N-50)] + [zeros(50);V[51:end,40]]
# plt = plot(f, legend = false, title = L"f = \phi_{9} + \phi_{19}(1:25) + \phi_{39}(51:end) + \phi_{74}")
# savefig(plt, "figs/path_signal.png")

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
heatmap(sortWaveletsByCenteredLocations(Wav_dual))

## plot approx. error figure w.r.t. fraction of kept coefficients
approx_error_plot([Wav, Wav_varimax, Wav_dual, V, I], f; label = ["WB_vertex" "WB_varimax" "WB_spectral" "Laplacian" "Standard Basis"])
current()

## generate gif files
# heatmap(wavelet_packet_varimax[2][2])
# anim = @animate for i=1:10
#     WW = wavelet_packet_varimax[3][2]
#     # WW = Matrix(qr(WW).Q)
#     sgn = (maximum(WW, dims = 1)[:] .> -minimum(WW, dims = 1)[:]) .* 2 .- 1
#     WW = (WW' .* sgn)'
#     ord = findmax(abs.(WW), dims = 1)[2][:]
#     idx = sortperm([i[1] for i in ord])
#     plot(WW[:,idx[i]], legend = false, ylim = [-0.3,0.7])
# end
# gif(anim, "gif/anim.gif", fps = 5)
