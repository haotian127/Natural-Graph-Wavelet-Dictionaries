using Plots, LightGraphs, JLD, LaTeXStrings
# include("../src/func_includer.jl")
include(joinpath("..", "src", "func_includer.jl"))

N = 256; G = path_graph(N)
X = zeros(N,2); X[:,1] = 1:N
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
V = (V' .* sign.(V[1,:]))'
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

dist_DAG = eigDAG_Distance(V,Q,N)
# W_dual = sparse(dualGraph(dist_DAG)) #sparse dual weighted adjacence matrix
W_dual = 1.0 * adjacency_matrix(path_graph(N))

ht_vlist, ht_elist = HTree_VElist(V,W)
ht_elist_dual, ht_vlist_dual = HTree_EVlist(V,W_dual)
ht_elist_varimax = ht_elist_dual


# ht_elist_varimax = ht_elist_dual[1:4]
# deleteat!(ht_vlist,5:7); deleteat!(ht_elist,5:7)
# deleteat!(ht_vlist_dual,5:7); deleteat!(ht_elist_dual,5:7)


parent_vertex = HTree_findParent(ht_vlist)
wavelet_packet = HTree_wavelet_packet(V,ht_vlist,ht_elist)
wavelet_packet_varimax = HTree_wavelet_packet_varimax(V,ht_elist_varimax)

parent_dual = HTree_findParent(ht_vlist_dual)
wavelet_packet_dual = HTree_wavelet_packet(V,ht_vlist_dual,ht_elist_dual)


wavelet_packet_dual_unortho = HTree_wavelet_packet_unorthogonalized(V,ht_vlist_dual,ht_elist_dual)


# plot(wavelet_packet[5][1][:,10],legend = false)
# f = [exp(-(k-N/3)^2/10)+0.5*exp(-(k-2*N/3)^2/30) for k = 1:N] .+ 0.05*randn(N); f ./= norm(f)
f = V[:,10] + [V[1:25,20]; zeros(N-25)] + [zeros(50);V[51:end,40]]  + V[:,75]
# f = spike(10,N)
# f = rand(N)
# f = [1 .- [(abs(k-24.9))^.5 for k = 1:50] ./ 5; zeros(N-50)] + [zeros(50);V[51:end,40]]
plt = plot(f, legend = false, title = L"f = \phi_{9} + \phi_{19}(1:25) + \phi_{39}(51:end) + \phi_{74}")
# savefig(plt, "figs/path_signal.png")

ht_coeff, ht_coeff_L1 = HTree_coeff_wavelet_packet(f,wavelet_packet)
# C = HTree_coeff2mat(ht_coeff,N)
dvec = best_basis_algorithm2(ht_vlist,parent_vertex,ht_coeff_L1)
Wav = assemble_wavelet_basis(dvec,wavelet_packet)

ht_coeff_varimax, ht_coeff_L1_varimax = HTree_coeff_wavelet_packet(f,wavelet_packet_varimax)
# C_varimax = HTree_coeff2mat(ht_coeff_varimax,N)
parent_varimax = HTree_findParent(ht_elist_varimax)
dvec_varimax = best_basis_algorithm2(ht_elist_varimax,parent_varimax,ht_coeff_L1_varimax)
Wav_varimax = assemble_wavelet_basis(dvec_varimax,wavelet_packet_varimax)

ht_coeff_dual, ht_coeff_L1_dual = HTree_coeff_wavelet_packet(f,wavelet_packet_dual)
# C_varimax = HTree_coeff2mat(ht_coeff_varimax,N)
dvec_dual = best_basis_algorithm2(ht_vlist_dual,parent_dual,ht_coeff_L1_dual)
Wav_dual = assemble_wavelet_basis(dvec_dual,wavelet_packet_dual)

# ### order wavelet by locations
# ord = findmax(abs.(Wav), dims = 1)[2][:]
# idx = sortperm([i[1] for i in ord])
# heatmap(Wav[:,idx])
# plot(Wav[:,idx[end]], legend = false)
#
# ord = findmax(abs.(Wav_varimax), dims = 1)[2][:]
# idx = sortperm([i[1] for i in ord])
# heatmap(Wav_varimax[:,idx])
# plot(Wav_varimax[:,idx[20]], legend = false)
#
ord = findmax(abs.(Wav_dual), dims = 1)[2][:]
idx = sortperm([i[1] for i in ord])
heatmap(Wav_dual[:,idx])
plot(Wav_dual[:,idx[130]], legend = false)



error_Wavelet = [1.0]
error_Wavelet_varimax = [1.0]
error_Wavelet_dual = [1.0]
error_Laplacian = [1.0]
error_Standard = [1.0]
for frac = 0.01:0.01:0.3
    numKept = Int(ceil(frac * N))
    ## wavelet reconstruction
    coeff_Wavelet = Wav' * f
    ind = sortperm(coeff_Wavelet.^2, rev = true)
    ind = ind[numKept+1:end]
    push!(error_Wavelet, norm(coeff_Wavelet[ind])/norm(f))

    ## wavelet varimax reconstruction
    coeff_Wavelet_varimax = Wav_varimax' * f
    ind = sortperm(coeff_Wavelet_varimax.^2, rev = true)
    ind = ind[numKept+1:end]
    push!(error_Wavelet_varimax, norm(coeff_Wavelet_varimax[ind])/norm(f))

    ## wavelet dual reconstruction
    coeff_Wavelet_dual = Wav_dual' * f
    ind = sortperm(coeff_Wavelet_dual.^2, rev = true)
    ind = ind[numKept+1:end]
    push!(error_Wavelet_dual, norm(coeff_Wavelet_dual[ind])/norm(f))

    ## Laplacian reconstruction
    coeff_Laplacian = V' * f
    ind = sortperm(coeff_Laplacian.^2, rev = true)
    ind = ind[numKept+1:end]
    # rc_f = V[:,ind] * coeff_Laplacian[ind]
    # push!(error_Laplacian, norm(f - rc_f) / norm(f))
    push!(error_Laplacian, norm(coeff_Laplacian[ind])/norm(f))

    ## Standard reconstruction
    ind = sortperm(f.^2, rev = true)
    ind = ind[numKept+1:end]
    push!(error_Standard, norm(f[ind])/norm(f))

end

# gr(dpi = 300)
fraction = 0:0.01:0.3
plt = plot(fraction,[error_Wavelet error_Wavelet_varimax error_Wavelet_dual error_Laplacian error_Standard], yaxis=:log, lab = ["WB_vertex" "WB_varimax" "WB_spectral" "Laplacian" "Standard Basis"], linewidth = 2)
# savefig(plt,"figs/path_signal_approx.png")





# heatmap(wavelet_packet_varimax[2][2])
# anim = @animate for i=1:32
#     WW = wavelet_packet_varimax[4][2]
#     # WW = Matrix(qr(WW).Q)
#     sgn = (maximum(WW, dims = 1)[:] .> -minimum(WW, dims = 1)[:]) .* 2 .- 1
#     WW = (WW' .* sgn)'
#     ord = findmax(abs.(WW), dims = 1)[2][:]
#     idx = sortperm([i[1] for i in ord])
#     plot(WW[:,idx[i]], legend = false, ylim = [-0.3,0.7])
# end
# gif(anim, "anim.gif", fps = 5)
