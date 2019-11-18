using Plots, LightGraphs, JLD, LaTeXStrings
# include("../src/func_includer.jl")
include(joinpath("..", "src", "func_includer.jl"))

N = 100; G = PathGraph(N)
X = 1:N
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
V = (V' .* sign.(V[1,:]))'

W = 1.0 .* adjacency_matrix(G)
ht_vlist, ht_elist = HTree_Elist(V,W)
deleteat!(ht_vlist,4:7); deleteat!(ht_elist,4:7)


parent = HTree_findParent(ht_vlist)
wavelet_packet = HTree_wavelet_packet(V,ht_vlist,ht_elist)
wavelet_packet_varimax = HTree_wavelet_packet_varimax(V,ht_elist)

# plot(wavelet_packet[5][1][:,10],legend = false)

f = [exp(-(k-N/3)^2/10)+exp(-(k-2*N/3)^2/30) for k = 1:N] .+ 0.02*randn(N); f ./= norm(f)

# f = V[:,10] + [V[1:25,20]; zeros(75)] + [V[1:50,40]; zeros(50)]  + V[:,75]

# f = randn(N)

ht_coeff, ht_coeff_L1 = HTree_coeff_wavelet_packet(f,wavelet_packet)
# C = HTree_coeff2mat(ht_coeff,N)
dvec = best_basis_algorithm(ht_vlist,parent,ht_coeff_L1)
Wav = assemble_wavelet_basis(dvec,wavelet_packet)

ht_coeff_varimax, ht_coeff_L1_varimax = HTree_coeff_wavelet_packet(f,wavelet_packet_varimax)
# C_varimax = HTree_coeff2mat(ht_coeff_varimax,N)
dvec_varimax = best_basis_algorithm(ht_vlist,parent,ht_coeff_L1_varimax)
Wav_varimax = assemble_wavelet_basis(dvec_varimax,wavelet_packet_varimax)

### order wavelet by locations
ord = findmax(abs.(Wav), dims = 1)[2][:]
idx = sortperm([i[1] for i in ord])
heatmap(Wav[:,idx])


plot(Wav[:,idx[1]])




error_Wavelet = [1.0]
error_Wavelet_varimax = [1.0]
error_Laplacian = [1.0]
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

    ## Laplacian reconstruction
    coeff_Laplacian = V' * f
    ind = sortperm(coeff_Laplacian.^2, rev = true)
    ind = ind[numKept+1:end]
    # rc_f = V[:,ind] * coeff_Laplacian[ind]
    # push!(error_Laplacian, norm(f - rc_f) / norm(f))
    push!(error_Laplacian, norm(coeff_Laplacian[ind])/norm(f))
end

# gr(dpi = 300)
fraction = 0:0.01:0.3
plt = plot(fraction,[error_Wavelet error_Wavelet_varimax error_Laplacian], yaxis=:log, lab = ["Wavelets","Wavelets_varimax","Laplacian"])
# savefig(plt,"figs/signal_approx_path_1.png")
