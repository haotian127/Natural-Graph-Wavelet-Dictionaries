using Plots, LightGraphs, JLD, LaTeXStrings
include("../../src/func_includer.jl")
include("Proj.jl")

G = loadgraph("../../datasets/new_toronto_graph.lgz")
N = nv(G)
X = load("../../datasets/new_toronto.jld","xy")
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
sgn = (maximum(V, dims = 1)[:] .> -minimum(V, dims = 1)[:]) .* 2 .- 1
V = (V' .* sgn)'

W = 1.0 .* adjacency_matrix(G)
ht_vlist, ht_elist = HTree_Elist(V,W)
deleteat!(ht_vlist,11:length(ht_vlist)); deleteat!(ht_elist,11:length(ht_elist))



parent = HTree_findParent(ht_vlist)
wavelet_packet = HTree_wavelet_packet(V,ht_vlist,ht_elist)
wavelet_packet_varimax = HTree_wavelet_packet_varimax(V,ht_elist)


# f = load("../../datasets/new_toronto.jld","fv")
f = load("../../datasets/new_toronto.jld","fp")



ht_coeff, ht_coeff_L1 = HTree_coeff_wavelet_packet(f,wavelet_packet)
# C = HTree_coeff2mat(ht_coeff,N)
dvec = best_basis_algorithm(ht_vlist,parent,ht_coeff_L1)
Wav = assemble_wavelet_basis(dvec,wavelet_packet)

ht_coeff_varimax, ht_coeff_L1_varimax = HTree_coeff_wavelet_packet(f,wavelet_packet_varimax)
# C = HTree_coeff2mat(ht_coeff,N)
dvec_varimax = best_basis_algorithm(ht_vlist,parent,ht_coeff_L1_varimax)
Wav_varimax = assemble_wavelet_basis(dvec_varimax,wavelet_packet_varimax)

l = 2060;wlt = Wav[:,l]; plt = scatter_gplot(X[:,1],X[:,2];marker = wlt, ms = 3 .* exp.(1.5*abs.(wlt)))
savefig(plt, "figs/wavelet_toronto_$(l).png")

# ### order wavelet by locations
# ord = findmax(abs.(Wav_varimax), dims = 1)[2][:]
# idx = sortperm([i[1] for i in ord])
# heatmap(Wav_varimax[:,idx])



error_Wavelet = [1.0]
error_Wavelet_varimax = [1.0]
error_Laplacian = [1.0]
for frac = 0.01:0.01:0.3
    numKept = Int(ceil(frac * N))
    ## wavelet reconstruction
    coeff_Wavelet = Wav' * f
    ind = sortperm(coeff_Wavelet.^2, rev = true)
    ind = ind[numKept+1:end]
    # rc_f = Wav[:,ind] * coeff_Wavelet[ind]
    # push!(error_Wavelet, norm(f - rc_f) / norm(f))
    push!(error_Wavelet, norm(coeff_Wavelet[ind])/norm(f))

    ## wavelet varimax reconstruction
    coeff_Wavelet_varimax = Wav_varimax' * f
    ind = sortperm(coeff_Wavelet_varimax.^2, rev = true)
    ind = ind[numKept+1:end]
    # rc_f = Wav[:,ind] * coeff_Wavelet[ind]
    # push!(error_Wavelet, norm(f - rc_f) / norm(f))
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
plt = plot(fraction,[error_Wavelet error_Wavelet_varimax error_Laplacian], yaxis=:log, lab = ["Wavelets", "Wavelets_varimax", "Laplacian"])
savefig(plt,"figs/signal_approx_toronto_fp.png")
