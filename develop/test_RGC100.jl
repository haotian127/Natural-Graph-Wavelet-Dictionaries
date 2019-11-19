using Plots, LightGraphs, JLD, LaTeXStrings
include(joinpath("..", "src", "func_includer.jl"))

G = loadgraph(joinpath(@__DIR__, "..", "datasets", "RGC100.lgz"))
N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,1:2]
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
sgn = (maximum(V, dims = 1)[:] .> -minimum(V, dims = 1)[:]) .* 2 .- 1
V = (V' .* sgn)'

W = 1.0 .* adjacency_matrix(G)
ht_vlist, ht_elist = HTree_Elist(V,W)
parent = HTree_findParent(ht_vlist)
wavelet_packet = HTree_wavelet_packet(V,ht_vlist,ht_elist)

ind = findall((X[:,1] .> -50) .& (X[:,1] .< 50) .& (X[:,2] .> -30) .& (X[:,2] .< 50))
for lvl in 1:10; plt = scatter_gplot(X[ind,:]; marker = wavelet_packet[lvl][1][ind,1], ms = 8); savefig(plt, "figs\\RGC100_wavelet_layer$(lvl-1)_zoomin.png"); end

# ### mutilated Gaussian signal
# f_mutilatedGaussian = zeros(N)
# constant = 0.2
# ind = 1:N
# f_mutilatedGaussian[ind] .= exp.((-(X[ind,1] .+ 79.4).^2 - (X[ind,2] .- 43.70).^2)/0.02)
# ind = findall(X[:,1] .< -79.4)
# f_mutilatedGaussian[ind] .= f_mutilatedGaussian[ind] .+ constant
# f = f_mutilatedGaussian


# f = zeros(N); f[findall((X[:,1] .< 0) .& (X[:,2] .> 20))] .= 1

f = spike(1,N)



ht_coeff, ht_coeff_L1 = HTree_coeff_wavelet_packet(f,wavelet_packet)
C = HTree_coeff2mat(ht_coeff,N)
dvec = best_basis_algorithm(ht_vlist,parent,ht_coeff_L1)
Wav = assemble_wavelet_basis(dvec,wavelet_packet)


### order wavelet by locations
ord = findmax(abs.(Wav), dims = 1)[2][:]
idx = sortperm([i[1] for i in ord])
heatmap(Wav[:,idx])

# l = 5;wlt = Wav[:,l]; plt = scatter_gplot(X[:,1],X[:,2];marker = wlt, ms = 3 .* exp.(1.5*abs.(wlt)))
# savefig(plt, "figs/wavelet_RGC100_$(l).png")



error_Wavelet = [1.0]
error_Laplacian = [1.0]
for frac = 0.01:0.01:0.3
    numKept = Int(ceil(frac * N))
    ## wavelet varimax reconstruction
    coeff_Wavelet = Wav' * f
    ind = sortperm(coeff_Wavelet.^2, rev = true)
    ind = ind[numKept+1:end]
    # rc_f = Wav[:,ind] * coeff_Wavelet[ind]
    # push!(error_Wavelet, norm(f - rc_f) / norm(f))
    push!(error_Wavelet, norm(coeff_Wavelet[ind])/norm(f))

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
plt = plot(fraction,[error_Wavelet error_Laplacian], yaxis=:log, lab = ["Wavelets","Laplacian"])
# savefig(plt,"figs/signal_approx_RGC100_constUL.png")
