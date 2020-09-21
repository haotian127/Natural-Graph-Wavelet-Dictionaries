## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))
using MAT, MTSG

## Build graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "RGC100.lgz"))
N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,1:2]
L = Matrix(laplacian_matrix(G))
lamb, 𝛷 = eigen(L); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

## Build Dual Graph
distDAG = eigDAG_Distance(𝛷,Q,N)
W_dual = sparse(dualGraph(distDAG))

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_distDAG_unweighted_wavelet_packet_varimax.jld"), "wavelet_packet_varimax")
# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distDAG_normalized_unweighted_wavelet_packet_varimax.jld"), "wavelet_packet_varimax", wavelet_packet_varimax)


## 1. thickness signal
f = matread(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_signal.mat"))["f"]
# gplot(W, X; width=1); scatter_gplot(X; marker = f, smallValFirst = true, ms = 3); signal_plt = plot!(aspect_ratio = 1, grid = false, xlim = [-250, 250], ylim = [-250, 250])
# savefig(signal_plt, "paperfigs/RGC100_fthick.png")

DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

# num_kept_coeffs = 10:10:570; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "neuron_trueThicknessF_DAG_k=15.csv")
# approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(legend = :bottomright, grid = false); approx_error_plt = current()
approx_error_plot2(DVEC); plot!(legend = :bottomright); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/RGC100_fthick_DAG_approx.png")

# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(f, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * f
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    w = Wav_dual[:,importance_idx[i]]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(aspect_ratio = 1, grid = false, xlim = [-250, 250], ylim = [-250, 250])
    savefig(important_NGW_basis_vectors, "paperfigs/RGC100_fthick_DAG_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(f, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * f
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_varimax[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(aspect_ratio = 1, grid = false, xlim = [-250, 250], ylim = [-250, 250])
    savefig(important_NGW_basis_vectors, "paperfigs/RGC100_fthick_DAG_VM_NGW_important_basis_vector$(i).png")
end



# ## 2. noisy thickness signal
# f = matread(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_signal.mat"))["f"]  # ground truth
# g = matread(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_signal.mat"))["g"]  # noisy signal
# DVEC = signal_transform_coeff2(f, g, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)
# num_kept_coeffs = 10:10:570; ERR = integrate_approx_results2(DVEC, num_kept_coeffs, "neuron_noisyThicknessF_DAG_k=15.csv")
# approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(legend = :bottomright, grid = false); approx_error_plt = current()
# savefig(approx_error_plt, "paperfigs/RGC100_fthick_noisy_nDAG_approx.png")


## 3. log(-z) signal
f = log.(-load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,3])
# scatter_gplot(X; marker = f, smallValFirst = true, ms = 3); signal_plt = plot!(aspect_ratio = 1, grid = false, xlim = [-250, 250], ylim = [-250, 250])
# savefig(signal_plt, "paperfigs/RGC100_fz.png")

DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

# num_kept_coeffs = 10:10:570; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "log(-z)_DAG_k=15.csv")
# approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(legend = :bottomleft, grid = false); approx_error_plt = current()
approx_error_plot2(DVEC); plot!(legend = :bottomleft); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/RGC100_fz_DAG_approx.png")


# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(f, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * f
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    w = Wav_dual[:,importance_idx[i]]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(aspect_ratio = 1, grid = false, xlim = [-250, 250], ylim = [-250, 250])
    savefig(important_NGW_basis_vectors, "paperfigs/RGC100_fz_DAG_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(f, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * f
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_varimax[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(aspect_ratio = 1, grid = false, xlim = [-250, 250], ylim = [-250, 250])
    savefig(important_NGW_basis_vectors, "paperfigs/RGC100_fz_DAG_VM_NGW_important_basis_vector$(i).png")
end
