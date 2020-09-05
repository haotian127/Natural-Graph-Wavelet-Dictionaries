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
distDAG = eigDAG_Distance_normalized(𝛷,Q,N)
W_dual = sparse(dualGraph(distDAG))

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_distDAG_normalized_unweighted_wavelet_packet_varimax.jld"), "wavelet_packet_varimax")
# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distDAG_normalized_unweighted_wavelet_packet_varimax.jld"), "wavelet_packet_varimax", wavelet_packet_varimax)


## 1. thickness signal
f = matread(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_signal.mat"))["f"]
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)
# approx_error_plot2(DVEC; frac = 0.5); plot!(legend = :bottomright); approx_error_plt = current()
num_kept_coeffs = 10:10:570; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "neuron_trueThicknessF_DAG_k=15.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(legend = :bottomright); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/RGC100_fthick_nDAG_approx.png")

# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(f, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * f
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    scatter_gplot(X; marker = Wav_dual[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(clim = (-0.075,0.075), xlim=[-200,250], frame = :none)
    savefig(important_NGW_basis_vectors, "paperfigs/RGC100_fthick_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(f, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * f
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    scatter_gplot(X; marker = Wav_varimax[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(clim = (-0.075,0.075), xlim=[-200,250], frame = :none)
    savefig(important_NGW_basis_vectors, "paperfigs/RGC100_fthick_VM_NGW_important_basis_vector$(i).png")
end



## 2. noisy thickness signal
f = matread(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_signal.mat"))["f"]  # ground truth
g = matread(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_signal.mat"))["g"]  # noisy signal
DVEC = signal_transform_coeff2(f, g, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)
num_kept_coeffs = 10:10:570; ERR = integrate_approx_results2(DVEC, num_kept_coeffs, "neuron_noisyThicknessF_DAG_k=15.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(legend = :bottomright); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/RGC100_fthick_noisy_nDAG_approx.png")

# parent_PC = HTree_findParent(ht_elist_dual)
# Wav_PC = best_basis_selection(f, wavelet_packet_dual, parent_PC)
# parent_VM = HTree_findParent(ht_elist_varimax)
# Wav_VM = best_basis_selection(f, wavelet_packet_varimax, parent_VM)
# dvec_PC = Wav_PC' * [f g]
# dvec_VM = Wav_VM' * [f g]
#
# dvec_f = dvec_VM[:,1]
# dvec_g = dvec_VM[:,2]
# ind = sortperm(dvec_g.^2, rev = true)  # the largest first
# k = 2; g_denoise = Wav_VM * (characteristic(ind[1:k], N) .* dvec_g); plot([f g_denoise])



## 3. log(-z) signal
f = log.(-load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,3])
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

num_kept_coeffs = 10:10:570; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "log(-z)_DAG_k=15.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(legend = :bottomleft); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/RGC100_fz_nDAG_approx.png")


# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(f, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * f
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    scatter_gplot(X; marker = Wav_dual[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(clim = (-0.075,0.075), xlim=[-200,250], frame = :none)
    savefig(important_NGW_basis_vectors, "paperfigs/RGC100_fz_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(f, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * f
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    scatter_gplot(X; marker = Wav_varimax[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(clim = (-0.075,0.075), xlim=[-200,250], frame = :none)
    savefig(important_NGW_basis_vectors, "paperfigs/RGC100_fz_VM_NGW_important_basis_vector$(i).png")
end
