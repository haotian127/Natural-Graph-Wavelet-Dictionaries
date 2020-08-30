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
approx_error_plot2(DVEC; frac = 0.5); approx_error_plt = current()



## 2. noisy thickness signal
f = matread(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_signal.mat"))["f"]  # ground truth
g = matread(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_signal.mat"))["g"]  # noisy signal
DVEC = signal_transform_coeff2(f, g, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)
num_kept_coeffs = 10:10:570; ERR = integrate_approx_results2(DVEC, num_kept_coeffs, "neuron_noisyThicknessF_DAG_k=15.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(legend = :bottomright); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/RGC100_fthick_noisy_nDAG_approx.png")



## 3. log(-z) signal
f = log.(-load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,3])
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

num_kept_coeffs = 10:10:280; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "log(-xyz100)_dim3.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/RGC100_fz_nDAG_approx.png")
