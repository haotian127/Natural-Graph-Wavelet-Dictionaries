## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build weighted graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "new_toronto_graph.lgz")); N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"xy")
W = 1.0 .* adjacency_matrix(G)
dist_X = pairwise(Euclidean(),X; dims = 1)
Weight = W .* dualGraph(dist_X; method = "inverse") # weighted adjacence matrix
L = Matrix(Diagonal(sum(Weight;dims = 1)[:]) - Weight)
lamb, 𝛷 = eigen(L); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')
Q = incidence_matrix(G; oriented = true)
edge_weight = 1 ./ sqrt.(sum((Q' * X).^2, dims = 2)[:])

## Build Dual Graph
distDAG = eigDAG_Distance_normalized(𝛷,Q,N; edge_weight = 1)
W_dual = sparse(dualGraph(distDAG)) #required: sparse dual weighted adjacence matrix

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = JLD.load(joinpath(@__DIR__, "..", "datasets", "Toronto_nDAG_NGWP.jld"), "wavelet_packet_dual")

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = JLD.load(joinpath(@__DIR__, "..", "datasets", "Toronto_nDAG_NGWP.jld"), "wavelet_packet_varimax")

## Graph signal
# pyplot(dpi = 400); gplot(1.0*adjacency_matrix(G), X, width = 1, color = :blue); plot!(aspect_ratio=1, framestyle = :none); Toronto_signal = scatter_gplot!(X; marker = f, smallValFirst = true)
# savefig(Toronto_signal, "paperfigs/Toronto_fneighbor.png")

## 1. f_density
f = zeros(N); for i in 1:N; f[i] = length(findall(dist_X[:,i] .< 1/minimum(edge_weight))); end #fneighbor
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

num_kept_coeffs = 10:10:1130; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "toronto_density_DAG_k=7.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/Toronto_fdensity_nDAG_approx.png")

## 2. f_pedestrain
fp = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fp")
DVEC = signal_transform_coeff(fp, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)
approx_error_plot2(DVEC; frac = 0.5); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/Toronto_fp_nDAG_approx_no_frames.png")

# num_kept_coeffs = 10:10:1130; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "toronto_density_DAG_k=7.csv")
# approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
# savefig(approx_error_plt, "paperfigs/Toronto_fp_nDAG_approx.png")

## 3. f_vehicle
fv = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fv")
DVEC = signal_transform_coeff(fv, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)
approx_error_plot2(DVEC; frac = 0.5); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/Toronto_fv_nDAG_approx_no_frames.png")
