## Load image and packages of NGW
include(joinpath("..", "src", "func_includer.jl"))
using MAT; gr(dpi = 400)
barbara = JLD.load(joinpath(@__DIR__, "..", "datasets", "barbara_gray_matrix.jld"), "barbara")

## Build weighted graph
G, L, X = SunFlowerGraph(N = 400); N = nv(G)
lamb, 𝛷 = eigen(Matrix(L)); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)
edge_weight = [e.weight for e in edges(G)]

## Build Dual Graph by DAG metric
distDAG = eigDAG_Distance_normalized(𝛷,Q,N; edge_weight = 1)
W_dual = sparse(dualGraph(distDAG)) #required: sparse dual weighted adjacence matrix

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)
ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = HTree_wavelet_packet_varimax(𝛷,ht_elist_varimax)

## Graph signal approximation
f = matread(joinpath(@__DIR__, "..", "datasets", "sunflower_barbara.mat"))["f_eye"]
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

num_kept_coeffs = 10:10:200; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "f_eye_DAG_k=1.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_feye_nDAG_approx.png")


## Graph signal approximation
f = matread(joinpath(@__DIR__, "..", "datasets", "sunflower_barbara.mat"))["f_face"]
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

num_kept_coeffs = 10:10:200; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "f_face_DAG_k=1.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_fface_nDAG_approx.png")


## Graph signal approximation
f = matread(joinpath(@__DIR__, "..", "datasets", "sunflower_barbara.mat"))["f_trouser"]
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

num_kept_coeffs = 10:10:200; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "f_trouser_DAG_k=1.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_ftrouser_nDAG_approx.png")
