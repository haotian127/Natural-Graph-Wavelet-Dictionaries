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
g = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_noisy_signal_8db.jld"), "g")
DVEC = signal_transform_coeff(g, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)
approx_error_plot2(DVEC; frac = 0.5); approx_error_plt = current()


parent_dual = HTree_findParent(ht_elist_dual); W_PC = best_basis_selection(f, wavelet_packet_dual, parent_dual)
parent_varimax = HTree_findParent(ht_elist_varimax); W_VM = best_basis_selection(f, wavelet_packet_varimax, parent_varimax)
############# spectral_prioritized PC NGW coefficients
dvec_spectral = W_PC' * f
############# varimax NGW coefficients
dvec_varimax = W_VM' * f
############# plain Laplacian eigenvectors coefficients
dvec_Laplacian = 𝛷' * f

tmp=zeros(length(f),1); tmp[:,1]=f; G_Sig=GraphSig(1.0*W, xy=X, f=tmp); G_Sig = Adj2InvEuc(G_Sig); GP = partition_tree_fiedler(G_Sig,:Lrw); dmatrix = ghwt_analysis!(G_Sig, GP=GP)
############# Haar
BS_haar = bs_haar(GP)
dvec_haar = dmatrix2dvec(dmatrix, GP, BS_haar)
# f_reconstruct_haar = ghwt_synthesis(dvec_haar, GP, BS_haar, G_Sig)[1][:]
############# Walsh
BS_walsh = bs_walsh(GP)
dvec_walsh = dmatrix2dvec(dmatrix, GP, BS_walsh)
############# GHWT_c2f
dvec_c2f, BS_c2f = ghwt_c2f_bestbasis(dmatrix, GP)
############# GHWT_f2c
dvec_f2c, BS_f2c = ghwt_f2c_bestbasis(dmatrix, GP)
############# eGHWT
dvec_eghwt, BS_eghwt = ghwt_tf_bestbasis(dmatrix, GP)
DVEC = [dvec_haar[:], dvec_walsh[:], dvec_Laplacian[:], dvec_c2f[:], dvec_f2c[:], dvec_eghwt[:], dvec_spectral[:], dvec_varimax[:]]

num_kept_coeffs = 10:10:570; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "neuron_noisyThicknessF_DAG_k=15.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/RGC100_fthick_noisy_nDAG_approx.png")






## 3. log(-z) signal
f = log.(-load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,3])
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

num_kept_coeffs = 10:10:280; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "log(-xyz100)_dim3.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/RGC100_fz_nDAG_approx.png")
