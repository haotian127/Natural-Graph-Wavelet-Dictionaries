## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

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

## Input signal
# use the 3rd dimension of xyz as an input signal
# f = log.(-load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,3])
# using MAT; f = matread(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_signal.mat"))["f"]
f = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_thickness_noisy_signal_8db.jld"), "g")

## Best basis selection algorithm
parent_dual = HTree_findParent(ht_elist_dual)
Wav_dual = best_basis_selection(f, wavelet_packet_dual, parent_dual)

parent_varimax = HTree_findParent(ht_elist_varimax)
Wav_varimax = best_basis_selection(f, wavelet_packet_varimax, parent_varimax)

############# varimax NGW coefficients
dvec_varimax = Wav_varimax' * f

############# spectral_prioritized PC NGW coefficients
dvec_spectral = Wav_dual' * f

############# plain Laplacian eigenvectors coefficients
dvec_Laplacian = 𝛷' * f


## MTSG tool box's results
using MTSG

tmp=zeros(length(f),1); tmp[:,1]=f; G_Sig=GraphSig(1.0*W, xy=X, f=tmp)
# GraphSig_Plot(G_Sig, linewidth = 1., markersize = 4., markercolor = :viridis, markerstrokealpha =0., notitle=true)

G_Sig = Adj2InvEuc(G_Sig)
GP = partition_tree_fiedler(G_Sig,:Lrw)
dmatrix = ghwt_analysis!(G_Sig, GP=GP)

############# Haar
BS_haar = bs_haar(GP)
dvec_haar = dmatrix2dvec(dmatrix, GP, BS_haar)

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

approx_error_plot2(DVEC; frac = 0.3); approx_error_plt = current()
# savefig(approx_error_plt, "paperfigs/RGC100_fthick_noisy_distDAG_normalized_reconstruct_errors.png")

ERR = Array{Float64,1}[]
num_kept_coeffs = 10:10:280
for i in 1:length(DVEC)
    dvec = DVEC[i]
    N = length(dvec)
    dvec_norm = norm(dvec,2)
    dvec_sort = sort(dvec.^2) # the smallest first
    er = reverse(cumsum(dvec_sort))/N # this is the relative L^2 error of the whole thing, i.e., its length is N
    push!(ERR, er[num_kept_coeffs])
end
using CSV
frames_approx_res = CSV.File(joinpath(@__DIR__, "..", "datasets", "log(-xyz100)_dim3.csv"))
er_soft_cluster_frame = [frames_approx_res[i][2] for i in 1:length(frames_approx_res)]
push!(ERR, er_soft_cluster_frame)
er_SGWT = [frames_approx_res[i][3] for i in 1:length(frames_approx_res)]
push!(ERR, er_SGWT)

approx_error_plot3(ERR); approx_error_plt = current()
# savefig(approx_error_plt, "paperfigs/RGC100_fz_distDAG_normalized_reconstruct_errors_with_frames.png")
