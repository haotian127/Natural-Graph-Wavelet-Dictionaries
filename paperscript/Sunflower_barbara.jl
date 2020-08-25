## Load image and packages of NGW
include(joinpath("..", "src", "func_includer.jl"))
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

## Start barbara experiment and generating different signals focused at different locations
gr(dpi = 400)
### focus on the eye, i.e., f_eye_NN, f_eye_Bilinear
N = 400; c=1.0/N; θ=(sqrt(5.0)-1)*π;
X_sf = zeros(N,2); for k=1:N X_sf[k,:]=c*(k-1)*[cos((k-1)*θ) sin((k-1)*θ)]; end; X_transform = transform2D(X_sf; s = 20, t = [395, 100]) # t = [395, 100] is the location of the eye.
f_eye_Bilinear = Bilinear_rendering(X_transform, barbara)

## focus on the face, i.e., f_face_NN, f_face_Bilinear
X_transform = transform2D(X_sf; s = 20, t = [398, 137]) # t = [398, 137] is the location of the face.
f_face_Bilinear = Bilinear_rendering(X_transform, barbara)

## focus on the trouser, i.e., f_trouser_NN, f_trouser_Bilinear
X_transform = transform2D(X_sf; s = 20, t = [280, 320]) # t = [280, 320] is the location of the trouser.
f_trouser_Bilinear = Bilinear_rendering(X_transform, barbara)

## Graph signal approximation
f = f_eye_Bilinear
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

num_kept_coeffs = 10:10:200; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "f_eye_DAG_k=1.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_feye_nDAG_approx.png")
# approx_error_plot2(DVEC); plt = current(); savefig(plt, "paperfigs/SunFlower_barbara_feye_nDAG_no_frames.png")


## Graph signal approximation
f = f_face_Bilinear
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

ERR = integrate_approx_results(DVEC, 10:10:200, "f_face_DAG_k=1.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_fface_nDAG_approx.png")
approx_error_plot2(DVEC); plt = current(); savefig(plt, "paperfigs/SunFlower_barbara_fface_nDAG_no_frames.png")


## Graph signal approximation
f = f_trouser_Bilinear
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

num_kept_coeffs = 10:10:200; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "f_trouser_DAG_k=1.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_ftrouser_nDAG_approx.png")
approx_error_plot2(DVEC); plt = current(); savefig(plt, "paperfigs/SunFlower_barbara_ftrouser_nDAG_no_frames.png")
