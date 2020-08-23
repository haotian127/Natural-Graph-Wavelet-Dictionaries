## Convert the barbara_gray.bmp to a matrix
# using FileIO, Images, JLD
# img_path = joinpath(@__DIR__, "..", "datasets", "barbara_gray.bmp")
# img = FileIO.load(img_path)
# mat = convert(Array{Float64}, Gray.(img))
# JLD.save(joinpath(@__DIR__, "..", "datasets", "barbara_gray_matrix.jld"), "barbara", mat)

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

# Check locations for sampling
# heatmap(barbara, yflip=true, ratio=1, c=:greys); sample_location_plt = scatter_gplot!(X_transform; ms = 1, smallValFirst = false, c = :red)
# savefig(sample_location_plt, "paperfigs/Sunflower_barbara_eye.png")

# Near neighbor rendering to sunflower graph
# f_eye_NN = NN_rendering(X_transform, barbara)
f_eye_Bilinear = Bilinear_rendering(X_transform, barbara)

# Overlay the rendering result to the original image and check if the result makes sense
# heatmap(barbara, yflip=true, ratio=1, c=:greys, clim=(0,1)); scatter_gplot!(X_transform; marker = f_eye_NN, ms = 1, smallValFirst = false, c = :greys)
# heatmap(barbara, yflip=true, ratio=1, c=:greys, clim=(0,1)); scatter_gplot!(X_transform; marker = f_eye_Bilinear, ms = 1, smallValFirst = false, c = :greys)

# Plot rendering results
# scatter_gplot(X_sf .* 50; marker = f_eye_NN, ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); nn_render_result = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, clim=(0,1), frame = :none)
# savefig(nn_render_result, "paperfigs/Sunflower_barbara_NN_feye.png")

# scatter_gplot(X_sf .* 50; marker = f_eye_Bilinear, ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); bilinear_render_result = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, clim=(0,1), frame = :none)
# savefig(bilinear_render_result, "paperfigs/Sunflower_barbara_Bilinear_feye.png")



## focus on the face, i.e., f_face_NN, f_face_Bilinear
X_transform = transform2D(X_sf; s = 20, t = [398, 137]) # t = [398, 137] is the location of the face.

# Check locations for sampling
# heatmap(barbara, yflip=true, ratio=1, c=:greys); sample_location_plt = scatter_gplot!(X_transform; ms = 1, smallValFirst = false, c = :red)
# savefig(sample_location_plt, "paperfigs/Sunflower_barbara_face.png")

# Near neighbor rendering to sunflower graph
# f_face_NN = NN_rendering(X_transform, barbara)
f_face_Bilinear = Bilinear_rendering(X_transform, barbara)

# Overlay the rendering result to the original image and check if the result makes sense
# heatmap(barbara, yflip=true, ratio=1, c=:greys, clim=(0,1)); scatter_gplot!(X_transform; marker = f_face_NN, ms = 1, smallValFirst = false, c = :greys)
# heatmap(barbara, yflip=true, ratio=1, c=:greys, clim=(0,1)); scatter_gplot!(X_transform; marker = f_face_Bilinear, ms = 1, smallValFirst = false, c = :greys)

# Plot rendering results
# scatter_gplot(X_sf .* 50; marker = f_face_NN, ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); nn_render_result = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, clim=(0,1), frame = :none)
# savefig(nn_render_result, "paperfigs/Sunflower_barbara_NN_fface.png")

# scatter_gplot(X_sf .* 50; marker = f_face_Bilinear, ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); bilinear_render_result = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, clim=(0,1), frame = :none)
# savefig(bilinear_render_result, "paperfigs/Sunflower_barbara_Bilinear_fface.png")



## focus on the trouser, i.e., f_trouser_NN, f_trouser_Bilinear
X_transform = transform2D(X_sf; s = 20, t = [280, 320]) # t = [280, 320] is the location of the trouser.

# Check locations for sampling
# heatmap(barbara, yflip=true, ratio=1, c=:greys); sample_location_plt = scatter_gplot!(X_transform; ms = 1, smallValFirst = false, c = :red)
# savefig(sample_location_plt, "paperfigs/Sunflower_barbara_trouser.png")

# Near neighbor rendering to sunflower graph
# f_trouser_NN = NN_rendering(X_transform, barbara)
f_trouser_Bilinear = Bilinear_rendering(X_transform, barbara)

# Overlay the rendering result to the original image and check if the result makes sense
# heatmap(barbara, yflip=true, ratio=1, c=:greys, clim=(0,1)); scatter_gplot!(X_transform; marker = f_trouser_NN, ms = 1, smallValFirst = false, c = :greys)
# heatmap(barbara, yflip=true, ratio=1, c=:greys, clim=(0,1)); scatter_gplot!(X_transform; marker = f_trouser_Bilinear, ms = 1, smallValFirst = false, c = :greys)

# Plot rendering results
# scatter_gplot(X_sf .* 50; marker = f_trouser_NN, ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); nn_render_result = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, clim=(0,1), frame = :none)
# savefig(nn_render_result, "paperfigs/Sunflower_barbara_NN_ftrouser.png")

# scatter_gplot(X_sf .* 50; marker = f_trouser_Bilinear, ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); bilinear_render_result = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, clim=(0,1), frame = :none)
# savefig(bilinear_render_result, "paperfigs/Sunflower_barbara_Bilinear_ftrouser.png")


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

ERR = Array{Float64,1}[]
num_kept_coeffs = 10:10:200
for i in 1:length(DVEC)
    dvec = DVEC[i]
    N = length(dvec)
    dvec_norm = norm(dvec,2)
    dvec_sort = sort(dvec.^2) # the smallest first
    er = reverse(cumsum(dvec_sort))/N # this is the MSE
    push!(ERR, er[num_kept_coeffs])
end
using CSV
frames_approx_res = CSV.File(joinpath(@__DIR__, "..", "datasets", "f_eye_DAG_k=1.csv"))
er_soft_cluster_frame = [frames_approx_res[i][2] for i in 1:length(frames_approx_res)]
push!(ERR, er_soft_cluster_frame)
er_SGWT = [frames_approx_res[i][3] for i in 1:length(frames_approx_res)]
push!(ERR, er_SGWT)

approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
# savefig(approx_error_plt, "paperfigs/RGC100_fz_distDAG_normalized_reconstruct_errors_with_frames.png")
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_feye_nDAG_approx.png")


# ## Show some important NGW basis vectors
# importance_idx = sortperm(abs.(dvec_spectral), rev = true)
# for i = 2:6
#     scatter_gplot(X_sf .* 50; marker = Wav_dual[:,importance_idx[i]], ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); important_NGW_basis_vectors = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, frame = :none)
#     savefig(important_NGW_basis_vectors, "figs/Barbara_Sunflower_Bilinear_feye_PC_NGW_important_basis_vector$(i).png")
# end
#
# importance_idx = sortperm(abs.(dvec_varimax), rev = true)
# for i = 2:6
#     scatter_gplot(X_sf .* 50; marker = Wav_varimax[:,importance_idx[i]], ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); important_NGW_basis_vectors = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, frame = :none)
#     savefig(important_NGW_basis_vectors, "figs/Barbara_Sunflower_Bilinear_feye_varimax_NGW_important_basis_vector$(i).png")
# end





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

ERR = Array{Float64,1}[]
num_kept_coeffs = 10:10:200
for i in 1:length(DVEC)
    dvec = DVEC[i]
    N = length(dvec)
    dvec_norm = norm(dvec,2)
    dvec_sort = sort(dvec.^2) # the smallest first
    er = reverse(cumsum(dvec_sort))/N # this is the MSE
    push!(ERR, er[num_kept_coeffs])
end
using CSV
frames_approx_res = CSV.File(joinpath(@__DIR__, "..", "datasets", "f_face_DAG_k=1.csv"))
er_soft_cluster_frame = [frames_approx_res[i][2] for i in 1:length(frames_approx_res)]
push!(ERR, er_soft_cluster_frame)
er_SGWT = [frames_approx_res[i][3] for i in 1:length(frames_approx_res)]
push!(ERR, er_SGWT)

approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_fface_nDAG_approx.png")


## Show some important NGW basis vectors
# importance_idx = sortperm(abs.(dvec_spectral), rev = true)
# for i = 2:6
#     scatter_gplot(X_sf .* 50; marker = Wav_dual[:,importance_idx[i]], ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); important_NGW_basis_vectors = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, frame = :none)
#     savefig(important_NGW_basis_vectors, "figs/Barbara_Sunflower_Bilinear_fface_PC_NGW_important_basis_vector$(i).png")
# end
#
# importance_idx = sortperm(abs.(dvec_varimax), rev = true)
# for i = 2:6
#     scatter_gplot(X_sf .* 50; marker = Wav_varimax[:,importance_idx[i]], ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); important_NGW_basis_vectors = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, frame = :none)
#     savefig(important_NGW_basis_vectors, "figs/Barbara_Sunflower_Bilinear_fface_varimax_NGW_important_basis_vector$(i).png")
# end



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

ERR = Array{Float64,1}[]
num_kept_coeffs = 10:10:200
for i in 1:length(DVEC)
    dvec = DVEC[i]
    N = length(dvec)
    dvec_norm = norm(dvec,2)
    dvec_sort = sort(dvec.^2) # the smallest first
    er = reverse(cumsum(dvec_sort))/N # this is the MSE
    push!(ERR, er[num_kept_coeffs])
end
using CSV
frames_approx_res = CSV.File(joinpath(@__DIR__, "..", "datasets", "f_trouser_DAG_k=1.csv"))
er_soft_cluster_frame = [frames_approx_res[i][2] for i in 1:length(frames_approx_res)]
push!(ERR, er_soft_cluster_frame)
er_SGWT = [frames_approx_res[i][3] for i in 1:length(frames_approx_res)]
push!(ERR, er_SGWT)

approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_ftrouser_nDAG_approx.png")



## Show some important NGW basis vectors
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:6
    scatter_gplot(X_sf .* 50; marker = Wav_dual[:,importance_idx[i]], ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); important_NGW_basis_vectors = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, frame = :none)
    savefig(important_NGW_basis_vectors, "figs/Barbara_Sunflower_Bilinear_ftrouser_PC_NGW_important_basis_vector$(i).png")
end

importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:6
    scatter_gplot(X_sf .* 50; marker = Wav_varimax[:,importance_idx[i]], ms = LinRange(3.0, 7.0, N), smallValFirst = false, c=:greys); important_NGW_basis_vectors = plot!(xlim = [-100,100], ylim = [-100,100], yflip = true, frame = :none)
    savefig(important_NGW_basis_vectors, "figs/Barbara_Sunflower_Bilinear_ftrouser_varimax_NGW_important_basis_vector$(i).png")
end
