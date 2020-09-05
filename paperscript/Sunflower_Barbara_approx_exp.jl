## Load image and packages of NGW
include(joinpath("..", "src", "func_includer.jl"))
using MAT, MTSG; gr(dpi = 400)
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

## 1. barbara eye
f = matread(joinpath(@__DIR__, "..", "datasets", "sunflower_barbara.mat"))["f_eye"]
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

num_kept_coeffs = 10:10:200; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "f_eye_DAG_k=1.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_feye_nDAG_approx.png")

# show some important Laplacian eigenvectors
dvec_Laplacian = 𝛷' * f; importance_idx = sortperm(abs.(dvec_Laplacian), rev = true)
for i = 2:10
    scatter_gplot(X; marker = 𝛷[:,importance_idx[i]], ms = LinRange(3.0, 11.0, N), smallValFirst = false, c = :greys); important_basis_vectors = plot!(clim = (-0.2,0.2), xlim = [-1,1], ylim = [-1,1], yflip = true, frame = :none)
    savefig(important_basis_vectors, "paperfigs/SunFlower_barbara_feye_Laplacian_important_basis_vector$(i).png")
end

# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(f, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * f
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    scatter_gplot(X; marker = Wav_dual[:,importance_idx[i]], ms = LinRange(3.0, 11.0, N), smallValFirst = false, c = :greys); important_NGW_basis_vectors = plot!(clim = (-0.2,0.2), xlim = [-1,1], ylim = [-1,1], yflip = true, frame = :none)
    savefig(important_NGW_basis_vectors, "paperfigs/SunFlower_barbara_feye_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(f, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * f
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    scatter_gplot(X; marker = Wav_varimax[:,importance_idx[i]], ms = LinRange(3.0, 11.0, N), smallValFirst = false, c = :greys); important_NGW_basis_vectors = plot!(clim = (-0.2,0.2), xlim = [-1,1], ylim = [-1,1], yflip = true, frame = :none)
    savefig(important_NGW_basis_vectors, "paperfigs/SunFlower_barbara_feye_VM_NGW_important_basis_vector$(i).png")
end



## 2. barbara face
f = matread(joinpath(@__DIR__, "..", "datasets", "sunflower_barbara.mat"))["f_face"]
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

num_kept_coeffs = 10:10:200; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "f_face_DAG_k=1.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_fface_nDAG_approx.png")

# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(f, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * f
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    scatter_gplot(X; marker = Wav_dual[:,importance_idx[i]], ms = LinRange(3.0, 11.0, N), smallValFirst = false, c = :greys); important_NGW_basis_vectors = plot!(clim = (-0.2,0.2), xlim = [-1,1], ylim = [-1,1], yflip = true, frame = :none)
    savefig(important_NGW_basis_vectors, "paperfigs/SunFlower_barbara_fface_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(f, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * f
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    scatter_gplot(X; marker = Wav_varimax[:,importance_idx[i]], ms = LinRange(3.0, 11.0, N), smallValFirst = false, c = :greys); important_NGW_basis_vectors = plot!(clim = (-0.2,0.2), xlim = [-1,1], ylim = [-1,1], yflip = true, frame = :none)
    savefig(important_NGW_basis_vectors, "paperfigs/SunFlower_barbara_fface_VM_NGW_important_basis_vector$(i).png")
end


## 3. barbara trouser
f = matread(joinpath(@__DIR__, "..", "datasets", "sunflower_barbara.mat"))["f_trouser"]
DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

num_kept_coeffs = 10:10:200; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "f_trouser_DAG_k=1.csv")
approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/SunFlower_barbara_ftrouser_nDAG_approx.png")

# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(f, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * f
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    scatter_gplot(X; marker = Wav_dual[:,importance_idx[i]], ms = LinRange(3.0, 11.0, N), smallValFirst = false, c = :greys); important_NGW_basis_vectors = plot!(clim = (-0.2,0.2), xlim = [-1,1], ylim = [-1,1], yflip = true, frame = :none)
    savefig(important_NGW_basis_vectors, "paperfigs/SunFlower_barbara_ftrouser_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(f, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * f
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    scatter_gplot(X; marker = Wav_varimax[:,importance_idx[i]], ms = LinRange(3.0, 11.0, N), smallValFirst = false, c = :greys); important_NGW_basis_vectors = plot!(clim = (-0.2,0.2), xlim = [-1,1], ylim = [-1,1], yflip = true, frame = :none)
    savefig(important_NGW_basis_vectors, "paperfigs/SunFlower_barbara_ftrouser_VM_NGW_important_basis_vector$(i).png")
end
