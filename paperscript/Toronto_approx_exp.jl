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
distDAG = eigDAG_Distance(𝛷,Q,N; edge_weight = edge_weight)
W_dual = sparse(dualGraph(distDAG)) #required: sparse dual weighted adjacence matrix

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = JLD.load(joinpath(@__DIR__, "..", "datasets", "Toronto_DAG_NGWP.jld"), "wavelet_packet_dual")

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = JLD.load(joinpath(@__DIR__, "..", "datasets", "Toronto_DAG_NGWP.jld"), "wavelet_packet_varimax")

## Graph signal
# pyplot(dpi = 400); gplot(1.0*adjacency_matrix(G), X, width = 1, color = :blue); plot!(aspect_ratio=1, framestyle = :none); Toronto_signal = scatter_gplot!(X; marker = f, smallValFirst = true)
# savefig(Toronto_signal, "paperfigs/Toronto_fneighbor.png")

## 1. f_density
f = zeros(N); for i in 1:N; f[i] = length(findall(dist_X[:,i] .< 1/minimum(edge_weight))); end #fneighbor
gplot(W, X; width=1); scatter_gplot!(X; marker = f, smallValFirst = true, ms = 3); signal_plt = plot!(aspect_ratio = 1, grid = false)
# savefig(signal_plt, "paperfigs/Toronto_fdensity.png")

DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

# num_kept_coeffs = 10:10:1130; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "toronto_density_DAG_k=7.csv")
# approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(grid = false); approx_error_plt = current()
approx_error_plot2(DVEC); plot!(legend = :topright, xguidefontsize=14, yguidefontsize=14, legendfontsize=10); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/Toronto_fdensity_DAG_approx.png")

# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(f, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * f
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    w = Wav_dual[:,importance_idx[i]]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    savefig(important_NGW_basis_vectors, "paperfigs/Toronto_fdensity_DAG_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(f, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * f
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_varimax[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    savefig(important_NGW_basis_vectors, "paperfigs/Toronto_fdensity_DAG_VM_NGW_important_basis_vector$(i).png")
end


# Show some important HGLET vectors
using MTSG
tmp=zeros(length(f),1); tmp[:,1]=f; G_Sig=GraphSig(1.0*W, xy=X, f=tmp)
G_Sig = Adj2InvEuc(G_Sig)
GP = partition_tree_fiedler(G_Sig,:Lrw)
dmatrixH, dmatrixHrw, dmatrixHsym = HGLET_Analysis_All(G_Sig, GP) # expansion coefficients of 3-way HGLET bases
dvec_hglet, BS_hglet, trans_hglet = HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixH, dmatrixHrw = dmatrixHrw, dmatrixHsym = dmatrixHsym, costfun = 1) # best-basis among all combinations of bases
ind = sortperm(dvec_hglet.^2; rev = true)[1:10]
# for i = 2:10
#     important_HGLET_basis_vector, _ = HGLET_Synthesis(reshape(spike(ind[i],N), N, 1), GP, BS_hglet, G_Sig)
#     w = important_HGLET_basis_vector[:,1]
#     sgn = (maximum(w) > -minimum(w)) * 2 - 1
#     gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_HGLET_basis_vector_plt = plot!(grid = false)
#     savefig(important_HGLET_basis_vector_plt, "paperfigs/Toronto_fdensity_HGLET_important_basis_vector$(i).png")
# end

for i = 2:10
    println("(j, k, l) = ", HGLET_jkl(GP, BS_hglet.levlist[ind[i]][1], BS_hglet.levlist[ind[i]][2]))
end
## 2. f_pedestrain
fp = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fp")
gplot(W, X; width=1); scatter_gplot!(X; marker = fp, smallValFirst = true, ms = 3); signal_plt = plot!(aspect_ratio = 1, grid = false)
# savefig(signal_plt, "paperfigs/Toronto_fp.png")

DVEC = signal_transform_coeff(fp, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

# num_kept_coeffs = 10:10:1130; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "toronto_pedestrian_DAGFull_k=7.csv")
# approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(grid = false, legend = :bottomleft); approx_error_plt = current()
approx_error_plot2(DVEC); plot!(legend = :topright, xguidefontsize=14, yguidefontsize=14, legendfontsize=10); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/Toronto_fp_DAG_approx.png")

# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(fp, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * fp
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    w = Wav_dual[:,importance_idx[i]]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    savefig(important_NGW_basis_vectors, "paperfigs/Toronto_fp_DAG_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(fp, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * fp
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_varimax[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    savefig(important_NGW_basis_vectors, "paperfigs/Toronto_fp_DAG_VM_NGW_important_basis_vector$(i).png")
end

## 3. f_vehicle
fv = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fv")
gplot(W, X; width=1); scatter_gplot!(X; marker = fv, smallValFirst = true, ms = 3); signal_plt = plot!(aspect_ratio = 1, grid = false)
savefig(signal_plt, "paperfigs/Toronto_fv.png")

DVEC = signal_transform_coeff(fv, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

# num_kept_coeffs = 10:10:1130; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "toronto_vehicle_DAGFull_k=7.csv")
# approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(grid = false, legend = :bottomleft); approx_error_plt = current()
approx_error_plot2(DVEC); plot!(legend = :topright); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/Toronto_fv_DAG_approx.png")



# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(fv, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * fv
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    w = Wav_dual[:,importance_idx[i]]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    savefig(important_NGW_basis_vectors, "paperfigs/Toronto_fv_DAG_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(fv, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * fv
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_varimax[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    savefig(important_NGW_basis_vectors, "paperfigs/Toronto_fv_DAG_VM_NGW_important_basis_vector$(i).png")
end




## 4. log10(f_pedestrain)
f = log10.(1 .+ load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fp"))
gplot(W, X; width=1); scatter_gplot!(X; marker = f, smallValFirst = true, ms = 3); signal_plt = plot!(aspect_ratio = 1, grid = false)
savefig(signal_plt, "paperfigs/Toronto_log10_fp.png")

DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

# num_kept_coeffs = 10:10:1130; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "toronto_pedestrian_DAGFull_k=7.csv")
# approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(grid = false, legend = :bottomleft); approx_error_plt = current()
approx_error_plot2(DVEC); plot!(legend = :topright); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/Toronto_log10_fp_DAG_approx.png")

# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(fp, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * fp
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    w = Wav_dual[:,importance_idx[i]]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    savefig(important_NGW_basis_vectors, "paperfigs/Toronto_log10_fp_DAG_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(fp, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * fp
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_varimax[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    savefig(important_NGW_basis_vectors, "paperfigs/Toronto_log10_fp_DAG_VM_NGW_important_basis_vector$(i).png")
end

## 5. log10(f_vehicle)
f = log10.(1 .+ load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fv"))
gplot(W, X; width=1); scatter_gplot!(X; marker = f, smallValFirst = true, ms = 3); signal_plt = plot!(aspect_ratio = 1, grid = false)
savefig(signal_plt, "paperfigs/Toronto_log10_fv.png")

DVEC = signal_transform_coeff(f, ht_elist_dual, ht_elist_varimax, wavelet_packet_dual, wavelet_packet_varimax, 𝛷, W, X)

# num_kept_coeffs = 10:10:1130; ERR = integrate_approx_results(DVEC, num_kept_coeffs, "toronto_vehicle_DAGFull_k=7.csv")
# approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); plot!(grid = false, legend = :bottomleft); approx_error_plt = current()
approx_error_plot2(DVEC); plot!(legend = :topright); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/Toronto_log10_fv_DAG_approx.png")



# Show some important NGW basis vectors
parent_dual = HTree_findParent(ht_elist_dual); Wav_dual = best_basis_selection(fv, wavelet_packet_dual, parent_dual); dvec_spectral = Wav_dual' * fv
importance_idx = sortperm(abs.(dvec_spectral), rev = true)
for i = 2:10
    w = Wav_dual[:,importance_idx[i]]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    savefig(important_NGW_basis_vectors, "paperfigs/Toronto_log10_fv_DAG_PC_NGW_important_basis_vector$(i).png")
end

parent_varimax = HTree_findParent(ht_elist_varimax); Wav_varimax = best_basis_selection(fv, wavelet_packet_varimax, parent_varimax); dvec_varimax = Wav_varimax' * fv
importance_idx = sortperm(abs.(dvec_varimax), rev = true)
for i = 2:10
    gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_varimax[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    savefig(important_NGW_basis_vectors, "paperfigs/Toronto_log10_fv_DAG_VM_NGW_important_basis_vector$(i).png")
end
