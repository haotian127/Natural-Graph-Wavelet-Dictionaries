## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build weighted graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "new_toronto_graph.lgz")); N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"xy")
W = 1.0 .* adjacency_matrix(G)
dist_X = pairwise(Euclidean(),X; dims = 1)
Weight = W .* dualGraph(dist_X; method = "inverse") # weighted adjacence matrix
L = Matrix(Diagonal(sum(Weight; dims = 1)[:]) - Weight)
𝛌, 𝚽 = eigen(L); sgn = (maximum(𝚽, dims = 1)[:] .> -minimum(𝚽, dims = 1)[:]) .* 2 .- 1; 𝚽 = 𝚽 .* sgn';
Q = incidence_matrix(G; oriented = true)
edge_weight = 1 ./ sqrt.(sum((Q' * X).^2, dims = 2)[:])

## Build Dual Graph
distDAG = eigDAG_Distance(𝚽,Q,N; edge_weight = edge_weight)
W_dual = sparse(dualGraph(distDAG)) #required: sparse dual weighted adjacence matrix

## Assemble wavelet packets
ht_elist_PC, ht_vlist_PC = HTree_EVlist(𝚽,W_dual)
wavelet_packet_PC = JLD.load(joinpath(@__DIR__, "..", "datasets", "Toronto_DAG_NGWP.jld"), "wavelet_packet_dual")

ht_elist_VM = ht_elist_PC
wavelet_packet_VM = JLD.load(joinpath(@__DIR__, "..", "datasets", "Toronto_DAG_NGWP.jld"), "wavelet_packet_varimax")

## 1. f_density
gr(dpi = 200)
f = zeros(N); for i in 1:N; f[i] = length(findall(dist_X[:,i] .< 1/minimum(edge_weight))); end #fneighbor
gplot(W, X; width=1); scatter_gplot!(X; marker = f, smallValFirst = true, ms = 3); signal_plt = plot!(aspect_ratio = 1, grid = false)
# savefig(signal_plt, "paperfigs/Toronto_fdensity.png")

DVEC = signal_transform_coeff(f, ht_elist_PC, ht_elist_VM, wavelet_packet_PC, wavelet_packet_VM, 𝚽, W, X)

approx_error_plot2(DVEC); approx_error_plt = plot!(legend = :topright, xguidefontsize=14, yguidefontsize=14, legendfontsize=10)
# savefig(approx_error_plt, "paperfigs/Toronto_fdensity_DAG_approx.png")

# Show some important PC-NGWP basis vectors
parent_PC = HTree_findParent(ht_elist_PC); Wav_PC = best_basis_selection(f, wavelet_packet_PC, parent_PC); dvec_PC = Wav_PC' * f
importance_idx = sortperm(abs.(dvec_PC), rev = true)
for i = 2:10
    w = Wav_PC[:,importance_idx[i]]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    # savefig(important_NGW_basis_vectors, "paperfigs/Toronto_fdensity_DAG_PC_NGW_important_basis_vector$(i).png")
end

# Show some important VM-NGWP basis vectors
parent_VM = HTree_findParent(ht_elist_VM); Wav_VM = best_basis_selection(f, wavelet_packet_VM, parent_VM); dvec_VM = Wav_VM' * f
importance_idx = sortperm(abs.(dvec_VM), rev = true)
for i = 2:10
    gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_VM[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    # savefig(important_NGW_basis_vectors, "paperfigs/Toronto_fdensity_DAG_VM_NGW_important_basis_vector$(i).png")
end

# Show some important HGLET vectors
using MTSG
tmp=zeros(length(f),1); tmp[:,1]=f; G_Sig=GraphSig(1.0*W, xy=X, f=tmp)
G_Sig = Adj2InvEuc(G_Sig); GP = partition_tree_fiedler(G_Sig,:Lrw)
dmatrixH, dmatrixHrw, dmatrixHsym = HGLET_Analysis_All(G_Sig, GP) # expansion coefficients of 3-way HGLET bases
dvec_hglet, BS_hglet, trans_hglet = HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixH, dmatrixHrw = dmatrixHrw, dmatrixHsym = dmatrixHsym, costfun = 1) # best-basis among all combinations of bases
importance_idx = sortperm(dvec_hglet.^2; rev = true)[1:10]
for i = 2:10
    important_HGLET_basis_vector, _ = HGLET_Synthesis(reshape(spike(importance_idx[i],N), N, 1), GP, BS_hglet, G_Sig)
    w = important_HGLET_basis_vector[:,1]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_HGLET_basis_vectors = plot!(grid = false)
    # savefig(important_HGLET_basis_vectors, "paperfigs/Toronto_fdensity_HGLET_important_basis_vector$(i).png")
end

for i = 2:10
    println("(j, k, l) = ", HGLET_jkl(GP, BS_hglet.levlist[importance_idx[i]][1], BS_hglet.levlist[importance_idx[i]][2]))
end

## 2. f_pedestrain
fp = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fp")
gplot(W, X; width=1); scatter_gplot!(X; marker = fp, smallValFirst = true, ms = 3); signal_plt = plot!(aspect_ratio = 1, grid = false)
# savefig(signal_plt, "paperfigs/Toronto_fp.png")

DVEC = signal_transform_coeff(fp, ht_elist_PC, ht_elist_VM, wavelet_packet_PC, wavelet_packet_VM, 𝚽, W, X)

approx_error_plot2(DVEC); approx_error_plt = plot!(legend = :topright, xguidefontsize=14, yguidefontsize=14, legendfontsize=10)
# savefig(approx_error_plt, "paperfigs/Toronto_fp_DAG_approx.png")

# Show some important PC-NGWP basis vectors
parent_PC = HTree_findParent(ht_elist_PC); Wav_PC = best_basis_selection(fp, wavelet_packet_PC, parent_PC); dvec_PC = Wav_PC' * fp
importance_idx = sortperm(abs.(dvec_PC), rev = true)
for i = 2:10
    w = Wav_PC[:,importance_idx[i]]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    # savefig(important_NGW_basis_vectors, "paperfigs/Toronto_fp_DAG_PC_NGW_important_basis_vector$(i).png")
end

# Show some important VM-NGWP basis vectors
parent_VM = HTree_findParent(ht_elist_VM); Wav_VM = best_basis_selection(fp, wavelet_packet_VM, parent_VM); dvec_VM = Wav_VM' * fp
importance_idx = sortperm(abs.(dvec_VM), rev = true)
for i = 2:10
    gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_VM[:,importance_idx[i]], smallValFirst = true, ms = 3); important_NGW_basis_vectors = plot!(grid = false)
    # savefig(important_NGW_basis_vectors, "paperfigs/Toronto_fp_DAG_VM_NGW_important_basis_vector$(i).png")
end

# Show some important HGLET vectors
tmp=zeros(length(fp),1); tmp[:,1]=fp; G_Sig=GraphSig(1.0*W, xy=X, f=tmp)
G_Sig = Adj2InvEuc(G_Sig); GP = partition_tree_fiedler(G_Sig,:Lrw)
dmatrixH, dmatrixHrw, dmatrixHsym = HGLET_Analysis_All(G_Sig, GP) # expansion coefficients of 3-way HGLET bases
dvec_hglet, BS_hglet, trans_hglet = HGLET_GHWT_BestBasis(GP, dmatrixH = dmatrixH, dmatrixHrw = dmatrixHrw, dmatrixHsym = dmatrixHsym, costfun = 1) # best-basis among all combinations of bases
importance_idx = sortperm(dvec_hglet.^2; rev = true)[1:10]
for i = 1:10
    important_HGLET_basis_vector, _ = HGLET_Synthesis(reshape(spike(importance_idx[i],N), N, 1), GP, BS_hglet, G_Sig)
    w = important_HGLET_basis_vector[:,1]
    sgn = (maximum(w) > -minimum(w)) * 2 - 1
    gplot(W, X; width=1); scatter_gplot!(X; marker = sgn .* w, smallValFirst = true, ms = 3); important_HGLET_basis_vectors = plot!(grid = false)
    # savefig(important_HGLET_basis_vectors, "paperfigs/Toronto_fp_HGLET_important_basis_vector$(i).png")
end

for i = 2:10
    println("(j, k, l) = ", HGLET_jkl(GP, BS_hglet.levlist[importance_idx[i]][1], BS_hglet.levlist[importance_idx[i]][2]))
end
