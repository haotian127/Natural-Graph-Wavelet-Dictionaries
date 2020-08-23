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
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = HTree_wavelet_packet_varimax(𝛷,ht_elist_varimax)

# JLD.save(joinpath(@__DIR__, "..", "datasets", "Toronto_DAG_NGWP.jld"), "wavelet_packet_varimax", wavelet_packet_varimax, "wavelet_packet_dual", wavelet_packet_dual)
# wavelet_packet_varimax = JLD.load(joinpath(@__DIR__, "..", "datasets", "Toronto_DAG_NGWP.jld"), "wavelet_packet_varimax")
# wavelet_packet_dual = JLD.load(joinpath(@__DIR__, "..", "datasets", "Toronto_DAG_NGWP.jld"), "wavelet_packet_dual")

## Graph signal
# fp = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fp")
# fv = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"fv")
# f = exp.( (- (X[:,1] .+ 79.4).^2 - (X[:,2] .- 43.65).^2) ./ 0.01 ) # fgaussian
f = zeros(N); for i in 1:N; f[i] = length(findall(dist_X[:,i] .< 1/minimum(edge_weight))); end #fneighbor
# pyplot(dpi = 400); gplot(1.0*adjacency_matrix(G), X, width = 1, color = :blue); plot!(aspect_ratio=1, framestyle = :none); Toronto_signal = scatter_gplot!(X; marker = f, smallValFirst = true)
# savefig(Toronto_signal, "paperfigs/Toronto_fneighbor.png")

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

tmp=zeros(length(f),1); tmp[:,1]=f;
G_Sig=GraphSig(1.0*W, xy=X, f=tmp)
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

ERR = Array{Float64,1}[]
num_kept_coeffs = 10:10:1130
for i in 1:length(DVEC)
    dvec = DVEC[i]
    N = length(dvec)
    dvec_norm = norm(dvec,2)
    dvec_sort = sort(dvec.^2) # the smallest first
    er = reverse(cumsum(dvec_sort))/N # this is the MSE
    push!(ERR, er[num_kept_coeffs])
end
using CSV
frames_approx_res = CSV.File(joinpath(@__DIR__, "..", "datasets", "toronto_density_DAG_k=7.csv"))
er_soft_cluster_frame = [frames_approx_res[i][2] for i in 1:length(frames_approx_res)]
push!(ERR, er_soft_cluster_frame[1:length(num_kept_coeffs)])
er_SGWT = [frames_approx_res[i][3] for i in 1:length(frames_approx_res)]
push!(ERR, er_SGWT[1:length(num_kept_coeffs)])

approx_error_plot3(ERR; num_kept_coeffs = num_kept_coeffs); approx_error_plt = current()
savefig(approx_error_plt, "paperfigs/Toronto_fdensity_nDAG_approx.png")

# approx_error_plot2(DVEC); current()


## Show some important NGW basis vectors
# # PC NGW
# importance_idx = sortperm(abs.(dvec_spectral), rev = true)
# for i = 2:6
#     gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_dual[:,importance_idx[i]], ms = 2); important_NGW_basis_vectors = plot!(frame = :none)
#     savefig(important_NGW_basis_vectors, "figs/Toronto_fp_PC_NGW_important_basis_vector$(i).png")
# end
#
# # Varimax NGW
# importance_idx = sortperm(abs.(dvec_varimax), rev = true)
# for i = 2:6
#     gplot(W, X; width=1); scatter_gplot!(X; marker = Wav_varimax[:,importance_idx[i]], ms = 2); important_NGW_basis_vectors = plot!(frame = :none)
#     savefig(important_NGW_basis_vectors, "figs/Toronto_fp_varimax_NGW_important_basis_vector$(i).png")
# end
