## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build weighted graph
G, L, X = SunFlowerGraph(); N = nv(G)
lamb, 𝛷 = eigen(Matrix(L)); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)
edge_weight = [e.weight for e in edges(G)]

## Build Dual Graph
distDAG = eigDAG_Distance(𝛷,Q,N; edge_weight = edge_weight)
W_dual = sparse(dualGraph(distDAG)) #required: sparse dual weighted adjacence matrix
# W_dual = 1.0 * adjacency_matrix(path_graph(N))

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = HTree_wavelet_packet_varimax(𝛷,ht_elist_varimax)


## Graph signal
f = zeros(N); f[:] .= sin.((X[:,1] .^ 2 + X[:,2] .^ 2) .* 10)
# pyplot(dpi = 400); gplot(1.0*adjacency_matrix(G), X, width = 1, color = :blue); plot!(aspect_ratio=1, framestyle = :none); SunFlower_signal = scatter_gplot!(X; marker = f, ms = (1:N) / 30, smallValFirst = false)
# savefig(SunFlower_signal, "paperfigs/SunFlower_signal.png")


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


approx_error_plot2([dvec_haar[:], dvec_walsh[:], dvec_Laplacian[:], dvec_c2f[:], dvec_f2c[:], dvec_eghwt[:], dvec_spectral[:], dvec_varimax[:]]); sunflower_approx_error_plt = current()
# savefig(sunflower_approx_error_plt, "paperfigs/SunFlower_reconstruct_errors.png")
