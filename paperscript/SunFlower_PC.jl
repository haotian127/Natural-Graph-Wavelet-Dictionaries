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

## Assemble wavelet packets
ht_vlist, ht_elist = HTree_VElist(𝛷,W)
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
ht_elist_varimax = ht_elist_dual

parent_vertex = HTree_findParent(ht_vlist)
wavelet_packet = HTree_wavelet_packet(𝛷,ht_vlist,ht_elist)

parent_varimax = HTree_findParent(ht_elist_varimax)
wavelet_packet_varimax = HTree_wavelet_packet_varimax(𝛷,ht_elist_varimax)

parent_dual = HTree_findParent(ht_vlist_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)


## Graph signal
f = zeros(N); f[:] .= sin.((X[:,1] .^ 2 + X[:,2] .^ 2) .* 10)
# pyplot(dpi = 400); gplot(1.0*adjacency_matrix(G), X, width = 1, color = :blue); plot!(aspect_ratio=1, framestyle = :none); SunFlower_signal = scatter_gplot!(X; marker = f, ms = (1:N) / 50, smallValFirst = false)
# savefig(SunFlower_signal, "paperfigs/SunFlower_signal.png")


## Best basis selection algorithm
ht_coeff, ht_coeff_L1 = HTree_coeff_wavelet_packet(f,wavelet_packet)
dvec = best_basis_algorithm(ht_coeff_L1, parent_vertex)
Wav = assemble_wavelet_basis(dvec,wavelet_packet)

ht_coeff_varimax, ht_coeff_L1_varimax = HTree_coeff_wavelet_packet(f,wavelet_packet_varimax)
dvec_varimax = best_basis_algorithm(ht_coeff_L1_varimax, parent_varimax)
Wav_varimax = assemble_wavelet_basis(dvec_varimax,wavelet_packet_varimax)

ht_coeff_dual, ht_coeff_L1_dual = HTree_coeff_wavelet_packet(f,wavelet_packet_dual)
dvec_dual = best_basis_algorithm(ht_coeff_L1_dual, parent_dual)
Wav_dual = assemble_wavelet_basis(dvec_dual,wavelet_packet_dual)

## sort wavelets by centered locations
# heatmap(sortWaveletsByCenteredLocations(Wav_dual))

## plot approx. error figure w.r.t. fraction of kept coefficients
approx_error_plot([Wav_varimax, Wav_dual, 𝛷, I], f; label = ["Varimax NGW Packet" "Pair Clustering NGW Packet" "Laplacian eigenvectors" "Standard Basis"], Save = false, path = "paperfigs/SunFlower_PC_DAG_reconstruct_errors.png")
current()
