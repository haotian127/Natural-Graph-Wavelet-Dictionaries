## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))
using LaTeXStrings
gr(dpi=400)

## Build graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "RGC100.lgz"))
N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,1:2]
L = Matrix(laplacian_matrix(G))
lamb, 𝛷 = eigen(L); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

###########################################################################################
### nDAG metric
###########################################################################################
## Build Dual Graph
distDAG = eigDAG_Distance_normalized(𝛷,Q,N)
W_dual = sparse(dualGraph(distDAG))

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_distDAG_normalized_unweighted_wavelet_packet_varimax.jld"), "wavelet_packet_varimax")

# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distDAG_unweighted_wavelet_packet_varimax.jld"), "wavelet_packet_varimax", wavelet_packet_varimax)

## focus nodes
node_lists = [698,784,977]
# for node in node_lists
#     plt = scatter_gplot(X; marker = spike(node, N))
#     savefig(plt, "paperfigs/RGC100_interest_node_$(node)")
# end

## Show SC-Frame vectors
sc_wavelet_1_698 = [CSV.File(joinpath(@__DIR__, "..", "datasets", "wavelet_1_698.csv"); delim=",", header = false)[1][k] for k in 1:N]
gplot(W, X; width=1); scatter_gplot!(X; marker = sc_wavelet_1_698, c = :viridis, ms = 3); plt = plot!( aspect_ratio = 1)
savefig(plt, "paperfigs/RGC100_nDAG_SC_Frame_wavelet_1_698_no_clim")

sc_wavelet_2_698 = [CSV.File(joinpath(@__DIR__, "..", "datasets", "wavelet_2_698.csv"); delim=",", header = false)[1][k] for k in 1:N]
gplot(W, X; width=1); scatter_gplot!(X; marker = sc_wavelet_2_698, c = :viridis, ms = 3); plt = plot!( aspect_ratio = 1)
savefig(plt, "paperfigs/RGC100_nDAG_SC_Frame_wavelet_2_698_no_clim")

sc_wavelet_1_784 = [CSV.File(joinpath(@__DIR__, "..", "datasets", "wavelet_1_784.csv"); delim=",", header = false)[1][k] for k in 1:N]
gplot(W, X; width=1); scatter_gplot!(X; marker = sc_wavelet_1_784, c = :viridis, ms = 3); plt = plot!( aspect_ratio = 1)
savefig(plt, "paperfigs/RGC100_nDAG_SC_Frame_wavelet_1_784_no_clim")

sc_wavelet_2_784 = [CSV.File(joinpath(@__DIR__, "..", "datasets", "wavelet_2_784.csv"); delim=",", header = false)[1][k] for k in 1:N]
gplot(W, X; width=1); scatter_gplot!(X; marker = sc_wavelet_2_784, c = :viridis, ms = 3); plt = plot!( aspect_ratio = 1)
savefig(plt, "paperfigs/RGC100_nDAG_SC_Frame_wavelet_2_784_no_clim")

sc_wavelet_1_977 = [CSV.File(joinpath(@__DIR__, "..", "datasets", "wavelet_1_977.csv"); delim=",", header = false)[1][k] for k in 1:N]
gplot(W, X; width=1); scatter_gplot!(X; marker = sc_wavelet_1_977, c = :viridis, ms = 3); plt = plot!( aspect_ratio = 1)
savefig(plt, "paperfigs/RGC100_nDAG_SC_Frame_wavelet_1_977_no_clim")

sc_wavelet_2_977 = [CSV.File(joinpath(@__DIR__, "..", "datasets", "wavelet_2_977.csv"); delim=",", header = false)[1][k] for k in 1:N]
gplot(W, X; width=1); scatter_gplot!(X; marker = sc_wavelet_2_977, c = :viridis, ms = 3); plt = plot!( aspect_ratio = 1)
savefig(plt, "paperfigs/RGC100_nDAG_SC_Frame_wavelet_2_977_no_clim")


## Show some PC NGW basis

# # Level 2
# # Father wavelet
# print("location: ", sortperm(wavelet_packet_dual[lvl][i][node,:].^2, rev = true)[1:5])
# lvl = 3; i = 1; k = 232; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.4, 0.5), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
# savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_father_wavelet")
#
# # Mother wavelet
# lvl = 3; i = 2; k = 227; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!( aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
# savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_mother_wavelet")
# # node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# # scatter_gplot(X; marker = spike(node, N))
#
# # Packet wavelet
# lvl = 3; i = 3; k = 159; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
# savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_wavelet_packet_vec")


# Level 3
# Father wavelet
# print("location: ", findmax(wavelet_packet_dual[lvl][i][node,:].^2)[2])
lvl = 4; i = 1; k = 98; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_father_wavelet")

# Mother wavelet
lvl = 4; i = 2; k = 64; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_mother_wavelet")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))

# Packet wavelet
lvl = 4; i = 3; k = 159; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_wavelet_packet_vec")



# Level 4
# Father wavelet
lvl = 5; i = 1; k = 1; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_father_wavelet")
node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node) # node = 698
# scatter_gplot(X; marker = spike(node, N))

# Mother wavelet
lvl = 5; i = 2; k = 20; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_mother_wavelet")
node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node) # node = 784
# scatter_gplot(X; marker = spike(node, N))

# Packet wavelet
lvl = 5; i = 5; k = 7; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_wavelet_packet_vec")
node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node) # node = 977
# scatter_gplot(X; marker = spike(node, N))


# Level 5
# Father wavelet
lvl = 6; i = 1; k = 3; gplot(W, X; width=2); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_father_wavelet")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))

# Mother wavelet
lvl = 6; i = 2; k = 11; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_mother_wavelet")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))

# Packet wavelet
lvl = 6; i = 10; k = 5; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_PC_NGWP_lvl$(lvl-1)_wavelet_packet_vec")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))





## Show some varimax NGW basis


# Level 3
# Father wavelet
lvl = 4; i = 1; k = 4; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_father_wavelet")

# Mother wavelet
lvl = 4; i = 2; k = 57; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_mother_wavelet")

# Packet wavelet
lvl = 4; i = 3; k = 29; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_wavelet_packet_vec")


# Level 4
# Father wavelet
lvl = 5; i = 1; k = 4; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_father_wavelet")

# Mother wavelet
lvl = 5; i = 2; k = 50; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_mother_wavelet")

# Packet wavelet
lvl = 5; i = 5; k = 29; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_wavelet_packet_vec")



# Level 5
# Father wavelet
lvl = 6; i = 1; k = 4; gplot(W, X; width=2); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_father_wavelet")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))

# Mother wavelet
lvl = 6; i = 2; k = 4; gplot(W, X; width=1); scatter_gplot!(X; marker = -wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.125), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_mother_wavelet")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))

# Packet wavelet
lvl = 6; i = 10; k = 13; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.25), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_nDAG_VM_NGWP_lvl$(lvl-1)_wavelet_packet_vec")
# node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)
# scatter_gplot(X; marker = spike(node, N))






###########################################################################################
### ROT metric
###########################################################################################
## Build Dual Graph
distROT = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_unweighted_alp1.jld"), "distROT")
W_dual = sparse(dualGraph(distROT))

## Assemble wavelet packets
ht_elist_dual, ht_vlist_dual = HTree_EVlist(𝛷,W_dual)
wavelet_packet_dual = HTree_wavelet_packet(𝛷,ht_vlist_dual,ht_elist_dual)

ht_elist_varimax = ht_elist_dual
wavelet_packet_varimax = JLD.load(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_unweighted_alp1_wavelet_packet_varimax.jld"), "wavelet_packet_varimax")
# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_unweighted_alp1_wavelet_packet_varimax.jld"), "wavelet_packet_varimax", wavelet_packet_varimax)

## Show some PC NGW basis

# 1. Father wavelet
lvl = 11; i = 1; k = 2; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.2), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_PC_NGWP_lvl$(lvl-1)_father_wavelet")
node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node) #363

lvl = 8; i = 1; k=sortperm(wavelet_packet_dual[lvl][i][node,:].^2, rev = true)[1]; print("location: ", sortperm(wavelet_packet_dual[lvl][i][node,:].^2, rev = true)[1:3])
gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.2), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_PC_NGWP_lvl$(lvl-1)_father_wavelet")

lvl = 5; i = 1; k=sortperm(wavelet_packet_dual[lvl][i][node,:].^2, rev = true)[1]; print("location: ", sortperm(wavelet_packet_dual[lvl][i][node,:].^2, rev = true)[1:3])
gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.4), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_PC_NGWP_lvl$(lvl-1)_father_wavelet")

# 2. Mother wavelet

lvl = 11; i = 2; k = 1; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.2), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_PC_NGWP_lvl$(lvl-1)_mother_wavelet")
node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)

lvl = 8; i = 2; k = 2; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.2), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_PC_NGWP_lvl$(lvl-1)_mother_wavelet")

lvl = 5; i = 2; k = 2; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.2), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_PC_NGWP_lvl$(lvl-1)_mother_wavelet")


# 3. Packet wavelet

lvl = 11; i = 3; k = 1; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.3, 0.3), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_PC_NGWP_lvl$(lvl-1)_wavelet_packet_vec")
node = findmax(wavelet_packet_dual[lvl][i][:,k].^2)[2]; print("location: ", node)

lvl = 8; i = 3; k = 2; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.3, 0.3), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_PC_NGWP_lvl$(lvl-1)_wavelet_packet_vec")

lvl = 5; i = 3; k = 2; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_dual[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.3, 0.3), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_PC_NGWP_lvl$(lvl-1)_wavelet_packet_vec")





## Show some varimax NGW basis

# 1. Father wavelet
lvl = 11; i = 1; k = 4; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.2), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_VM_NGWP_lvl$(lvl-1)_father_wavelet")
node = findmax(wavelet_packet_varimax[lvl][i][:,k].^2)[2]; print("location: ", node)

lvl = 8; i = 1; k=sortperm(wavelet_packet_varimax[lvl][i][node,:].^2, rev = true)[1]; print("location: ", sortperm(wavelet_packet_varimax[lvl][i][node,:].^2, rev = true)[1:3])
gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.1, 0.2), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_VM_NGWP_lvl$(lvl-1)_father_wavelet")

lvl = 5; i = 1; k=sortperm(wavelet_packet_varimax[lvl][i][node,:].^2, rev = true)[1]; print("location: ", sortperm(wavelet_packet_varimax[lvl][i][node,:].^2, rev = true)[1:3])
gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.4), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_VM_NGWP_lvl$(lvl-1)_father_wavelet")

# 2. Mother wavelet

lvl = 11; i = 2; k = 1; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.2), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_VM_NGWP_lvl$(lvl-1)_mother_wavelet")
node = findmax(wavelet_packet_varimax[lvl][i][:,k].^2)[2]; print("location: ", node)

lvl = 8; i = 2; k = 2; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.2), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_VM_NGWP_lvl$(lvl-1)_mother_wavelet")

lvl = 5; i = 2; k = 2; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.2, 0.2), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_VM_NGWP_lvl$(lvl-1)_mother_wavelet")


# 3. Packet wavelet

lvl = 11; i = 3; k = 1; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.3, 0.3), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_VM_NGWP_lvl$(lvl-1)_wavelet_packet_vec")
node = findmax(wavelet_packet_varimax[lvl][i][:,k].^2)[2]; print("location: ", node)

lvl = 8; i = 3; k = 2; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.3, 0.3), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_VM_NGWP_lvl$(lvl-1)_wavelet_packet_vec")

lvl = 5; i = 3; k = 2; gplot(W, X; width=1); scatter_gplot!(X; marker = wavelet_packet_varimax[lvl][i][:,k], c = :viridis, ms = 3); plt = plot!(clim = (-0.3, 0.3), aspect_ratio = 1, title = latexstring("\\psi_{", i-1, ",", k-1, "}", "^{(", lvl-1, ")}"))
savefig(plt, "paperfigs/RGC100_ROT_VM_NGWP_lvl$(lvl-1)_wavelet_packet_vec")
