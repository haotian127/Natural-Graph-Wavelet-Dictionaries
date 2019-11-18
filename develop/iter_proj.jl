using Plots, LightGraphs, Random, Distances, JLD, LaTeXStrings, Clustering

include("../../src/func_includer.jl")
include("Proj.jl")

G = loadgraph("../../datasets/new_toronto_graph.lgz")
N = nv(G)
X = load("../../datasets/new_toronto.jld","xy")
dist_X = pairwise(Euclidean(),X; dims = 1)
A = 1.0 .* adjacency_matrix(G)
W = A .* dualGraph(dist_X; method = "inverse")
L = Matrix(Diagonal(sum(W;dims = 1)[:]) - W)
Q = incidence_matrix(G; oriented = true)
lamb, V = eigen(L)
sgn = (maximum(V, dims = 1)[:] .> -minimum(V, dims = 1)[:]) .* 2 .- 1
V = (V' .* sgn)'

edge_length = sum((Q' * X).^2, dims = 2)[:]

# save("../../datasets/wToronto_datasets.jld","L",L,"V",V,"lamb",lamb,"Q",Q,"X",X)


numClusters = 5
IDX = assignments(kmeans(V[:,2:numClusters]',numClusters))
scatter_gplot(X[:,1],X[:,2]; marker = IDX)


thres = .05
Cluster_vlist = []
active_eigenVecList = []
active_eigenVecList_length = Int.(zeros(numClusters))
for k = 1:numClusters
    vlist = findall(IDX .== k)
    push!(Cluster_vlist,vlist)
    energy_vec = sum(V[vlist,:] .^ 2, dims = 1)[:]
    active_vec_ind = findall(energy_vec .> thres)
    active_vec_ind = setdiff(active_vec_ind, [1,2]) ## remove DC and Fiedler
    push!(active_eigenVecList, active_vec_ind)
    active_eigenVecList_length[k] = length(active_vec_ind)
end

Φ = zeros(N,length(Cluster_vlist[1]))
B = V[:,active_eigenVecList[1]]
for k in 1:length(Cluster_vlist[1])
    global B
    wavelet = Proj(spike(Cluster_vlist[1][k],N),B)
    Φ[:,k] .= wavelet ./ norm(wavelet)
    B = wavelet_perp_Matrix(wavelet,B)
end

heatmap(Φ'*Φ)
scatter_gplot(X[:,1],X[:,2]; marker = Φ[:,241])
