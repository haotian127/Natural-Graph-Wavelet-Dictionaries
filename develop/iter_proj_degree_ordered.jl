using Plots, LightGraphs, Random, Distances, JLD, LaTeXStrings, Clustering

include("../../src/func_includer.jl")
include("Proj.jl")

G = loadgraph("../../datasets/new_toronto_graph.lgz")
N = nv(G)
@load("../../datasets/new_toronto.jld")
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

vert_deg = sum(A,dims = 2)[:]


numClusters = 2
IDX = assignments(kmeans(V[:,2:numClusters]',numClusters))
scatter_gplot(X[:,1],X[:,2]; marker = IDX)


# thres = .05
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



### eigenvector classification based on energy distribution on each spatial cluster
Ve = V.^2
energyDistr_eigvecs = zeros(numClusters,N)
for j in 1:numClusters
    energyDistr_eigvecs[j,:] .= sum(Ve[Cluster_vlist[j],:],dims = 1)[:]
end

Cluster_vlist_length = length.(Cluster_vlist)
# cvl_ind = sortperm(Cluster_vlist_length; rev = false)
cvl_ind = 1:numClusters

tmp_ind = []
tmp = sortperm(energyDistr_eigvecs[cvl_ind[1],:];rev = true)[1:Cluster_vlist_length[cvl_ind[1]]]
Cluster_elist = [tmp]
for j in cvl_ind[2:end]
    global tmp_ind,tmp
    tmp_ind = union(tmp_ind,tmp)
    rest_ind = setdiff(1:N,tmp_ind)
    tmp = rest_ind[sortperm(energyDistr_eigvecs[j,rest_ind]; rev = true)[1:Cluster_vlist_length[j]]]
    push!(Cluster_elist,tmp)
end

# s = 0
# for j = 1:numClusters
#     global s
#     s += sum(energyDistr_eigvecs[cvl_ind[j],Cluster_elist[j]])
# end
# print(s)
# print(sum([energyDistr_eigvecs[eig_cidx[k],k] for k = 1:N]))

plot([length.(Cluster_vlist) length.(Cluster_elist)], lab = ["spatial", "spectral"])



### Construct wavelet
Φ = zeros(N,N)
Cluster_start_pts = [sum(length.(Cluster_vlist)[1:k]) for k = 0:numClusters-1]
for j = 1:numClusters
    B = V[:,Cluster_elist[j]]
    for k in 1:length(Cluster_vlist[j])
        wavelet = Proj(spike(Cluster_vlist[j][k],N),B)
        Φ[:,k + Cluster_start_pts[j]] .= wavelet ./ norm(wavelet)
        B = wavelet_perp_Matrix(wavelet,B)
    end
end
# print(norm(Φ'*Φ - I)/norm(Φ'*Φ))

# wlt = Φ[:,1800];scatter_gplot(X[:,1],X[:,2]; marker = abs.(wlt), ms = 4 .*exp.(abs.(wlt)))
# plot(wlt)


Wav,~ = qr(Φ)
Wav = Matrix(Wav)

heatmap(Wav)

###############################
### Graph Signal Analysis
###############################

# ### mutilated Gaussian signal
# f_mutilatedGaussian = zeros(N)
# constant = 0.2
# ind = 1:N
# f_mutilatedGaussian[ind] .= exp.((-(X[ind,1] .+ 79.4).^2 - (X[ind,2] .- 43.70).^2)/0.02)
# ind = findall(X[:,1] .< -79.4)
# f_mutilatedGaussian[ind] .= f_mutilatedGaussian[ind] .+ constant
# f = f_mutilatedGaussian

# ### multi gaussian
# f_multiGaussian = exp.((-(X[:,1] .+ 79.3).^2 - (X[:,2] .- 43.75).^2)/0.01) + exp.((-(X[:,1] .+ 79.55).^2 - (X[:,2] .- 43.65).^2)/0.001)+exp.((-(X[:,1] .+ 79.5).^2 - (X[:,2] .- 43.76).^2)/0.005)
# f = f_multiGaussian + 0.2 .* rand(N)

# f = fv

# f = fp

# f = randn(N)

f = spike(100,N) + spike(101,N) + spike(102,N) + spike(1000,N)


scatter_gplot(X[:,1],X[:,2]; marker = f)



error_Wavelet = [1.0]
error_Laplacian = [1.0]
for frac = 0.01:0.01:0.3
    numKept = Int(ceil(frac * N))
    ## wavelet varimax reconstruction
    coeff_Wavelet = Wav' * f
    ind = sortperm(coeff_Wavelet.^2, rev = true)
    ind = ind[numKept+1:end]
    # rc_f = Wav[:,ind] * coeff_Wavelet[ind]
    # push!(error_Wavelet, norm(f - rc_f) / norm(f))
    push!(error_Wavelet, norm(coeff_Wavelet[ind])/norm(f))

    ## Laplacian reconstruction
    coeff_Laplacian = V' * f
    ind = sortperm(coeff_Laplacian.^2, rev = true)
    ind = ind[numKept+1:end]
    # rc_f = V[:,ind] * coeff_Laplacian[ind]
    # push!(error_Laplacian, norm(f - rc_f) / norm(f))
    push!(error_Laplacian, norm(coeff_Laplacian[ind])/norm(f))
end

# gr(dpi = 300)
fraction = 0:0.01:0.3
plt = plot(fraction,[error_Wavelet error_Laplacian], yaxis=:log, lab = ["Wavelets","Laplacian"])
# savefig(plt,"figs/Toronto_errorResults_mutilatedGaussian_signal.png")
