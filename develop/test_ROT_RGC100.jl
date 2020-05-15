using Plots, LightGraphs, JLD, Distances, MultivariateStats
include(joinpath("..", "src", "func_includer.jl"))

## Build weighted RGC#100 graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "RGC100.lgz")); N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")
A = 1.0 .* adjacency_matrix(G)
dist_X = pairwise(Euclidean(),X; dims = 1)
Weight = A .* dualGraph(dist_X; method = "inverse") # weighted adjacence matrix
L = Matrix(Diagonal(sum(Weight;dims = 1)[:]) - Weight)
lamb, V = eigen(L)
sgn = (maximum(V, dims = 1)[:] .> -minimum(V, dims = 1)[:]) .* 2 .- 1
V = (V' .* sgn)'
Q = incidence_matrix(G; oriented = true)
edge_lengths = sqrt.(sum((Q' * X).^2, dims = 2)[:])
# edge_weights = 1 ./ edge_lengths
## Test weighted ROT distance on RGC100
# Runtime for α = 1 case: 4 hours (175.62 G allocations: 7.767 TiB, 5.64% gc time)
# @time distROT = eigROT_Distance(V.^2, Q; le = edge_lengths)
# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_weighted_alp1.jld"), "distROT", distROT)

# Runtime for α = 0.5 case:
@time distROT = eigROT_Distance(V.^2, Q; le = edge_lengths, α = 0.5)
JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_weighted_alp05.jld"), "distROT", distROT)

## MDS: low dimensional embedding
E = transform(fit(MDS, distROT, maxoutdim = 3, distances = true))

# scatter(E[1,:], E[2,:], E[3,:], zcolor = 1:N, m=(:viridis, 0.8, 3), cbar = true, legend = false)

# find the eigenvectors which concentrated on the upper-left branch
Ve100 = V.^2

branchind = 220:358 # upper-left branch
tmp=zeros(N); for k=1:N tmp[k]= norm(Ve100[branchind,k]) end
branchfreq=findall(tmp .> 0.7)

junctionind = findall(degree(G) .> 2)
tmp = zeros(N); for k=1:N tmp[k]= norm(Ve100[junctionind,k]) end
junctionfreq=findall(tmp .> 0.5)

X0 = E[1,1:N]
Y0 = E[2,1:N]
Z0 = E[3,1:N]

# localized on junction eigenvectors
X1 = E[1,junctionfreq]
Y1 = E[2,junctionfreq]
Z1 = E[3,junctionfreq]

# semi-oscillation eigenvectors
X2 = E[1,branchfreq]
Y2 = E[2,branchfreq]
Z2 = E[3,branchfreq]

# DC vector and Fiedler vector
X3 = E[1,1:2]
Y3 = E[2,1:2]
Z3 = E[3,1:2]

plotlyjs()
scatter(X0,Y0,Z0, zcolor = 1:N, m=(:grays, 0.6, 2),cbar = true, legend = false)
scatter!(X1,Y1,Z1, m=(:red, 0.9,3), cbar = false, legend = false)
scatter!(X2,Y2,Z2, zcolor = 1:Int(round(N/length(branchfreq))):N,m=(:viridis, 0.8,5), cbar = false, legend = false)
scatter!(X3,Y3,Z3, zcolor = [1,N], m=(:isolum, 1,6), cbar = false, legend = false)
plot!(xaxis = "X1", yaxis = "X2", zaxis = "X3")
