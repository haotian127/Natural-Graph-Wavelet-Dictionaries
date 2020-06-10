## Load packages and functions
using MultivariateStats
include(joinpath("..", "src", "func_includer.jl"))

### Unweighted Version
## Build unweighted RGC#100 graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "RGC100.lgz")); N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,1:2]
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L); sgn = (maximum(V, dims = 1)[:] .> -minimum(V, dims = 1)[:]) .* 2 .- 1; V = Matrix((V' .* sgn)')
Q = incidence_matrix(G; oriented = true)

## Test unweighted ROT distance on RGC100
# Runtime for α = 1 case: 4 hours (163.35 G allocations: 6.885 TiB, 5.22% gc time)
@time distROT = eigROT_Distance(V.^2, Q)
# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_unweighted_alp1.jld"), "distROT", distROT)


### Weighted Version
## Build weighted RGC#100 graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "RGC100.lgz")); N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")
A = 1.0 .* adjacency_matrix(G)
dist_X = pairwise(Euclidean(),X; dims = 1)
Weight = A .* dualGraph(dist_X; method = "inverse") # weighted adjacence matrix
L = Matrix(Diagonal(sum(Weight;dims = 1)[:]) - Weight)
lamb, V = eigen(L)
sgn = (maximum(V, dims = 1)[:] .> -minimum(V, dims = 1)[:]) .* 2 .- 1
V = (V' .* sgn)'; V = Matrix(V);
Q = incidence_matrix(G; oriented = true)
edge_lengths = sqrt.(sum((Q' * X).^2, dims = 2)[:])
## Test weighted ROT distance on RGC100
# Runtime for α = 1 case: 4 hours (175.62 G allocations: 7.767 TiB, 5.64% gc time)
# @time distROT = eigROT_Distance(V.^2, Q; le = edge_lengths)
# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_weighted_alp1.jld"), "distROT", distROT)

# Runtime for α = 0.5 case: seconds (175.62 G allocations: 7.767 TiB, 5.76% gc time)
# @time distROT = eigROT_Distance(V.^2, Q; le = edge_lengths, α = 0.5)
# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distROT_weighted_alp05.jld"), "distROT", distROT)

## Try out OptimalTransport.jl
using OptimalTransport, SimpleWeightedGraphs
sources, destinations =Int64[], Int64[]; for e in collect(edges(G)); push!(sources, e.src); push!(destinations, e.dst); end
wG = SimpleWeightedGraph(sources, destinations, edge_lengths) #edge weights are the Euclidean length of the edge
costmx = floyd_warshall_shortest_paths(G, weights(wG)).dists

P = V.^2

i = 565; j = 721;
p = P[:,i]; q = P[:,j]; p ./= norm(p,1); q ./= norm(q,1)
u = (p-q .> 0) .* (p-q); v =  - (p-q .< 0) .* (p-q);
print("=================\n")
@time _, d = ROT_Distance(u, v, Q; le = edge_lengths)
@time emdcost = emd2(u, v, costmx)

# Runtime: give up after 8.5 hours running
# Error log:
# """
# C:\Users\lihao\.julia\conda\3\lib\site-packages\ot\lp\__init__.py:421: UserWarning: Problem infeasible. Check that a and b are in the simplex
#   check_result(result_code)
# RESULT MIGHT BE INACURATE * 50
# Max number of iteration reached, currently 100000. Sometimes iterations go on in cycle even though the solution has been reached, to check if it's the case here have a look at the minimal reduced cost. If it is very close to machine precision, you might actually have the correct solution, if not try setting the maximum number of iterations a bit higher
# C:\Users\lihao\.julia\conda\3\lib\site-packages\ot\lp\__init__.py:421: UserWarning: numItermax reached before optimality. Try to increase numItermax.
#   check_result(result_code)
# """
# print("==================\n")
# @time distEMD = eigEMD_Distance(V.^2, costmx)
# JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_distEMD_weighted.jld"), "distEMD", distEMD)

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

plotly()
scatter(X0,Y0,Z0, zcolor = 1:N, m=(:grays, 0.6, 2),cbar = true, legend = false)
scatter!(X1,Y1,Z1, m=(:red, 0.9,3), cbar = false, legend = false)
scatter!(X2,Y2,Z2, zcolor = 1:Int(round(N/length(branchfreq))):N,m=(:viridis, 0.8,5), cbar = false, legend = false)
scatter!(X3,Y3,Z3, zcolor = [1,N], m=(:isolum, 1,6), cbar = false, legend = false)
plot!(xaxis = "X1", yaxis = "X2", zaxis = "X3")
