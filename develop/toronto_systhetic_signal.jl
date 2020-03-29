## load packages
using Plots, LightGraphs, JLD, LaTeXStrings, Distances, StatsBase
include(joinpath("..", "src", "func_includer.jl"))

## load graph info
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "new_toronto_graph.lgz"))
N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "new_toronto.jld"),"xy")
W = 1.0 .* adjacency_matrix(G)
dist_X = pairwise(Euclidean(),X; dims = 1)
Weight = W .* dualGraph(dist_X; method = "inverse") # weighted adjacence matrix
L = Matrix(Diagonal(sum(Weight;dims = 1)[:]) - Weight)

## get a sampling distribution by considering density of vertices in terms of locations
Q = incidence_matrix(G; oriented = true)
edge_lengths = sqrt.(sum((Q' * X).^2, dims = 2)[:])
thres = maximum(edge_lengths)
vertex_distr = zeros(N)
for i in 1:N
    vertex_distr[i] = length(findall(dist_X[i,:] .< thres))
end
vertex_distr ./= sum(vertex_distr)
N_sample = 100000
vertex_density = zeros(N)
for x in sample(1:N, Weights(vertex_distr), N_sample)
    vertex_density[x] += 1
end

scatter_gplot(X;marker = vertex_density)

## function to random cross N_step traffic lights
function one_time_cross(vert_begin, G, f_signal; N_step = 5)
    vert = vert_begin
    for i in 1:N_step
        f_signal[vert] += 1
        vert = sample(neighbors(G, vert))
    end
    return f_signal
end

scatter_gplot(X;marker = one_time_cross(1000, G, zeros(N); N_step = 5))

## create systhetic traffic signal (pedestrian volume simulation)

function toronto_systhetic_signal(f, N_sample, N_step)
    for x in sample(1:N, Weights(vertex_distr), N_sample)
        f = one_time_cross(x, G, f; N_step = N_step)
    end
    return f
end

f6 = toronto_systhetic_signal(zeros(N), 1000000, 6)
f12 = toronto_systhetic_signal(zeros(N), 1000000, 12)
f18 = toronto_systhetic_signal(zeros(N), 1000000, 18)
f24 = toronto_systhetic_signal(zeros(N), 1000000, 24)
f30 = toronto_systhetic_signal(zeros(N), 1000000, 30)
scatter_gplot(X;marker = f30)

## save
save(joinpath(@__DIR__, "..", "datasets/toronto_systhetic_signal.jld"), "f6", f6, "f12", f12, "f18", f18, "f24", f24, "f30",f30)
