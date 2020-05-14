using Plots, LightGraphs
include(joinpath("..", "src", "func_includer.jl"))

## Build Graph
N1, N2 = 11, 5; G = LightGraphs.grid([N1,N2]); N = nv(G)
X = zeros(N1, N2, 2); for i in 1:N1; for j in 1:N2; X[i,j,1] = i; X[i,j,2] = j; end; end; X = reshape(X, (N,2))
L = Matrix(laplacian_matrix(G))
lamb, V = eigen(L)
V = (V' .* sign.(V[1,:]))'
Q = incidence_matrix(G; oriented = true)
W = 1.0 * adjacency_matrix(G)

## Test
Q2 = [Q -Q]
m2 = size(Q, 2) * 2
f = spike(1,N) - spike(N,N)
md = Model(with_optimizer(Clp.Optimizer, LogLevel = 0))
@variable(md, w[1:m2] >= 0.0)
@objective(md, Min, sum(w))
@constraint(md, Q2 * w .== f)
JuMP.optimize!(md)
wt = abs.(JuMP.value.(w))


gplot(W, X); plot!([X[1,2] X[2,1]]', [X[1,1] X[2,2]]', linecolor = :red, linewidth = 3, arrow = 0.4, aspect_ratio = 1, legend = false)
