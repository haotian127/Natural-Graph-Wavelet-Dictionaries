## Load packages and functions
include(joinpath("..", "src", "func_includer.jl"))

## Build graph
G = loadgraph(joinpath(@__DIR__, "..", "datasets", "RGC100.lgz"))
L = Matrix(laplacian_matrix(G))
lamb, 𝛷 = eigen(L); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')

## Compute aHAD affinity
aHAD = eigHAD_Affinity(𝛷, lamb)
JLD.save(joinpath(@__DIR__, "..", "datasets", "RGC100_aHAD_unweighted.jld"), "aHAD", aHAD)
