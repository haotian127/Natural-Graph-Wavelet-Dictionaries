## Load packages and functions
using MAT, LinearAlgebra, SparseArrays, Optim, JLD

include(joinpath("..", "src", "eigHAD_Distance.jl"))

## Compute graph Laplacian eigenvectors
L = Matrix(matread(joinpath(@__DIR__, "..", "datasets", "toronto.mat"))["L"])
lamb, 𝛷 = eigen(L); sgn = (maximum(𝛷, dims = 1)[:] .> -minimum(𝛷, dims = 1)[:]) .* 2 .- 1; 𝛷 = Matrix((𝛷' .* sgn)')

## Compute aHAD affinity
aHAD = eigHAD_Affinity(𝛷, lamb)
JLD.save(joinpath(@__DIR__, "..", "datasets", "Toronto_aHAD.jld"), "aHAD", aHAD)
