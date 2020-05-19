mutable struct GraphStruct
    G::SimpleGraph{Int64}
    X::Array{Float64,2}
    N::Int64
    L::Array{Int64,2}
    lamb::Array{Float64,1}
    V::Array{Float64,2}
end
