using LightGraphs
"""
    StarLikeTree(n, m)

STARLIKETREE construct a simple star-like tree graph with n branches and m nodes on each branch.

# Input Arguments
- `n::Int64`: number of branches.
- `m::Int64`: number of nodes on each branch.

# Output Argument
- `G::SimpleGraph{Int64}`: a simple unweighted graph of the star-like tree.
- `X::Matrix{Float64}`: a matrix whose i-th row represent the 2D coordinates of the i-th node.

"""
function StarLikeTree(n, m)
    N = n * m + 1; G = Graph(N); X = zeros(N,2);
    for i = 1:n, j = 1:m
        add_edge!(G, Edge(1,i+1))
        add_edge!(G, Edge(i+1+(j-1)*n,i+1+j*n))
        X[(j-1)*n+i+1,:] = j .* [cos(2*π/n * i), sin(2*π/n * i)]
    end
    return G, X
end
