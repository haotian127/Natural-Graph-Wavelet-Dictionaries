"""
    eigDAG_Distance(V, Q, numEigs; edge_weights = 1)

EIGDAG_DISTANCE compute DAG distances between pairwise graph Laplacian eigenvectors.

# Input Arguments
- `V::Matrix{Float64}`: matrix of graph Laplacian eigenvectors, 𝜙ⱼ₋₁ (j = 1,...,size(V,1)).
- `Q::Matrix{Float64}`: incidence matrix of the graph.
- `numEigs::Int`: number of eigenvectors considered.
- `edge_weights`: default value is 1, stands for unweighted graph (i.e., all edge weighs equal to 1).

# Output Argument
- `dis::Matrix{Float64}`: a numEigs x numEigs distance matrix, dis[i,j] = d_DAG(𝜙ᵢ₋₁, 𝜙ⱼ₋₁).

"""
function eigDAG_Distance(V,Q,numEigs; edge_weight = 1)
    # Case: Unweighted Graph
    if edge_weight == 1
        dis = zeros(numEigs,numEigs)
        Ve = abs.(Q' * V)
        for i = 1:numEigs, j = i+1:numEigs
            dis[i,j] = norm(Ve[:,i]-Ve[:,j],2)
        end
    # Case: weighted Graph
    else
        dis = zeros(numEigs,numEigs)
        Ve = abs.(Q' * V)
        for i = 1:numEigs, j = i+1:numEigs
            dis[i,j] = sqrt(sum((Ve[:,i]-Ve[:,j]).^2 .* sqrt.(edge_weight)))
        end
    end

    return dis + dis'
end
