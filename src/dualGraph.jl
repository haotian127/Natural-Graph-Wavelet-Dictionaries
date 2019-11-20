function dualGraph(D; method = "inverse", σ = 1)
# inputs: eigenvector distance matrix
# outputs: weight matrix of the dual graph
    n = size(D)[1]
    W = zeros(n,n)
    if method == "Gaussian"
        for i = 1:n-1, j = i+1:n
            W[i,j] = exp(- D[i,j] / σ^2)
        end
    elseif method == "inverse"
        for i = 1:n-1, j = i+1:n
            W[i,j] = 1/D[i,j]
        end
    end
    return W + W'
end
