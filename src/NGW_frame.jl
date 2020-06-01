function softmax_clustering(dist; j = 1)

    """
    arguments:
    dist, n by n matrix measuring behaviorial difference between graph Laplacian eigenvectors
    j, concentrate on the j-th frequence

    output: filter, diagonal matrix Fj
    """
    exp_dist = exp.(-dist)
    s = sum(exp_dist, dims = 1)
    n = size(dist)[1]
    f = zeros(n)
    for l in 1:n
        f[l] = exp_dist[l,j] / s[l]
    end
    return Diagonal(f)
end

function spike(i,n)
    a = zeros(n)
    a[i] = 1
    return a
end

function NGW_frame(dist, L, V; cluster_method = "softmax_clustering", variance = 0.3)
    """Arguments: dist, distance matrix (or affinity matrix) of eigenvectors;
                  L, graph Laplacian matrix;
                  V, matrix of Laplacian eigenvectors.
       Return: Natual Graph Wavelet frame W, shape = (n,n,n) tensor,
       e.g., W[1,2,:]: wavelet concentrate on 1st eigenvector and 2nd location (i.e., spike 2)
    """
    n = nv(G)        # n = number of vertices = number of clusters
    W = zeros(n,n,n) # W stores n x n wavelets
    if cluster_method == "softmax_clustering"
        for j = 1:n, k = 1:n
            F = softmax_clustering(dist; j = j)
            W[j,k,:] = V * F * V' * spike(k,n)
        end
    elseif cluster_method == "gaussian_clustering"
        for j = 1:n, k = 1:n
            F = gaussian_clustering(dist; j = j, k = variance)
            W[j,k,:] = V * F * V' * spike(k,n)
        end
    end

    return W
end

function NGW_frame_coeff(f, W)
    n = length(f)
    Coff = zeros(size(W)[1:2])
    for j = 1:size(W)[1]
        for k = 1:size(W)[2]
            w = W[j,k,:]
            Coff[j,k] = w' * f
        end
    end
    return Coff
end

function gaussian_clustering(dist; j = 1, k = 0.3, m = 4)

    """
    arguments:
    dist, n by n matrix measuring behaviorial difference between graph Laplacian eigenvectors
    j, concentrate on the j-th frequence
    k, hyperparameter of the variance
    m,

    output: filter, diagonal matrix Fj
    """
    exp_dist = exp.(-(dist ./ k).^m)
    #exp_dist = 1 ./ (1 .+ (dist ./ k)^2)
    s = sum(exp_dist, dims = 1)
    n = size(dist)[1]
    f = zeros(n)
    for l in 1:n
        f[l] = exp_dist[l,j] / s[l]
    end
    return Diagonal(f)
end

function standardize(dist)
    n = size(dist)[1]
    s = maximum(dist[2,2:n])
    return dist ./ s
end


function NGW_frame_partial(dist, L, V; spec_freq = [2,3,4,5,6,7,8], cluster_method = "softmax_clustering", variance = 0.1)
    """Arguments: dist, distance matrix (or affinity matrix) of eigenvectors;
                  L, graph Laplacian matrix;
                  V, matrix of Laplacian eigenvectors.
       Return: Natual Graph Wavelet frame W, shape = (n,n,n) tensor,
       e.g., W[1,2,:]: wavelet concentrate on 1st eigenvector and 2nd location (i.e., spike 2)
    """
    n = nv(G)
    J = length(spec_freq)        # n = number of vertices = number of clusters
    W = zeros(J,n,n) # W stores n x n wavelets
    if cluster_method == "softmax_clustering"
        for j = 1:J, k = 1:n
            F = softmax_clustering(dist; j = spec_freq[j])
            W[j,k,:] = V * F * V' * spike(k,n)
        end
    elseif cluster_method == "gaussian_clustering"
        for j = 1:J, k = 1:n
            F = gaussian_clustering(dist; j = spec_freq[j], k = variance)
            W[j,k,:] = V * F * V' * spike(k,n)
        end
    end

    return W
end
