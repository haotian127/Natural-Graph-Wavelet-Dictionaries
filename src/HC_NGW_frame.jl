"""
    hardClustering(W⃰, K)

HARDCLUSTERING partition the dual graph G⃰ = (V⃰, E⃰, W⃰) into K clusters based on spectral clustering method.

# Input Arguments
- `W⃰::Matrix{Float64}`: N by N weighted adjacency matrix measuring behaviorial affinities between graph Laplacian eigenvectors.
- `K::Int64`: the number of dual graph clusters

# Output Argument
- `dualClusters::Array{Array{Int64}}`: dual graph cluster indices.

"""
function hardClustering(W⃰, K)
    D⃰ = Diagonal(sum(W⃰; dims = 1)[:])
    L⃰ = Matrix(D⃰ - W⃰)
    # This is Lv = λDv case, i.e., Lrw's eigenvectors
    Φ⃰ = eigen(L⃰, Matrix(D⃰)).vectors
    dualClusters = spectral_clustering(Φ⃰, K)
    return dualClusters
end



"""
    HC_NGW_frame(dist, 𝛷; σ = 0.3, β = 4)

HC\\_NGW\\_FRAME return the Hard Clustering NGW frame Ψ[j,n,:] is the wavelet focused on node n, with jᵗʰ eigenvector cluster.

# Input Arguments
- `W⃰::Matrix{Float64}`: N by N weighted adjacency matrix measuring behaviorial affinities between graph Laplacian eigenvectors.
- `𝛷::Matrix{Float64}`: the matrix of graph Laplacian eigenvectors.
- `K::Int64`: the number of dual graph clusters

# Output Argument
- `Ψ::Tensor{Float64}`: Soft Clustering NGW frame, (N, N, N) tensor.

"""
function HC_NGW_frame(W⃰, 𝛷, K)
    N = size(𝛷,1)
    dualClusters = hardClustering(W⃰, K)
    Ψ = zeros(K,N,N)
    for j = 1:K, n = 1:N
        f = characteristic(dualClusters[j],N)
        wavelet = 𝛷 * Diagonal(f) * 𝛷' * spike(n,N)
        Ψ[j,n,:] = wavelet ./ norm(wavelet)
    end
    return Ψ, dualClusters
end
