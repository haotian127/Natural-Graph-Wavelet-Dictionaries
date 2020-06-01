"""
    SoftFilter(dist, j; σ = 0.3, β = 4)

SOFTFILTER return the membership vector which concentrates on φⱼ₋₁∈V⃰, based on the non-trivial eigenvector metric `dist`.

# Input Arguments
- `dist::Matrix{Float64}`: N by N matrix measuring behaviorial difference between graph Laplacian eigenvectors
- `j::Int64`: concentrate on φⱼ.
- `σ::Float64`: default is 0.3. Gaussian parameter of the variance.
- `β::Int64`: default is 4. Gaussian parameter of tailing (decay rate).

# Output Argument
- `f::Array{Float64}`: membership vector in G⃰, i.e., spectral filter in G.

"""
function SoftFilter(dist, j; σ = 0.3, β = 4)
    dist = standardize(dist)
    N = size(dist,1)
    exp_dist = exp.(-(dist ./ σ).^β)
    row_sum = sum(exp_dist, dims = 2)[:]
    f = zeros(N)
    for i in 1:N
        f[i] = exp_dist[i,j] / row_sum[i]
    end
    return f
end

# standardize the input distance matrix
function standardize(dist)
    N = size(dist,1)
    return dist ./ maximum(dist[2:N,2:N])
end


"""
    SC_NGW_frame(dist, 𝛷; σ = 0.3, β = 4)

SC\\_NGW\\_ FRAME return the Soft Clustering NGW frame Ψ[j,n,:] is the wavelet focused on node n, with filter focused on φⱼ₋₁∈V⃰.

# Input Arguments
- `dist::Matrix{Float64}`: N by N matrix measuring behaviorial difference between graph Laplacian eigenvectors.
- `𝛷::Matrix{Float64}`: the matrix of graph Laplacian eigenvectors.
- `σ::Float64`: default is 0.3. SoftFilter parameter of variance.
- `β::Int64`: default is 4. SoftFilter parameter of tailing (decay rate).

# Output Argument
- `Ψ::Tensor{Float64}`: Soft Clustering NGW frame, (N, N, N) tensor.

"""
function SC_NGW_frame(dist, 𝛷; σ = 0.3, β = 4)
    N = size(dist,1)
    Ψ = zeros(N,N,N)
    for j = 1:N, n = 1:N
        f = SoftFilter(dist, j; σ = σ, β = β)
        wavelet = 𝛷 * Diagonal(f) * 𝛷' * spike(n,N)
        Ψ[j,n,:] = wavelet ./ norm(wavelet)
    end
    return Ψ
end
