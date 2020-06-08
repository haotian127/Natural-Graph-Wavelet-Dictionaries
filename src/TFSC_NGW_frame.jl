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

SC\\_NGW\\_FRAME return the Soft Clustering NGW frame Ψ[j,n,:] is the wavelet focused on node n, with filter focused on φⱼ₋₁∈V⃰.

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

"""
    TFSC_NGW_frame(dist, 𝛷; σ = 0.3, β = 4)

TFSC\\_NGW\\_ FRAME return a M-dim list of the Time-Frequency adapted Soft Clustering NGW frame Ψ[j,n,:] is the wavelet focused on node n, with filter focused on φⱼ₋₁∈V⃰.

# Input Arguments
- `partial_dist_ls::Array{Matrix{Float64}}`: an M-dim array of N by N matrix measuring partial node behaviorial difference between graph Laplacian eigenvectors.
- `𝛷::Matrix{Float64}`: the matrix of graph Laplacian eigenvectors.
- `M::Int64`: the number of graph clusters.
- `γ::Float64`: default is 0.05, the threshold for active eigenvectors on each subgraph Gₖ.
- `σ::Float64`: default is 0.3, the SoftFilter parameter of variance.
- `β::Int64`: default is 4, the SoftFilter parameter of tailing (decay rate).

# Output Argument
- `TF_Ψ::Array{Tensor{Float64}}`: a M-dim array of Time-Frequency adapted Soft Clustering NGW frame, (N, N, N) tensor.

"""
function TFSC_NGW_frame(partial_dist_ls, 𝛷, M, graphClusters, activeEigenVecs; σ = 0.3, β = 4)
    N = size(𝛷,1)
    TF_Ψ = Array{Array{Float64,3},1}()
    for k in 1:M
        J = length(activeEigenVecs[k])
        Ψ = zeros(J,N,N)
        for j in 1:J
            f = SoftFilter(partial_dist_ls[k], activeEigenVecs[k][j]; σ = σ, β = β)
            for n in 1:N
                wavelet = 𝛷 * Diagonal(f) * 𝛷' * spike(n,N)
                Ψ[j,n,:] = wavelet ./ norm(wavelet)
            end
        end
        push!(TF_Ψ, Ψ)
    end
    return TF_Ψ
end

# Find active eigenvectors for each subgraph Gₖ
function find_active_eigenvectors(𝛷, M, graphClusters; γ = 0.05)
    activeEigenVecs = Array{Array{Int64,1},1}()
    for k in 1:M
        currentActiveEigenVecs = Array{Int64,1}()
        for ℓ in 1:N
            if sum(𝛷[graphClusters[k],ℓ].^2) > γ
                push!(currentActiveEigenVecs, ℓ)
            end
        end
        push!(activeEigenVecs, currentActiveEigenVecs)
    end
    return activeEigenVecs
end
