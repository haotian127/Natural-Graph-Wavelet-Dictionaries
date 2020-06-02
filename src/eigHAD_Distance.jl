using Optim

"""
    eigHAD_Distance(𝛷,lamb,numEigs)

EIGHAD\\_DISTANCE compute HAD distance between pairwise graph Laplacian eigenvectors, i.e., d_HAD(𝜙ᵢ₋₁, 𝜙ⱼ₋₁) = -log(a_HAD(𝜙ᵢ₋₁, 𝜙ⱼ₋₁)).

# Input Arguments
- `𝛷::Matrix{Float64}`: matrix of graph Laplacian eigenvectors, 𝜙ⱼ₋₁ (j = 1,...,size(𝛷,1)).
- `lamb::Array{Float64}`: array of eigenvalues. (ascending order)
- `numEigs::Int`: number of eigenvectors considered.

# Output Argument
- `dis::Matrix{Float64}`: a numEigs x numEigs affinity matrix, dis[i,j] = d_HAD(𝜙ᵢ₋₁, 𝜙ⱼ₋₁).
"""
function eigHAD_Distance(𝛷,lamb,numEigs)
    A = eigHAD_Affinity(𝛷,lamb,numEigs)
    dis = -log.(A)
    return dis
end

"""
    eigHAD_Affinity(𝛷,lamb,numEigs)

EIGHAD_AFFINITY compute Hadamard (HAD) affinity between pairwise graph Laplacian eigenvectors.

# Input Arguments
- `𝛷::Matrix{Float64}`: matrix of graph Laplacian eigenvectors, 𝜙ⱼ₋₁ (j = 1,...,size(𝛷,1)).
- `lamb::Array{Float64}`: array of eigenvalues. (ascending order)
- `numEigs::Int`: number of eigenvectors considered.

# Output Argument
- `A::Matrix{Float64}`: a numEigs x numEigs affinity matrix, A[i,j] = a_HAD(𝜙ᵢ₋₁, 𝜙ⱼ₋₁).
"""
function eigHAD_Affinity(𝛷,lamb,numEigs)
    N = size(𝛷,1)
    A = zeros(numEigs,numEigs)
    for i = 2:numEigs
        tmp = Diagonal(𝛷[:,i]) * 𝛷[:,i:N]
        measure = sqrt.(sum(tmp.^2, dims = 1))[:]
        index = findall(measure .> .01/sqrt(N)) .+ (i - 1)
        for j in index
            λ, μ = lamb[i], lamb[j]
            x₀ = 1 ./ (max(λ, μ))
            # Find minimizer t
            result = optimize(t -> abs(exp(-t[1]*λ) + exp(-t[1]*μ) - 1), [x₀], BFGS());
            t = Optim.minimizer(result)[1]
            # Compute Hadamard affinity
            hadamardProd = 𝛷[:,i] .* 𝛷[:,j]
            heatEvolution = 𝛷 * Diagonal(exp.(-t .* lamb)) * 𝛷' * hadamardProd
            A[i,j] = norm(heatEvolution,2) / (norm(hadamardProd,2) + 1e-6)
        end
    end
    A = A + A'
    # Set affinity measure of 𝜙₀ with itself to be the maximum and equals to 1.
    A[1,1] = maximum(A)
    return A ./ A[1,1]
end
