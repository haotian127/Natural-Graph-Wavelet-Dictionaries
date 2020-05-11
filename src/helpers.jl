"""
    findminimum(v, n)

FINDMINIMUM finds the first n smallest elements' indices.

# Input Arguments
- `v::Array{Float64}`: the candidate values for selection.
- `n::Int`: number of smallest elements for consideration.

# Output Argument
- `idx::Array{Int}`: n smallest elements' indices.

"""
function findminimum(v, n)
    idx = sortperm(v)[1:n]
    return idx
end


"""
    spike(i,n)

SPIKE gives the n-dim spike vector with i-th element equals 1.

# Input Arguments
- `i::Int`: index for one.
- `n::Int`: dimension of the target spike vector.

# Output Argument
- `a::Array{Float}`: the n-dim spike vector with i-th element equals 1.

"""
function spike(i,n)
    a = zeros(n)
    a[i] = 1
    return a
end

"""
    characteristic(list,n)

CHARACTERISTIC gives the characteristic function in n-dim vector space with values of index in list equal to 1.

# Input Arguments
- `list::Array{Int}`: list of indices.
- `n::Int`: dimension of the target vector.

# Output Argument
- `v::Array{Float}`: the n-dim characteristic vector with values of index in list equal to 1.

"""
function characteristic(list,n)
    v = zeros(n)
    v[list] .= 1.0
    return v
end


"""
    heat_sol(f0,Φ,Σ,t)

HEAT_SOL gives the solution of heat partial differential equation with initial condition u(⋅, 0) = f0

# Input Arguments
- `f0::Array{Float}`: initial condition vector.
- `Φ::Matrix{Float}`: graph Laplacian eigenvectors, served as graph Fourier transform matrix
- `Σ::Array{Int}`: diagonal matrix of eigenvalues.
- `t::Float`: time elapse.

# Output Argument
- `u::Array{Float}`: the solution vector at time t

"""
function heat_sol(f0,Φ,Σ,t)
    u = Φ * (exp.(-t .* Σ) .* Φ' * f0)
    return u
end


"""
    freq_band_matrix(ls,n)

FREQ_BAND_MATRIX provides characteristic diagonal matrix, which is useful for spectral graph filters design.

# Input Arguments
- `ls::Array{Int}`: list of indices.
- `n::Int`: dimension of the target vector.

# Output Argument
- `D::Array{Float}`: the zero/one diagonal matrix.

"""
# using LinearAlgebra
function freq_band_matrix(ls, n)
    f = characteristic(list,n)
    return Diagonal(f)
end


"""
    scatter_gplot(X; marker = nothing, ms = 4)

SCATTER_GPLOT generates a scatter plot figure, which is for quick viewing of a graph signal.
SCATTER_GPLOT!(X; ...) adds a plot to `current` one.

# Input Arguments
- `X::Matrix{Float}`: points locations, can be 2-dim or 3-dim.
- `marker::Array{Float}`: default is nothing. Present different colors given different signal value at each node.
- `ms::Array{Float}`: default is 4. Present different node sizes given different signal value at each node.

"""
function scatter_gplot(X; marker = nothing, ms = 4)
    dim = size(X,2)
    if dim == 2
        scatter(X[:,1],X[:,2],marker_z = marker,ms = ms, c = :viridis, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
    elseif dim == 3
        scatter(X[:,1],X[:,2],X[:,3], marker_z = marker, ms = ms, c = :viridis, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
    else
        print("Dimension Error: scatter_gplot only supports for 2-dim or 3-dim scatter plots.")
    end
end

function scatter_gplot!(X; marker = nothing, ms = 4)
    dim = size(X,2)
    if dim == 2
        scatter!(X[:,1],X[:,2],marker_z = marker,ms = ms, c = :viridis, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
    elseif dim == 3
        scatter!(X[:,1],X[:,2],X[:,3], marker_z = marker, ms = ms, c = :viridis, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
    else
        print("Dimension Error: scatter_gplot! only supports for 2-dim or 3-dim scatter plots.")
    end
end

function cat_plot(X; marker = nothing, ms = 4)
    scatter(X[:,1],X[:,2],X[:,3], marker_z = marker, ms = ms, c = :viridis, legend = false, cbar = true, aspect_ratio = 1, xlims = [-100, 100], ylims = [-100, 100], zlims = [-100, 100])
end

function cat_plot!(X; marker = nothing, ms = 4)
    scatter!(X[:,1],X[:,2],X[:,3], marker_z = marker, ms = ms, c = :viridis, legend = false, cbar = true, aspect_ratio = 1, xlims = [-100, 100], ylims = [-100, 100], zlims = [-100, 100])
end
