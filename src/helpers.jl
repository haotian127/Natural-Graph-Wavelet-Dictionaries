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
- `a::Array{Float64}`: the n-dim spike vector with i-th element equals 1.

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
- `v::Array{Float64}`: the n-dim characteristic vector with values of index in list equal to 1.

"""
function characteristic(list,n)
    v = zeros(n)
    v[list] .= 1.0
    return v
end


"""
    heat_sol(f0,Φ,Σ,t)

HEAT\\_SOL gives the solution of heat partial differential equation with initial condition u(⋅, 0) = f0

# Input Arguments
- `f0::Array{Float64}`: initial condition vector.
- `Φ::Matrix{Float64}`: graph Laplacian eigenvectors, served as graph Fourier transform matrix
- `Σ::Array{Int}`: diagonal matrix of eigenvalues.
- `t::Float`: time elapse.

# Output Argument
- `u::Array{Float64}`: the solution vector at time t

"""
function heat_sol(f0,Φ,Σ,t)
    u = Φ * (exp.(-t .* Σ) .* Φ' * f0)
    return u
end


"""
    freq_band_matrix(ls,n)

FREQ\\_BAND\\_MATRIX provides characteristic diagonal matrix, which is useful for spectral graph filters design.

# Input Arguments
- `ls::Array{Int}`: list of indices.
- `n::Int`: dimension of the target vector.

# Output Argument
- `D::Array{Float64}`: the zero/one diagonal matrix.

"""
# using LinearAlgebra
function freq_band_matrix(ls, n)
    f = characteristic(list,n)
    return Diagonal(f)
end


"""
    scatter_gplot(X; marker = nothing, ms = 4)

SCATTER\\_GPLOT generates a scatter plot figure, which is for quick viewing of a graph signal.
SCATTER\\_GPLOT!(X; ...) adds a plot to `current` one.

# Input Arguments
- `X::Matrix{Float64}`: points locations, can be 2-dim or 3-dim.
- `marker::Array{Float64}`: default is nothing. Present different colors given different signal value at each node.
- `ms::Array{Float64}`: default is 4. Present different node sizes given different signal value at each node.

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



"""
    cat_plot(X; marker = nothing, ms = 4)

CAT\\_PLOT generates a scatter plot figure for cat example, which is for quick viewing of a graph signal within a specific range (i.e., xlims, ylims, zlims).
CAT\\_PLOT!(X; ...) adds a plot to `current` one.

# Input Arguments
- `X::Matrix{Float64}`: 3-dim points.
- `marker::Array{Float64}`: default is nothing. Present different colors given different signal value at each node.
- `ms::Array{Float64}`: default is 4. Present different node sizes given different signal value at each node.

"""
function cat_plot(X; marker = nothing, ms = 4)
    scatter(X[:,1],X[:,2],X[:,3], marker_z = marker, ms = ms, c = :viridis, legend = false, cbar = true, aspect_ratio = 1, xlims = [-100, 100], ylims = [-100, 100], zlims = [-100, 100])
end

function cat_plot!(X; marker = nothing, ms = 4)
    scatter!(X[:,1],X[:,2],X[:,3], marker_z = marker, ms = ms, c = :viridis, legend = false, cbar = true, aspect_ratio = 1, xlims = [-100, 100], ylims = [-100, 100], zlims = [-100, 100])
end


"""
    approx_error_plot(ortho_mx_list, f; fraction_cap = 0.3, label = false, doSave = false, path = "")

APPROX\\_ERROR\\_PLOT draw approx. error figure w.r.t. fraction of kept coefficients

# Input Arguments
- `ortho_mx_list::Array{Matrix{Float64}}`: a list of orthonormal matrices.
- `f::Array{Float64}`: target graph signal for approximation.
- `fraction_cap::Float`: default is 0.3. The capital of fration of kept coefficients.

"""
function approx_error_plot(ortho_mx_list, f; fraction_cap = 0.3, label = false, doSave = false, path = "")
    N = length(f)
    L = length(ortho_mx_list)
    err = [[1.0] for _ in 1:L]
    coeff = [mx'*f for mx in ortho_mx_list]

    for frac = 0.01:0.01:fraction_cap
        numKept = Int(ceil(frac * N))
        for l in 1:L
            ind = sortperm(coeff[l].^2, rev = true)[numKept+1:end]
            push!(err[l], norm(coeff[l][ind])/norm(f))
        end
    end

    gr(dpi = 300)
    fraction = 0:0.01:fraction_cap
    plt = plot(fraction, err, yaxis=:log, lab = label, linewidth = 3)
    if doSave
        savefig(plt, path)
        return "figure saved! @ " * path
    end
    return "use current() to show figure."
end
