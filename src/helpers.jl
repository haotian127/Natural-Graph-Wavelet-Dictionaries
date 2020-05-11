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

function heat_sol(f0,V,D,t)
    u = V * (exp.(-t .* D) .* V' * f0)
    return u
end



function freq_band_matrix(p,n)
    F = zeros(n,n)
    for k = 1:length(p)
        F[p[k],p[k]] = 1
    end
    return F
end


function scatter_gplot(X; marker = nothing, ms = 4)
    scatter(X[:,1],X[:,2],marker_z = marker,ms = ms, c = :viridis, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
end

function scatter_gplot!(X; marker = nothing, ms = 4)
    scatter!(X[:,1],X[:,2],marker_z = marker,ms = ms, c = :viridis, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
end

function cat_plot(X; marker = nothing, ms = 4)
    scatter(X[:,1],X[:,2],X[:,3], marker_z = marker, ms = ms, c = :viridis, legend = false, cbar = true, aspect_ratio = 1, xlims = [-100, 100], ylims = [-100, 100], zlims = [-100, 100])
end

function cat_plot!(X; marker = nothing, ms = 4)
    scatter!(X[:,1],X[:,2],X[:,3], marker_z = marker, ms = ms, c = :viridis, legend = false, cbar = true, aspect_ratio = 1, xlims = [-100, 100], ylims = [-100, 100], zlims = [-100, 100])
end
