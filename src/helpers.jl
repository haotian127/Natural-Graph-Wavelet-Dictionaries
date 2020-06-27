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
function scatter_gplot(X; marker = nothing, ms = 4, smallValFirst = true, c = :viridis)
    dim = size(X,2)
    if marker != nothing && smallValFirst
        idx = sortperm(marker)
        X = X[idx,:]
        marker = marker[idx]
    end
    if dim == 2
        scatter(X[:,1],X[:,2],marker_z = marker,ms = ms, c = c, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
    elseif dim == 3
        scatter(X[:,1],X[:,2],X[:,3], marker_z = marker, ms = ms, c = c, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
    else
        print("Dimension Error: scatter_gplot only supports for 2-dim or 3-dim scatter plots.")
    end
end

function scatter_gplot!(X; marker = nothing, ms = 4, smallValFirst = true, c = :viridis)
    dim = size(X,2)
    if marker != nothing && smallValFirst
        idx = sortperm(marker)
        X = X[idx,:]
        marker = marker[idx]
    end
    if dim == 2
        scatter!(X[:,1],X[:,2],marker_z = marker,ms = ms, c = c, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
    elseif dim == 3
        scatter!(X[:,1],X[:,2],X[:,3], marker_z = marker, ms = ms, c = c, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
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
    approx_error_plot(ortho_mx_list, f; fraction_cap = 0.3, label = false, Save = false, path = "")

APPROX\\_ERROR\\_PLOT draw approx. error figure w.r.t. fraction of kept coefficients

# Input Arguments
- `ortho_mx_list::Array{Matrix{Float64}}`: a list of orthonormal matrices.
- `f::Array{Float64}`: target graph signal for approximation.
- `fraction_cap::Float`: default is 0.3. The capital of fration of kept coefficients.

"""
function approx_error_plot(ortho_mx_list, f; fraction_cap = 0.3, label = false, Save = false, path = "")
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
    plt = plot(fraction, err, yscale=:log10, lab = label, linewidth = 3, xaxis = "Fraction of Coefficients Retained", yaxis = "Relative Approximation Error")
    if Save
        savefig(plt, path)
        return "figure saved! @ " * path
    end
    return "use current() to show figure."
end

################################################################################
####################### Approximation error plot################################
################################################################################
### function to plot the approximation error curve
function approx_error_plot2(DVEC::Array{Array{Float64,1},1})
    gr(dpi = 400)
    plot(xaxis = "Fraction of Coefficients Retained", yaxis = "Relative Approximation Error")
    frac = 0.3
    T = ["Haar", "Walsh", "Laplacian", "GHWT_c2f", "GHWT_f2c", "eGHWT", "PC_NGW", "varimax_NGW"]
    L = [(:dashdot,:orange), (:dashdot,:pink), (:dashdot, :red), (:solid, :gray), (:solid, :green), (:solid, :blue), (:solid, :purple), (:solid, :black)]
    LW = [1, 1, 1, 1, 1, 2, 2, 2]
    for i = 1:length(DVEC)
        dvec = DVEC[i]
        N = length(dvec)
        dvec_norm = norm(dvec,2)
        dvec_sort = sort(dvec.^2) # the smallest first
        er = sqrt.(reverse(cumsum(dvec_sort)))/dvec_norm # this is the relative L^2 error of the whole thing, i.e., its length is N
        p = Int64(floor(frac*N)) + 1 # upper limit
        plot!(frac*(0:(p-1))/(p-1), er[1:p], yaxis=:log, xlims = (0.,frac), label = T[i], line = L[i], linewidth = LW[i])
    end
end


"""
    sortWaveletsByCenteredLocations(Wav)

sort wavelets by centered locations

# Input Argument
- `Wav::Matrix{Float64}`: a matrix whose columns are wavelet vectors.

# Output Argument
- `Wav::Matrix{Float64}`: the sorted matrix.
"""
function sortWaveletsByCenteredLocations(Wav)
    ord = findmax(abs.(Wav), dims = 1)[2][:]
    idx = sortperm([i[1] for i in ord])
    return Wav[:,idx]
end

using Clustering
"""
    spectral_clustering(𝛷, M)

SPECTRAL_CLUSTERING return M graph clusters, i.e., {Vₖ| k = 1,2,...,M}.

# Input Argument
- `𝛷::Matrix{Float64}`: the matrix of graph Laplacian eigenvectors.
- `M::Int64`: the number of graph clusters.

# Output Argument
- `clusters::Array{Array{Int64}}`: graph cluster indices.

"""
function spectral_clustering(𝛷, M)
    if M < 2
        return [1:size(𝛷,1)]
    end
    cluster_indices = assignments(kmeans(𝛷[:,2:M]', M))
    clusters = Array{Array{Int64,1},1}()
    for k in 1:M
        push!(clusters, findall(cluster_indices .== k)[:])
    end
    return clusters
end

"""
    transform2D(X; s = 1, t = [0,0])

TRANSFORM2D dilate each point of `X` by scale s and translate by 2D vector t.
"""
function transform2D(X; s = 1, t = [0,0])
    X1 = X .* s
    X2 = zeros(size(X))
    for i in 1:size(X,1)
        X2[i,1] = X1[i,1] + t[1]
        X2[i,2] = X1[i,2] + t[2]
    end
    return X2
end

"""
    NN_rendering(X, Img_Mat)

NN\\_RENDERING generates a rendering signal at each point of `X` from the image `Img_Mat` by nearest neighbor method.
"""
function NN_rendering(X, Img_Mat)
    N = size(X,1)
    f = zeros(N)
    for i in 1:N
        nn_x, nn_y = Int(round(X[i, 2])), Int(round(X[i, 1]))
        if nn_x < 1 || nn_x > size(Img_Mat, 2) || nn_y < 1 || nn_y > size(Img_Mat, 1)
            print("Error: pixel out of boundary!")
            return
        end
        f[i] = Img_Mat[nn_x, nn_y]
    end
    return f
end

"""
    Bilinear_rendering(X, Img_Mat)

NN\\_RENDERING generates a rendering signal at each point of `X` from the image `Img_Mat` by bilinear interpolation method.
"""
function Bilinear_rendering(X, Img_Mat)
    N = size(X,1)
    f = zeros(N)
    for i in 1:N
        x1, x2, y1, y2 = Int(floor(X[i, 2])), Int(floor(X[i, 2])) + 1, Int(floor(X[i, 1])), Int(floor(X[i, 1])) + 1
        x, y = X[i,2], X[i,1]
        F = [Img_Mat[x1,y1] Img_Mat[x1,y2]
            Img_Mat[x2,y1] Img_Mat[x2,y2]]
        prod_res = 1/((x2 - x1) * (y2 - y1)) * [x2-x x-x1] * F * [y2-y y-y1]'
        f[i] = prod_res[1,1]
    end
    return f
end
