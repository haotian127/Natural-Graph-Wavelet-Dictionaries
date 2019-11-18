## helper functions
function findminimum(v, n)
    idx = sortperm(v)[1:n]
    return idx
end

function freq_band_matrix(p,n)
    F = zeros(n,n)
    for k = 1:length(p)
        F[p[k],p[k]] = 1
    end
    return F
end

function spike(i,n)
    a = zeros(n)
    a[i] = 1
    return a
end

function heat_sol(f0,V,D,t)
    u = V * (exp.(-t .* D) .* V' * f0)
    return u
end

function characteristic(list,n)
    v = zeros(n)
    v[list] .= 1
    return v
end

function scatter_gplot(X;marker = false, ms = 4)
    scatter(X[:,1],X[:,2],marker_z = marker,ms = ms, c = :viridis, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
end

function scatter_gplot!(X;marker = false, ms = 4)
    scatter!(X[:,1],X[:,2],marker_z = marker,ms = ms, c = :viridis, legend = false, mswidth = 0, cbar = true, aspect_ratio = 1)
end
