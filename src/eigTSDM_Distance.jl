"""
    eigTSDM_Distance(Ve,V,lambda,Q,L;m = "Inf",dt = 0.1,tol = 1e-5)

EIGTSDM\\_DISTANCE computes the TSDM distance matrix of Ve's column vectors on a graph.

# Input Argument
- `Ve::Matrix{Float64}`: feature matrix of eigenvectors, e.g., V.^2 or exp.(V)./sum(exp.(V), dims=1).
- `V::Matrix{Float64}`: matrix of graph Laplacian eigenvectors.
- `lambda::Array{Float64}`: vector of eigenvalues.
- `Q::Matrix{Float64}`: the oriented incidence matrix of the graph.
- `L::Matrix{Float64}`: the graph Laplacian matrix.
- `m*dt::Float64`: default is T = ∞, the stopping time T = m⋅dt in TSDM.
- `tol::Float64`: tolerance for convergence.

# Output Argument
- `dis::Matrix{Float64}`: distance matrix, d(φᵢ,φⱼ;T).

"""
function eigTSDM_Distance(Ve,V,lambda,Q,L;m = "Inf",dt = 0.1,tol = 1e-5)
    n = size(Ve,1)
    dis = zeros(n,n)
    if m == "Inf"
        for i = 1:n, j = i+1:n
            cost = 0
            f₀ = Ve[:,i] - Ve[:,j]
            f = f₀
            c = V'*f₀
            global ind
            ind = 0
            while(norm(L*f,1)>tol)
                ind += 1
                cost += dt * norm(Q' * f,1)
                f = u_sol(c,V,lambda,ind,dt)
            end
            dis[i,j] = cost
        end
    else
        for i = 1:n, j = i+1:n
            cost = 0
            f₀ = Ve[:,i] - Ve[:,j]
            f = f₀
            c = V'*f₀
            for k = 1:m
                cost = cost + dt * norm(Q' * f,1)
                f = u_sol(c,V,lambda,k,dt)
            end
            dis[i,j] = cost
        end
    end
    return dis + dis'
end

"""
    TSD_Distance(p, q, V, lambda, Q, L; m = "Inf", dt = 0.1, tol = 1e-5)

TSD\\_DISTANCE computes the TSD distance between two vector meassures p and q on a graph.

# Input Argument
- `p::Array{Float64}`: the source vector measure.
- `q::Array{Float64}`: the destination vector measure.
- `V::Matrix{Float64}`: matrix of graph Laplacian eigenvectors.
- `lambda::Array{Float64}`: vector of eigenvalues.
- `Q::Matrix{Float64}`: the oriented incidence matrix of the graph.
- `L::Matrix{Float64}`: the graph Laplacian matrix.
- `m*dt::Float64`: default is T = ∞, the stopping time T = m⋅dt in TSDM.
- `tol::Float64`: tolerance for convergence.

# Output Argument
- `dist::Float64`: TSD distance d\\_TSD(p,q;T).

"""
function TSD_Distance(p,q,V,lambda,Q,L;m = "Inf",dt = 0.1,tol = 1e-5)
    cost = 0
    f₀ = q - p
    f = f₀
    c = V'*f₀
    if m == "Inf"
        global ind
        ind = 0
        while(norm(L*f,1)>tol)
            ind += 1
            cost += dt * norm(Q' * f,1)
            f = u_sol(c,V,lambda,ind,dt)
        end    
    else
        for k = 1:m
            cost = cost + dt * norm(Q' * f,1)
            f = u_sol(c,V,lambda,k,dt)
        end
    end
    dist = cost
    return dist
end

function u_sol(c,V,D,k,dt)
    t = k * dt
    u = V * (exp.(-t .* D) .* c)
    return u
end
