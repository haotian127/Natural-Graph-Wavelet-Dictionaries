"""
    eigTSDM_Distance(V,Ve,lambda,Q,L;m = "Inf",dt = 0.1,tol = 1e-5)

EIGTSDM\\_DISTANCE computes the TSDM distance matrix of Ve's column vectors on a graph.

# Input Argument
- `V::Matrix{Float64}`: matrix of graph Laplacian eigenvectors.
- `Ve::Matrix{Float64}`: feature matrix of eigenvectors, e.g., V.^2 or exp.(V) .÷ sum(exp.(V), dims=1).
- `lambda::Matrix{Float64}`: vector of eigenvalues.
- `Q::Matrix{Float64}`: oriented incidence matrix.
- `L::Matrix{Float64}`: Laplacian matrix.
- `m*dt::Matrix{Float64}`: stopping time.
- `tol::Float64`: toleration of convergence.

# Output Argument
- `dis::Matrix{Float64}`: distance matrix, d\\_TSDM.
"""

function eigTSDM_Distance(V,Ve,lambda,Q,L;m = "Inf",dt = 0.1,tol = 1e-5)
    n = size(Ve)[1]
    dis = zeros(n,n)
    if m == "Inf"
        for i = 1:n, j = i+1:n
            cost = 0
            f0 = Ve[:,i] - Ve[:,j]
            f = f0
            c = V'*f0
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
            f0 = Ve[:,i] - Ve[:,j]
            f = f0
            c = V'*f0
            for k = 1:m
                cost = cost + dt * norm(Q' * f,1)
                f = u_sol(c,V,lambda,k,dt)
            end
            dis[i,j] = cost
        end
    end
    return dis + dis'
end

function u_sol(c,V,D,k,dt)
    t = k * dt
    u = V * (exp.(-t .* D) .* c)
    return u
end
