# varimax Julia version
# implemented by Haotian Li


function varimax(A; gamma = 1.0, minit = 50, maxit = 1000, reltol = 1e-14)
	# gamma = 0, 1, m/2, and d(m - 1)/(d + m - 2), corresponding to quartimax, varimax, equamax, and parsimax.

	# Get the sizes of input matrix
    d,m = size(A)

	# If there is only one vector, then do nothing.
	if m == 1
		return A
	end


	# Warm up step: start with better orthogonal matrix T
    T = Matrix{Float64}(I, m, m)
    B = A * T

    L,_,M = svd(A' * (d*B.^3 - gamma*B * Diagonal(sum(B.^2, dims = 1)[:])))
    T = L * M'
	if norm(T-Matrix{Float64}(I, m, m)) < reltol
        T,_ = qr(randn(m,m)).Q
        B = A * T
    end

	# Iteration step: get better T in order to maximize the objective (as described in Factor Analysis book)
    D = 0
    for k in 1:maxit
        Dold = D
        L,s,M = svd(A' * (d*B.^3 - gamma*B * Diagonal(sum(B.^2, dims = 1)[:])))
        T = L * M'
        D = sum(s)
		B = A * T
        if (abs(D - Dold)/D < reltol) && k >= minit
            break
        end
    end

	# Adjust the sign of each rotated vector such that the maximum absolute value is positive.
	for i in 1:m
	   if abs(maximum(B[:,i])) < abs(minimum(B[:,i]))
	       B[:,i] .= - B[:,i]
	   end
	end
	return B
end
