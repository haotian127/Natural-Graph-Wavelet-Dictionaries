# varimax Julia version
# implemented by Haotian Li


function varimax(A; gamma = 1.0, maxit = 1000, reltol = 1e-14)
    d,m = size(A);

	if m == 1
		return A
	end
	
    T = Matrix{Float64}(I, m, m);
    B = A * T;

    L,_,M = svd(A' * (d*B.^3 - gamma*B * Diagonal(sum(B.^2, dims = 1)[:])))
    T = L * M'
	if norm(T-Matrix{Float64}(I, m, m)) < reltol
        T,_ = qr(randn(m,m)).Q;
        B = A * T;
    end

    D = 0;
    for k in 1:maxit
        Dold = D;
        L,s,M = svd(A' * (d*B.^3 - gamma*B * Diagonal(sum(B.^2, dims = 1)[:])))
        T = L * M';
        D = sum(s)
		B = A * T;
        if abs(D - Dold)/D < reltol
            break
        end
    end
   for i in 1:m
       if abs(maximum(B[:,i])) < abs(minimum(B[:,i]))
           B[:,i] .= - B[:,i]
       end
   end
    return B
end
