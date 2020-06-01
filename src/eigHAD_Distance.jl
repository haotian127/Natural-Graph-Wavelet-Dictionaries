using Optim

function eigHAD_Distance(V,D,numEigs)
    body
end

function eigHAD_Affinity(V,D,numEigs)
#input: V, matrix eigenvectors; D, vector of eigenvalues; numEigs, number of eigenvectors.

N = size(V)[1]
dis = zeros(numEigs,numEigs)

for i = 2:numEigs
    measure = Diagonal(V[:,i])*V[:,i:n]
    measure = sqrt.(sum(measure.^2, dims = 1))[:]
    index = findall(measure .> .01/sqrt(N))
    index = index .+ (i-1)

    #display([i,length(index)/length(measure)])

    for j in index

        lambda = D[i]
        mu = D[j]

        x0 = 1 ./ (max(lambda,mu))

        #minimizer t
        result = optimize(t -> abs(exp(-t[1]*lambda) + exp(-t[1]*mu) - 1), [x0], BFGS());
        t = Optim.minimizer(result)[1]
        #print("t = ", t,"; ")

        hadamardProd = V[:,i] .* V[:,j]

        heatEvolution = V * Diagonal(exp.(-t .* D)) * V' * hadamardProd
        dis[i,j] = norm(heatEvolution,2)
        #/ (norm(hadamardProd,2) + 1e-6)
    end
end

dis = dis + dis'
dis[1,1] = maximum(dis)

return dis
end
