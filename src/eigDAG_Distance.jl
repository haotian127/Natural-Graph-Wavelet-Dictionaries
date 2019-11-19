function eigDAG_Distance(V,Q,numEigs; edge_weight = 1)
#input: V, matrix eigenvectors; Q, incidence_matrix of the graph; numEigs, number of eigenvectors.
    if edge_weight == 1
        dis = zeros(numEigs,numEigs)
        Ve = abs.(Q' * V)
        for i = 1:numEigs, j = i+1:numEigs
            dis[i,j] = norm(Ve[:,i]-Ve[:,j],2)
        end
    else
        dis = zeros(numEigs,numEigs)
        Ve = abs.(Q' * V)
        for i = 1:numEigs, j = i+1:numEigs
            dis[i,j] = sqrt(sum((Ve[:,i]-Ve[:,j]).^2 .* sqrt.(edge_weight)))
        end
    end

    return dis + dis'
end
