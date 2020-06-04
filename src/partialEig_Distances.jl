"""
    partialEig_Distance(graphClusters, activeEigenVecs, lamb, 𝛷, Q; eigen_metric = :DAG)

partialEig\\_Distance provides the partial node non-trivial eigenvector metric results for each cluster
"""
function partialEig_Distance(graphClusters, activeEigenVecs, lamb, 𝛷, Q, L; eigen_metric = :DAG, edge_weight = 1, edge_length = 1, α = 1.0, m = "Inf", dt = 0.1, tol = 1e-5)
    M = length(graphClusters)
    partial_dist_ls = Array{Array{Float64,2},1}()
    for k in 1:M
        J = length(activeEigenVecs[k])
        restrict_𝛷 = zeros(N,J); restrict_𝛷[graphClusters[k], :] = 𝛷[graphClusters[k], activeEigenVecs[k]]
        tmp_dist = zeros(N,N); for i in 1:N, j in 1:N; if i != j; tmp_dist[i,j] = Inf; end; end
        if eigen_metric == :DAG
            tmp_dist[activeEigenVecs[k],activeEigenVecs[k]] = eigDAG_Distance(restrict_𝛷, Q, J; edge_weight = edge_weight)
        elseif eigen_metric == :HAD
            tmp_dist[activeEigenVecs[k],activeEigenVecs[k]] = eigHAD_Distance(𝛷, lamb; indexEigs = activeEigenVecs[k]) # 𝛷 instead of restrict_𝛷 !
        elseif eigen_metric == :ROT
            P = restrict_𝛷 .^ 2
            P ./= sum(P; dims = 1)
            tmp_dist[activeEigenVecs[k],activeEigenVecs[k]] = eigROT_Distance(P, Q; edge_length = edge_length, α = α)
        elseif eigen_metric == :TSD
            tmp_dist[activeEigenVecs[k],activeEigenVecs[k]] = eigTSD_Distance(restrict_𝛷, 𝛷, lamb, Q, L; m = m, dt = dt, tol = tol)
        else
            print("Error: do not support such non-trivial eigenvector metric!")
        end
        push!(partial_dist_ls, tmp_dist)
    end
    return partial_dist_ls
end
