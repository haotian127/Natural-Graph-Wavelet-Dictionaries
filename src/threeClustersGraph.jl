using LightGraphs, SimpleWeightedGraphs, Random, Distances
"""
    threeClustersGraph()

THREECLUSTERSGRAPH construct a simple weighted threeClusters graph with 360 vertices. Edge weights are the Gaussian w.r.t Euclidean distances.

# Output Argument
- `G::SimpleWeightedGraph{Int64,Float64}`: a simple weighted graph of the threeClusters.
- `L::Matrix{Float64}`: the weighted symmetric graph Laplacian matrix.
- `X::Matrix{Float64}`: a matrix whose i-th row represent the 2D coordinates of the i-th node.

"""
function threeClustersGraph()
    Random.seed!(2019)
    N = 300; ecc = .05; delta = 1.6*ecc; frac = 1/10

    r1 = sqrt.(rand(Int(N*frac),1)); theta1 = 2*pi*rand(Int(N*frac),1);
    data1 = [r1.*cos.(theta1) r1.*sin.(theta1)]; data1 = data1*[ecc 0;0 ecc] .+ [-4*delta 2*delta];

    r2 = sqrt.(rand(N,1)); theta2 = 2*pi*rand(N,1);
    data2 = [r2.*cos.(theta2) r2.*sin.(theta2)]; data2 = data2*[1 0;0 ecc] .+ [0 0]

    r3 = sqrt.(rand(Int(N*frac),1)); theta3 = 2*pi*rand(Int(N*frac),1);
    data3 = [r3.*cos.(theta3) r3.*sin.(theta3)]; data3 = data3*[ecc 0;0 ecc] .+ [4*delta 2*delta]

    ind = sortperm(data1[:,1]); data1 = data1[ind,:];
    ind = sortperm(data2[:,1]); data2 = data2[ind,:];
    ind = sortperm(data3[:,1]); data3 = data3[ind,:];
    X = [data1; data2; data3];

    N = size(X,1);
    A = exp.( - pairwise(Euclidean(),X, dims = 1).^2 ./ (delta/2)^2);
    K = Diagonal(sum(A,dims = 2)[:].^(-1/2)) * A * Diagonal(sum(A,dims = 2)[:].^(-1/2)); K[findall(K .< 0.0001)].=0;
    L = Diagonal(sum(K,dims = 2)[:]) - K

    G = SimpleWeightedGraph(N)
    for i = 1:N-1
        for j = i+1:N
            if K[i,j] > 0
                G.weights[i,j] = K[i,j]
                G.weights[j,i] = K[j,i]
            end
        end
    end
    return G, L, X
end
