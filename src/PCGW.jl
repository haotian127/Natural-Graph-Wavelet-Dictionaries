function Proj(x,A)
    ### project x onto matrix A's column space, given A is full column rank
    y = try
        pinv(A'*A)*(A'*x)
    catch err
        if err != Nothing
            A'*x
        end
    end
    return A*y
end

function wavelet_perp_Matrix(w,A)
    N,k = size(A)
    B = zeros(N,k-1)
    if k == 1
        return
    end
    tmp = A'*w
    if tmp[1] > 1e-6
        M = Matrix{Float64}(I, k-1, k-1)
        return A*vcat((-tmp[2:end] ./ tmp[1])', M)
    end
    if tmp[end] > 1e-6
        M = Matrix{Float64}(I, k-1, k-1)
        return A*vcat(M,(-tmp[1:end-1] ./ tmp[end])')
    end
    ind = findall(tmp .== maximum(tmp))[1]
    rest_ind = setdiff([i for i in 1:k],ind)
    M = Matrix{Float64}(I, k-1, k-1)
    B = A * vcat(vcat(M[1:ind-1,:], (-tmp[rest_ind] ./ tmp[ind])'), M[ind:end,:])
    return B
end

function HTree_Vlist(W)
    #input: weighted adjacency_matrix W
    #output: hierarchical tree of vertices
    ht_vlist = [Vlist_Part(W)]
    while length(ht_vlist[end]) < N
        lvl_vlist = Vlist_Part(W[ht_vlist[end][1],ht_vlist[end][1]]; v_idx = ht_vlist[end][1])
        for i = 2:length(ht_vlist[end])
            lvl_vlist = vcat(lvl_vlist, Vlist_Part(W[ht_vlist[end][i],ht_vlist[end][i]]; v_idx = ht_vlist[end][i]))
        end
        push!(ht_vlist,lvl_vlist)
    end
    return ht_vlist
end


function Vlist_Part(W;v_idx = 1:size(W,1))
    ## partition v_idx into two sets of indicies
    if size(W,1) == 1
        return [v_idx]
    end
    p = partition_fiedler(W)[1]
    vlist1 = findall(p .> 0)
    vlist2 = findall(p .< 0)
    return [v_idx[vlist1],v_idx[vlist2]]
end

function HTree_VElist(V,W)
    #input: hierarchical tree of vertices; eigenvectors V
    #output: hierarchical tree of eigenvectors
    ht_elist = [Elist_Part(V,W)]
    ht_vlist = [Vlist_Part(W)]
    while length(ht_elist[end]) < N
        lvl_elist = Elist_Part(V,W; e_idx = ht_elist[end][1],v_idx = ht_vlist[end][1])
        lvl_vlist = Vlist_Part(W[ht_vlist[end][1],ht_vlist[end][1]]; v_idx = ht_vlist[end][1])
        for i = 2:length(ht_elist[end])
            lvl_elist = vcat(lvl_elist, Elist_Part(V,W; e_idx = ht_elist[end][i],v_idx = ht_vlist[end][i]))
            lvl_vlist = vcat(lvl_vlist, Vlist_Part(W[ht_vlist[end][i],ht_vlist[end][i]]; v_idx = ht_vlist[end][i]))
        end
        push!(ht_elist,lvl_elist)
        push!(ht_vlist,lvl_vlist)
    end
    return ht_vlist, ht_elist
end

function Elist_Part(V,W; e_idx = 1:size(V,2), v_idx = 1:size(V,2))
    ## partition e_idx into two sets of indicies
    if length(v_idx) == 1
        return [e_idx]
    end

    p = partition_fiedler(W[v_idx,v_idx])[1]
    vlist1 = findall(p .> 0)
    # vlist2 = findall(p .< 0)

    # Ve = V[:,e_idx] .^ 2
    # Ve = (W + I) * V[:,e_idx] .^ 2
    Ve = (W + Diagonal(sum(W;dims=2)[:])) * V[:,e_idx] .^ 2
    energyDistr_eigvecs = sum(Ve[vlist1,:],dims = 1)[:]
    elist1 = sort(e_idx[sortperm(energyDistr_eigvecs; rev = true)[1:length(vlist1)]])
    elist2 = setdiff(e_idx,elist1)

    return [elist1,elist2]
end


function HTree_EVlist(V,W_dual)
    #input: hierarchical tree of vertices; eigenvectors V
    #output: hierarchical tree of eigenvectors
    ht_elist = [DElist_Part(W_dual)]
    ht_vlist = [DVlist_Part(V,W_dual)]
    while length(ht_elist[end]) < N
        lvl_elist = DElist_Part(W_dual; e_idx = ht_elist[end][1])
        lvl_vlist = DVlist_Part(V, W_dual; e_idx = ht_elist[end][1], v_idx = ht_vlist[end][1])
        for i = 2:length(ht_elist[end])
            lvl_elist = vcat(lvl_elist, DElist_Part(W_dual; e_idx = ht_elist[end][i]))
            lvl_vlist = vcat(lvl_vlist, DVlist_Part(V, W_dual; e_idx = ht_elist[end][i], v_idx = ht_vlist[end][i]))
        end
        push!(ht_elist,lvl_elist)
        push!(ht_vlist,lvl_vlist)
    end
    return ht_elist, ht_vlist
end

function DVlist_Part(V,W_dual; v_idx = 1:size(V,2), e_idx = 1:size(V,2))
    ## partition v_idx into two sets of indicies
    if length(v_idx) == 1
        return [e_idx]
    end

    p = partition_fiedler(W_dual[e_idx,e_idx])[1]
    elist1 = findall(p .> 0)
    # elist2 = findall(p .< 0)

    Ve = V[v_idx,:] .^ 2
    energyDistr_eigvecs = sum(Ve[:,elist1],dims = 2)[:]
    vlist1 = sort(v_idx[sortperm(energyDistr_eigvecs; rev = true)[1:length(elist1)]])
    vlist2 = setdiff(v_idx,vlist1)

    return [vlist1,vlist2]
end

function DElist_Part(W_dual; e_idx = 1:size(W_dual,1))
    ## partition e_idx into two sets of indicies
    if length(e_idx) == 1
        return [e_idx]
    end
    p = partition_fiedler(W_dual[e_idx,e_idx])[1]
    elist1 = findall(p .> 0)
    elist2 = findall(p .< 0)
    return [e_idx[elist1],e_idx[elist2]]
end




function HTree_coeff_wavelet_packet(f,wavelet_packet)
    ht_coeff = [[f]]
    ht_coeff_abs = [[norm(f,1)]]
    L = length(wavelet_packet)
    for lvl in 2:L
        tmp = [wavelet_packet[lvl][1]'*f]
        tmp_energy = [norm(wavelet_packet[lvl][1]'*f,1)]
        for i in 2:length(wavelet_packet[lvl])
            push!(tmp,wavelet_packet[lvl][i]'*f)
            push!(tmp_energy, norm(wavelet_packet[lvl][i]'*f,1))
        end
        push!(ht_coeff,tmp)
        push!(ht_coeff_abs,tmp_energy)
    end
    return ht_coeff,ht_coeff_abs
end

function HTree_wavelet_packet(V,ht_vlist,ht_elist)
    N = size(V,1)
    wav_packet = [[Matrix{Float64}(I, N, N)]]
    L = length(ht_vlist)
    for lvl in 1:L
        tmp = [const_proj_wavelets(V,ht_vlist[lvl][1],ht_elist[lvl][1])]
        for i in 2:length(ht_vlist[lvl])
            vlist = ht_vlist[lvl][i]
            elist = ht_elist[lvl][i]
            push!(tmp,const_proj_wavelets(V,vlist,elist))
        end
        push!(wav_packet,tmp)
    end
    return wav_packet
end

function HTree_wavelet_packet_varimax(V,ht_elist)
    N = size(V,1)
    wav_packet = [[Matrix{Float64}(I, N, N)]]
    L = length(ht_elist)
    for lvl in 1:L
        tmp = [varimax(V[:,ht_elist[lvl][1]])]
        for i in 2:length(ht_elist[lvl])
            elist = ht_elist[lvl][i]
            push!(tmp,varimax(V[:,elist]))
        end
        push!(wav_packet,tmp)
    end
    return wav_packet
end

function const_proj_wavelets(V,vlist,elist; method = "Modified Gram-Schmidt with Lp Pivoting")
    if length(vlist) == 1
        return V[:,elist]
    end
    N = size(V,1)
    m = length(vlist)
    Wav = zeros(N,m)

    B = V[:,elist]

    if method == "Iterative-Projection"
        for k in 1:length(vlist)
            wavelet = Proj(spike(vlist[k],N),B)
            Wav[:,k] .= wavelet ./ norm(wavelet)
            B = wavelet_perp_Matrix(wavelet,B)
        end
    elseif method == "Gram-Schmidt"
        P = B * B'
        for k in 1:length(vlist)
            wavelet = P * spike(vlist[k], N)
            Wav[:,k] .= wavelet ./ norm(wavelet)
        end
        Wav, complement_dim = gram_schmidt(Wav)
        if complement_dim != 0
            complement_space = B * nullspace(Wav' * B)
            Wav = hcat(Wav, complement_space)
        end
    elseif method == "Modified Gram-Schmidt with Lp Pivoting"
        P = B * B'
        for k in 1:length(vlist)
            wavelet = P * spike(vlist[k], N)
            Wav[:,k] .= wavelet ./ norm(wavelet)
        end
        Wav, complement_dim = modified_gram_schmidt_lp_pivoting(Wav)
        if complement_dim != 0
            complement_space = B * nullspace(Wav' * B)
            Wav = hcat(Wav, complement_space)
        end
    end

    return Wav
end





function assemble_wavelet_basis(dvec,wavelet_packet)
    W = wavelet_packet[dvec[1][1]][dvec[1][2]]
    if length(dvec) < 2
        return W
    end
    for i in 2:length(dvec)
        W = hcat(W,wavelet_packet[dvec[i][1]][dvec[i][2]])
    end
    return W
end

function assemble_wavelet_basis_at_certain_layer(wavelet_packet, ht_vlist; layer = 1)
    W = zeros(size(wavelet_packet[1][1]))
    if layer < 2
        return wavelet_packet[1][1]
    end
    for i in 2:length(wavelet_packet[layer])
        W[:,ht_vlist[layer-1][i]] .= wavelet_packet[layer][i]
    end
    return W
end



function best_basis_algorithm(ht_vlist,parent,ht_coeff_L1)
    Lvl = length(ht_vlist)
    dvec = [[Lvl+1, k] for k = 1:length(ht_vlist[Lvl])]
    for i in 1:length(parent[Lvl])
        pair = parent[Lvl][i]
        if length(pair) == 1
            ind = findall(x -> x == [Lvl+1,pair[1]],dvec)[:]
            deleteat!(dvec,ind)
            push!(dvec,[Lvl, i])
        else
            if sum(ht_coeff_L1[Lvl+1][pair]) > ht_coeff_L1[Lvl][i]
                ind1 = findall(x -> x == [Lvl+1,pair[1]],dvec)[:]
                ind2 = findall(x -> x == [Lvl+1,pair[2]],dvec)[:]
                deleteat!(dvec,union(ind1,ind2))
                push!(dvec,[Lvl, i])
            else
                continue
            end
        end
    end

    for lvl = Lvl-1:-1:1
        for i in 1:length(parent[lvl])
            pair = parent[lvl][i]
            if length(pair) == 1
                ind = findall(x -> x == [lvl+1,pair[1]],dvec)[:]
                deleteat!(dvec,ind)
                push!(dvec,[lvl, i])
            else
                if ([lvl+1,pair[1]] in dvec) && ([lvl+1,pair[2]] in dvec)
                    if sum(ht_coeff_L1[lvl+1][pair]) > ht_coeff_L1[lvl][i]
                        ind1 = findall(x -> x == [lvl+1,pair[1]],dvec)[:]
                        ind2 = findall(x -> x == [lvl+1,pair[2]],dvec)[:]
                        deleteat!(dvec,union(ind1,ind2))
                        push!(dvec,[lvl, i])
                    else
                        continue
                    end
                else
                    continue
                end
            end
        end

    end

    return dvec
end




function HTree_findParent(ht_vlist)
    #input: hierarchical tree vertex list
    #output: find parent
    L = length(ht_vlist)
    parent = [[[1,2]]]

    for lvl in 1:L-1
        lvl_parent = Array{Int64,1}[]
        ind = 1
        for i = 1:length(ht_vlist[lvl])
            if length(ht_vlist[lvl][i]) == 1
                push!(lvl_parent, [ind])
                ind += 1
            else
                push!(lvl_parent, [ind, ind+1])
                ind += 2
            end
        end
        push!(parent,lvl_parent)
    end
    return parent
end





function HTree_coeff2mat(ht_coeff,N)
    L = length(ht_coeff)
    C = zeros(N,L)
    C[:,1] = ht_coeff[1][1]
    for lvl in 2:L
        tmp = ht_coeff[lvl][1]
        for i in 2:length(ht_coeff[lvl])
            tmp = vcat(tmp, ht_coeff[lvl][i])
        end
        C[:,lvl] = tmp
    end
    return C'
end



function HTree_wavelet_packet_unorthogonalized(V,ht_vlist,ht_elist)
    N = size(V,1)
    wav_packet = [[Matrix{Float64}(I, N, N)]]
    L = length(ht_vlist)
    for lvl in 1:L
        tmp = [const_proj_wavelets_unorthogonalized(V,ht_vlist[lvl][1],ht_elist[lvl][1])]
        for i in 2:length(ht_vlist[lvl])
            vlist = ht_vlist[lvl][i]
            elist = ht_elist[lvl][i]
            push!(tmp,const_proj_wavelets_unorthogonalized(V,vlist,elist))
        end
        push!(wav_packet,tmp)
    end
    return wav_packet
end

function const_proj_wavelets_unorthogonalized(V,vlist,elist)
    if length(vlist) == 1
        return V[:,elist]
    end
    N = size(V,1)
    m = length(vlist)
    Wav = zeros(N,m)

    B = V[:,elist]
    P = B*B'
    for k in 1:length(vlist)
        wavelet = P*spike(vlist[k],N)
        Wav[:,k] .= wavelet ./ norm(wavelet)
    end

    return Wav
end


# Gram-Schmidt Process Orthogonalization
function gram_schmidt(A; tol = 1e-10)
    # Input: matirx A
    # Output: orthogonalization matrix of A's column vectors

    # Convert the matrix to a list of column vectors
    a = [A[:,i] for i in 1:size(A,2)]

    # Start Gram-Schmidt process
    q = []
    complement_dim = 0
    for i = 1:length(a)
        qtilde = a[i]
        for j = 1:i-1
            qtilde -= (q[j]'*a[i]) * q[j]
        end
        if norm(qtilde) < tol
            complement_dim = size(A,2) - i + 1
            break
        end
        push!(q, qtilde/norm(qtilde))
    end
    Q = zeros(size(A,1), length(q))
    for i = 1:length(q)
        Q[:,i] .= q[i]
    end
    return Q, complement_dim
end


# Modified Gram-Schmidt Process Orthogonalization
function modified_gram_schmidt_lp_pivoting(A; tol = 1e-10, p = 1)
    # Input: matirx A
    # Output: orthogonalization matrix of A's column vectors based on L^p pivoting

    n = size(A,2)
    # Convert the matrix to a list of column vectors
    a = [A[:,i] for i in 1:n]

    # Start modified Gram-Schmidt process
    q = []
    v = copy(a)
    vec_lp_norm = [norm(v[i], p) for i in 1:n]
    complement_dim = 0
    for i = 1:n
        # Pivoting based on minimum lp-norm
        idx = findmin(vec_lp_norm)[2]
        v[i], v[idx+i-1] = v[idx+i-1], v[i]
        # Check the linear dependency
        if norm(v[i]) < tol
            complement_dim = n - i + 1
            break
        end
        qtilde = v[i]/norm(v[i], 2)
        vec_lp_norm = []
        for j = i+1:n
            v[j] .-= (qtilde'*v[j]) * qtilde
            push!(vec_lp_norm, norm(v[j], p))
        end
        push!(q, qtilde)
    end
    Q = zeros(size(A,1), length(q))
    for i = 1:length(q)
        Q[:,i] .= q[i]
    end
    return Q, complement_dim
end
