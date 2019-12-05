
function best_basis_algorithm2(ht_vlist,Parent,ht_coeff_L1)
    Lvl = length(ht_vlist)
    dvec = [[Lvl+1, k] for k = 1:length(ht_vlist[Lvl])]
    Subspaces = [[tmp for tmp in pair] for pair in Parent[Lvl]]
    dvec_copy = copy(dvec)
    Subspaces_copy = copy(Subspaces)
    tmp_count = 0
    tmp_loc = []
    for i in 1:length(Subspaces)
        global tmp_count
        pair = Subspaces[i]
        if length(pair) == 1
            ind = findall(x -> x == dvec[pair[1]],dvec_copy)[1]
            deleteat!(dvec_copy,ind)
            insert!(dvec_copy,ind, [Lvl, i])
            Subspaces_copy[i] = [ind]
        else
            if compute_subspace_cost(dvec,pair,ht_coeff_L1) > ht_coeff_L1[Lvl][i]
                ind = findall(x -> x == dvec[pair[1]],dvec_copy)[1]
                deleteat!(dvec_copy,ind:ind+length(pair)-1)
                insert!(dvec_copy,ind,[Lvl, i])
                tmp_count -= length(pair) - 1
                push!(tmp_loc, i)
                Subspaces_copy[i] = [ind]
            else
                Subspaces_copy[i] = [k + tmp_count for k in Subspaces[i]]
            end
        end
    end
    dvec = dvec_copy
    Subspaces = [union_array_of_arrays(Subspaces_copy[pair]) for pair in Parent[Lvl-1]]

    for lvl = Lvl-1:-1:1
        global dvec, Subspaces
        dvec_copy = copy(dvec)
        Subspaces_copy = copy(Subspaces)
        tmp_count = 0
        tmp_loc = []
        for i in 1:length(Subspaces)
            pair = Subspaces[i]
            if length(pair) == 1
                ind = findall(x -> x == dvec[pair[1]],dvec_copy)[1]
                deleteat!(dvec_copy,ind)
                insert!(dvec_copy,ind, [lvl, i])
                Subspaces_copy[i] = [ind]
            else
                if compute_subspace_cost(dvec,pair,ht_coeff_L1) > ht_coeff_L1[lvl][i]

                    ind = findall(x -> x == dvec[pair[1]],dvec_copy)[1]
                    deleteat!(dvec_copy,ind:ind+length(pair)-1)
                    insert!(dvec_copy,ind,[lvl, i])

                    tmp_count -= length(pair) - 1
                    push!(tmp_loc, i)

                    Subspaces_copy[i] = [ind]

                else
                    Subspaces_copy[i] = [k + tmp_count for k in Subspaces[i]]
                end
            end
        end

        dvec = dvec_copy
        if lvl > 1
            Subspaces = [union_array_of_arrays(Subspaces_copy[pair]) for pair in Parent[lvl-1]]
        else
            Subspaces = [[1]]
        end
    end

    return dvec
end



function compute_subspace_cost(dvec,arr,ht_coeff_L1)
    s = 0
    for ele in dvec[arr]
        s += ht_coeff_L1[ele[1]][ele[2]]
    end
    return s
end

function union_array_of_arrays(arr)
    s = []
    for k in arr
        s = union(s, k)
    end
    return s
end
