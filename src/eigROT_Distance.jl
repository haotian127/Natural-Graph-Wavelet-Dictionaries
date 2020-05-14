# require: JuMP.jl and Clp.jl
using JuMP, Clp

function eigROT_Distance(Ve,Q; le = 0, alp = 1.0)
#input: Ve, feature matrix of eigenvectors, e.g., V.^2; Q is the oriented incidence_matrix of the graph;
#le is the length vector, which stores the length of each edge;
#output: ROT distance matrix between eigenvectors
n = size(Ve)[2]
dis = zeros(n,n)
if le == 0
        Q2 = [Q -Q]
        m2 = size(Q2)[2]
        for i = 1:n-1, j = i+1:n
                f = Ve[:,i] - Ve[:,j]
                md = Model(with_optimizer(Clp.Optimizer, LogLevel = 0));
                @variable(md, w[1:m2] >= 0.0);
                @objective(md, Min, sum(w));
                @constraint(md, Q2 * w .== f);
                status = optimize!(md);
                wt = abs.(JuMP.value.(w));
                dis[i,j] = norm(wt .^ alp,1)
        end
else
        le2 = [le;le]
        Q2 = [Q -Q]
        m2 = size(Q2)[2]
        for i = 1:n-1, j = i+1:n
                f = Ve[:,i] - Ve[:,j]
                md = Model(with_optimizer(Clp.Optimizer, LogLevel = 0));
                @variable(md);@variable(md, w[1:m2] >= 0.0);
                @objective(md, Min, sum(w.*le2));
                @constraint(md, Q2 * w .== f);
                status = optimize!(md);
                wt = abs.(JuMP.value.(w));
                dis[i,j] = norm((wt .^ alp) .* le2,1)
        end
end
        return dis + dis'
end



function ROT_Distance(A,B,Q; le = 0, alp = 1.0)
#input: A, matrix of initial pmfs; B, matrix of terminal pmfs; Q is the oriented incidence_matrix of the graph;
#le is the length vector, which stores the length of each edge;
#output: ROT distance matrix between eigenvectors
m = size(A)[2]
n = size(B)[2]
dis = zeros(m,n)
if le == 0
        Q2 = [Q -Q]
        m2 = size(Q2)[2]
        for i = 1:m, j = 1:n
                f = A[:,i] - B[:,j]
                md = Model(with_optimizer(Clp.Optimizer, LogLevel = 0));
                @variable(md, w[1:m2] >= 0.0);
                @objective(md, Min, sum(w));
                @constraint(md, Q2 * w .== f);
                status = optimize!(md);
                wt = abs.(JuMP.value.(w));
                dis[i,j] = norm(wt .^ alp,1)
        end
else
        le2 = [le;le]
        Q2 = [Q -Q]
        m2 = size(Q2)[2]
        wt = zeros(m2)
        for i = 1:m, j = 1:n
                f = A[:,i] - B[:,j]
                md = Model(with_optimizer(Clp.Optimizer, LogLevel = 0));
                @variable(md);@variable(md, w[1:m2] >= 0.0);
                @objective(md, Min, sum(w.*le2));
                @constraint(md, Q2 * w .== f);
                status = optimize!(md);
                wt = abs.(JuMP.value.(w));
                dis[i,j] = norm((wt .^ alp) .* le2,1)
        end
end
        return dis
end
