using LightGraphs, Plots, LaTeXStrings, MultivariateStats

include(joinpath("..", "src", "eigDAG_Distance.jl"))

n1, n2 = 7, 3
G = LightGraphs.grid([n1,n2])
n = nv(G)
L = Matrix(laplacian_matrix(G))
Q = incidence_matrix(G; oriented = true)
V = eigen(L).vectors
V = V.*sign.(V[1,:])'
lamb = eigen(L).values

dist_DAG = eigDAG_Distance(V,Q,n)

D = dist_DAG
E = transform(fit(MDS, D, maxoutdim=2, distances=true))

dx = 0.01; dy = dx;
xej = zeros(n1,n); yej=zeros(n2,n);
a = 5.0;
b = 7.0;
for k=1:n
    xej[:,k]=LinRange(E[1,k]-n2*a*dx,E[1,k]+n2*a*dx, n1); yej[:,k]=LinRange(E[2,k]-a*dy,E[2,k]+a*dy, n2)
end

pyplot()
heatmap(xej[:,1],yej[:,1],reshape(V[:,1],(n1,n2))',c=:viridis,colorbar=false,clims=(-0.4,0.4), ratio=1, annotations=(xej[4,1],yej[3,1]+b*dy,text(latexstring("\\varphi_","{0,0}"),10)))
#annotations=(xej[4,1],yej[3,1]+b*dy,text(latexstring("\\varphi_","{0,0}"),10))

eig2dct = [[0 0];[1 0];[2 0];[0 1];[1 1];[3 0];[2 1];[4 0];[3 1];[0 2];[1 2];[5 0];[4 1];[2 2];[6 0];[5 1];[3 2];[6 1];[4 2];[5 2];[6 2]]
for k=2:n
    heatmap!(xej[:,k],yej[:,k],reshape(V[:,k],(n1,n2))',c=:viridis,colorbar=false,clims=(-0.4,0.4),ratio=1, annotations=(xej[4,k], yej[3,k]+b*dy, text(latexstring("\\varphi_{", string(eig2dct[k,1]), ",", string(eig2dct[k,2]), "}"),10)))
    #annotations=(xej[4,k], yej[3,k]+b*dy, text(latexstring("\\varphi_{", string(eig2dct[k,1]), ",", string(eig2dct[k,2]), "}"),10))
end
plt = plot!(aspect_ratio = 1, ylim = [-1.4, 1.3])

savefig(plt, "figs/Grid_DAG.png")
display(plt)
