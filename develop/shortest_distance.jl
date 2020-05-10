using LightGraphs, Plots
include(joinpath("..", "src", "func_includer.jl"))

G = loadgraph(joinpath(@__DIR__, "..", "datasets", "RGC100.lgz"))
N = nv(G)
X = load(joinpath(@__DIR__, "..", "datasets", "RGC100_xyz.jld"),"xyz")[:,1:2]
distmx = floyd_warshall_shortest_paths(G).dists

gr(dpi = 300)
plt = scatter_gplot(X; marker = distmx[1,:])
title!("Floyd_Warshall algorithm: distance to vertex 1")

# savefig(plt, "figs/floyd_warshall_demo_RGC100.png")
