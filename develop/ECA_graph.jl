using CSV, Plots
import StatsBase: countmap
include(joinpath("..", "src", "helpers.jl"))
include(joinpath("..", "src", "gplot.jl"))

file = joinpath(@__DIR__, "../datasets/ECA_indexRH/stations.txt")

f = CSV.File(file; header = false, skipto = 17)

struct ECA_STAINFO
   STAID::Array{Int64,1}
   STATIONNAME::Array{String,1}
   COUNTRYNAME::Array{String,1}
   LAT::Array{Float64,1}
   LON::Array{Float64,1}
   HGT::Array{Float64,1}
end

function degree2decimal(arr)
   return arr[1] > 0 ? arr[1] + arr[2]/60 + arr[3]/60^2 : arr[1] - arr[2]/60 - arr[3]/60^2
end

function countryname2colormap(arr)
   N = length(arr)
   v = ones(N)
   for i in 2:N
      if arr[i] != arr[i-1]
         v[i] = v[i-1] + 1
      else
         v[i] = v[i-1]
      end
   end

   return v
end

N = length(f)
c1 = Array{Int64,1}(UndefInitializer(), N)
c2 = Array{String,1}(UndefInitializer(), N)
c3 = Array{String,1}(UndefInitializer(), N)
c4 = Array{Float64,1}(UndefInitializer(), N)
c5 = Array{Float64,1}(UndefInitializer(), N)
c6 = Array{Float64,1}(UndefInitializer(), N)

for i in 1:N
   c1[i] = f[i][1]
   c2[i] = rstrip(f[i][2])
   c3[i] = rstrip(f[i][3])
   c4[i] = degree2decimal(parse.(Float64, split(f[i][4], ":")))
   c5[i] = degree2decimal(parse.(Float64, split(f[i][5], ":")))
   c6[i] = Float64(f[i][6])
end

eca_stainfo = ECA_STAINFO(c1, c2, c3, c4, c5, c6)
X = hcat(eca_stainfo.LON, eca_stainfo.LAT)

scatter_gplot(X; marker = countryname2colormap(eca_stainfo.COUNTRYNAME)); plot!(grid = false)

## Load packages and functions
using VoronoiDelaunay, VoronoiCells

for i = 1:2; X[:,i] .-= (maximum(X[:,i]) + minimum(X[:,i]))/2; end
width_x = maximum(X[:,1]) - minimum(X[:,1]); width_y = maximum(X[:,2]) - minimum(X[:,2])
X_transform = zeros(N,2); for i in 1:N; X_transform[i,:] = X[i,:] ./ [width_x/0.99, width_y/0.99] + [(VoronoiDelaunay.min_coord + VoronoiDelaunay.max_coord)/2, (VoronoiDelaunay.min_coord + VoronoiDelaunay.max_coord)/2]; end
X_transform = round.(X_transform; digits = 5)

pts = [Point2D(X_transform[i,1], X_transform[i,2]) for i in 1:N]
tess = DelaunayTessellation(N)
push!(tess, pts)

xx, yy = getplotxy(delaunayedges(tess))
plt = plot(xx, yy, xlim=[1,2], ylim=[1,2], linestyle=:auto, linewidth=1, linecolor=:blue, grid=false, label="", aspect_ratio=1, frame=:box)

d = Dict()
for i in 1:N
   d[(X_transform[i,1], X_transform[i,2])] = i
end

using LightGraphs

G = Graph(N)
for e in delaunayedges(tess)
   p1 = round.((getx(geta(e)), gety(geta(e))); digits = 5)
   p2 = round.((getx(getb(e)), gety(getb(e))); digits = 5)
   add_edge!(G, Edge(d[p1], d[p2]))
end

gplot(1.0*adjacency_matrix(G), hcat(eca_stainfo.LON, eca_stainfo.LAT); width = 1); plt = scatter_gplot!(hcat(eca_stainfo.LON, eca_stainfo.LAT); marker = countryname2colormap(eca_stainfo.COUNTRYNAME), ms = 5)
savefig(plt, "figs/ECA_station_graph.png")
