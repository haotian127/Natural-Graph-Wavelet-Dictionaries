## Load packages and functions
using VoronoiDelaunay, VoronoiCells, GeometricalPredicates
include(joinpath("..", "src", "func_includer.jl"))

barbara = JLD.load(joinpath(@__DIR__, "..", "datasets", "barbara_gray_matrix.jld"), "barbara")
G, L, X = SunFlowerGraph(; N = 400); N = nv(G)
width_x = maximum(abs.(X[:,1])) * 2; width_y = maximum(abs.(X[:,2])) * 2
X_transform = zeros(N,2); for i in 1:N; X_transform[i,:] = X[i,:] ./ [width_x/(VoronoiDelaunay.max_coord - VoronoiDelaunay.min_coord), width_y/(VoronoiDelaunay.max_coord - VoronoiDelaunay.min_coord)] + [(VoronoiDelaunay.min_coord + VoronoiDelaunay.max_coord)/2, (VoronoiDelaunay.min_coord + VoronoiDelaunay.max_coord)/2]; end

pts = [Point2D(X_transform[i,1], X_transform[i,2]) for i in 1:N]
tess = DelaunayTessellation(N)
push!(tess, pts)

x, y = getplotxy(voronoiedges(tess))
plt = plot(x, y, xlim=[1,2], ylim=[1,2], linestyle=:auto, linewidth=1, linecolor=:blue, grid=false, label="", aspect_ratio=1, frame=:box)
# ; scatter_gplot!(X_transform; ms = LinRange(4.0, 14.0, N), smallValFirst = false, c = :grey)
savefig(plt, "figs/Sunflower_Barbara_eye_Voronoi_cells.png")

idx_pts = [IndexablePoint2D(X_transform[i,1], X_transform[i,2], i) for i in 1:N]
C = voronoicells(idx_pts)
# A = voronoiarea(C)

N1 = 51; N2 = 51; X_barbara = zeros(N1, N2, 2); for i in 1:N1; for j in 1:N2; X_barbara[i,j,1] = VoronoiDelaunay.min_coord + i * (VoronoiDelaunay.max_coord - VoronoiDelaunay.min_coord) / (N1+1); X_barbara[i,j,2] = VoronoiDelaunay.max_coord - j * (VoronoiDelaunay.max_coord - VoronoiDelaunay.min_coord) / (N2+1); end; end; X_barbara = reshape(X_barbara, (N1*N2,2))

f_barbara = reshape(barbara[80:130, 375:425]', N1*N2)[:]
scatter_gplot(X_barbara; marker = f_barbara, ms = 6, c = :greys, smallValFirst = false); plt = plot!(cbar = false, grid = false)
savefig(plt, "figs/Sunflower_Barbara_eye_raw.png")
# plot!(x, y, xlim=[1,2], ylim=[1,2], linestyle=:auto, linewidth=1, linecolor=:blue, grid=false, label="", aspect_ratio=1, frame=:box); scatter_gplot!(X_transform; ms = LinRange(4.0, 14.0, N), smallValFirst = false, c = :grey)


function vertex_sort(plist)
    plist_sorted = plist[1:2]
    n = length(plist)
    idx = 2
    for i in 2:n
        if abs(sum([orientation(Line(plist[1], plist[i]), plist[j]) for j in setdiff(1:n, [1, i])])) == n-2
            idx = i
            break
        end
    end
    rlist = setdiff(1:n, [1,idx])
    l = Line(plist[1], plist[idx])
    side = orientation(l, plist[rlist[1]])
    while length(plist_sorted) < n
        for i in rlist
            if all([orientation(Line(plist[idx], plist[i]), plist[j]) for j in setdiff(1:n, [idx, i])] .== side)
                push!(plist_sorted, plist[i])
                rlist = setdiff(rlist, i)
                idx = i
                break
            end
        end
    end
    return plist_sorted
end


PG = Dict()
for k in 1:N
    plist = vertex_sort(C[k])
    if length(plist) == 4
        PG[k] = Polygon(plist[1], plist[2], plist[3], plist[4])
    elseif length(plist) == 5
        PG[k] = Polygon(plist[1], plist[2], plist[3], plist[4], plist[5])
    elseif length(plist) == 6
        PG[k] = Polygon(plist[1], plist[2], plist[3], plist[4], plist[5], plist[6])
    else
        PG[k] = Polygon(plist[1], plist[2], plist[3], plist[4], plist[5], plist[6], plist[7])
    end
end


f = zeros(N); num_pxls = zeros(N)
for pxl in 1:N1*N2
    px, py = X_barbara[pxl,:]
    P = Point(px, py)
    for k in 1:N
        if inpolygon(PG[k], P)
            f[k] += f_barbara[pxl]
            num_pxls[k] += 1
        end
    end
end

plot(x, y, xlim=[1,2], ylim=[1,2], linestyle=:auto, linewidth=1, linecolor=:blue, grid=false, label="", aspect_ratio=1, frame=:box); plt = scatter_gplot!(X_transform; marker = f ./ num_pxls, ms = LinRange(4.0, 14.0, N), smallValFirst = false, c = :greys)
savefig(plt, "figs/Sunflower_Barbara_eye_Voronoi_sampling_with_cells.png")

scatter_gplot(X_transform; marker = f ./ num_pxls, ms = LinRange(4.0, 14.0, N), smallValFirst = false, c = :greys); plt = plot!(xlim = [0.8,2.2], ylim = [0.8,2.2], clim=(0,1), frame = :none)
savefig(plt, "figs/Sunflower_Barbara_eye_Voronoi_sampling.png")
