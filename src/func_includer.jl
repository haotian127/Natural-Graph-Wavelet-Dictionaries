# Load all .jl files in src folder

filenames = readdir(joinpath(@__DIR__, "..", "src"))

include(joinpath(@__DIR__, "require.jl"))
for f in filenames
    if f == "func_includer.jl" || f == "pSGWT.jl" || f == "require.jl"
        continue
    elseif occursin(r"^.*\.jl$",f)
        include(joinpath(@__DIR__, f))
    end
end

# Load module
push!(LOAD_PATH, @__DIR__)
using pSGWT
