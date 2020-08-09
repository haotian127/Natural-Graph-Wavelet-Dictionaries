# Load all .jl files in src folder

filenames = readdir(joinpath(@__DIR__, "..", "src"))

for f in filenames
    if f == "func_includer.jl" || f == "pSGWT.jl"
        continue
    elseif occursin(r"^.*\.jl$",f)
        include(joinpath(@__DIR__, f))
    end
end

# Load module
push!(LOAD_PATH,"src")
using pSGWT
