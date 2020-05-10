# Load all .jl files in src folder

filenames = readdir("src")

for f in filenames
    if f == "func_includer.jl"
        continue
    elseif occursin(r"^.*\.jl$",f)
        print(joinpath(@__DIR__, f), "\n")
        include(joinpath(@__DIR__, f))
    end
end
