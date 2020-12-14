using CSV, Plots
include(joinpath("..", "src", "helpers.jl"))

data = Dict()
staids = (Int64)[]
filenames = readdir(joinpath(@__DIR__, "../datasets/ECA_indexRH"))

for file in filenames[1:end-1]
    f = CSV.File(joinpath(@__DIR__, "../datasets/ECA_indexRH/", file); header = false, skipto = 31, delim = ":")
    staid_data = Dict()
    for row in f
        staid_data[parse(Int64, row[1][8:11])] = [parse(Float64, row[1][69+(k-1)*8:76+(k-1)*8])*0.01 for k = 1:12]
    end
    staid = parse(Int64, f[1][1][1:6])
    data[staid] = staid_data
    push!(staids, staid)
end


year, month = 2008, 1

arr = []
for year in 1900:2020
    x = (Int64)[]
    y = (Float64)[]
    for k in staids
        if haskey(data[k], year)
            push!(x, k)
            push!(y, data[k][year][month])
        end
    end
    push!(arr, length(x))
end
plot(arr)
