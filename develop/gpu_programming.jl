using Revise, CUDAdrv, CUDAnative, CuArrays

N = 100
x_d = CuArrays.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CuArrays.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

CuArrays.@time x_d + y_d

CuArrays.memory_status()
CuArrays.unsafe_free!
