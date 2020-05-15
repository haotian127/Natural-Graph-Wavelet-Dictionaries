using Revise, CUDAdrv, CUDAnative, CuArrays

N = 100
x_d = CuArrays.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CuArrays.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

CuArrays.@time x_d + y_d

CuArrays.memory_status()
CuArrays.unsafe_free!

##
# A * x = b

n = 1000

A = sprand(Float32, n, n, 0.5)
rank(A)
A = sparse(A*A')
d_A = CuArrays.CUSPARSE.CuSparseMatrixCSR(A)

b = rand(Float32, n)
d_b = CuArray(b)

x = zeros(Float32, n)
d_x = CuArray(x)

tol = convert(real(Float32), 1e-6)
d_x = CUSOLVER.csrlsvqr!(d_A, d_b, d_x, tol, one(Cint), 'O')
h_x = collect(d_x)

h_x ≈ Array(A)\b
