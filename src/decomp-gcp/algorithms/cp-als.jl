## Algorithm: CP_ALS

"""
    CP_ALS

**A**lternating **L**east **S**quares.
Workhorse algorithm for `LeastSquares` loss with no constraints.

Algorithm parameters:

+ `maxiters::Int` : max number of iterations (default: `200`)
"""
Base.@kwdef struct CP_ALS <: AbstractGCPAlgorithm
    maxiters::Int = 200
end

function _gcp!(
    rng::AbstractRNG,
    M::CPD{Float64,N},
    X::Array{<:Real,N},
    loss::LeastSquaresLoss,
    constraints::Tuple{},
    algorithm::CP_ALS,
) where {N}
    # Pre-allocate MTTKRP buffers
    mttkrp_buffers = ntuple(n -> create_mttkrp_buffer(X, M.U, n), N)

    # Alternating Least Squares (ALS) iterations
    for _ in 1:algorithm.maxiters
        for n in 1:N
            V = reduce(.*, M.U[i]'M.U[i] for i in setdiff(1:N, n))
            mttkrp!(M.U[n], X, M.U, n, mttkrp_buffers[n])
            rdiv!(M.U[n], lu!(V))
            M.λ .= norm.(eachcol(M.U[n]))
            M.U[n] ./= permutedims(M.λ)
        end
    end

    return M
end

function _gcp!(
    rng::AbstractRNG,
    M::CPD{Float32,N},
    X::MtlArray{<:Real,N},
    loss::LeastSquaresLoss,
    constraints::Tuple{},
    algorithm::CP_ALS,
) where {N}
    # Pre-allocate MTTKRP buffers
    mttkrp_buffers = ntuple(n -> create_mttkrp_buffer(X, M.U, n), N)
    r = ncomps(M)

    # Alternating Least Squares (ALS) iterations
    for _ in 1:algorithm.maxiters
        for n in 1:N
            V = reduce(.*, M.U[i]'M.U[i] for i in setdiff(1:N, n))
            mttkrp!(M.U[n], X, M.U, n, mttkrp_buffers[n])

            # Custom Metal kernel for least-squares solve
            # Updates A in place with its cholesky factorization,
            function cholesky_factor(A)
                i = thread_position_in_grid().x
                j = thread_position_in_grid().y

            end
            # Linear solve using backward substituion with lower triangular matrix L
            # and vector b
            function backward_solve(L, b)

            end

            # @metal threads=(r,r) groups=1 cholesky_factor(M.U[n], V)

            U_cpu = Array(M.U[n])
            V_cpu = Array(V)
            rdiv!(U_cpu, lu!(V_cpu))
            copyto!(M.U[n], U_cpu)
            #M.λ .= norm.(eachcol(M.U[n]))
            copyto!(M.λ, norm.(eachcol(U_cpu)))  # Cannot use .= for MtlArrays


            M.U[n] ./= permutedims(M.λ)
        end
    end

    return M
end


# Sequential version of Cholesky factorization
# Updates A in place with its Cholesky factorization
function cholesky_sequential!(A)
    n = size(A)[1]
    for col in 1:n
        A[col,col] = sqrt(A[col,col])
        A[col+1:n,col] .= A[col+1:n,col] / A[col,col]
        A[col+1:n,col+1:n] .= A[col+1:n,col+1:n] .- (A[col+1:n,col] * A[col+1:n,col]')
    end
end

# Sequential version of Cholesky factorization with given block size
# Updates A in place with its Cholesky factorization
function cholesky_sequential_blocked!(A, block_size)
    n = size(A)[1]
    num_blocks = cld(n, block_size)  
    for col in 1:block_size:n
        b = min(block_size, n-col+1)
        cholesky_sequential!(A[col:col+b, col:col+b])
        A[col+b+1:n, col:col+b] .= A[col+b+1:n, col:col+b] * A[col:col+b, col:col+b]'
    end
end

# Invert lower triangular matrix recursively
# Assuming original call is size 32
function ltri_inv(buffer, L)
    m = 4
    k = 3
    n = size(buffer)[1]
    if n == 

end


# Metal kernel verson of Cholesky factorization
# Updates A in place with its Cholesky factorization
# Can only handle matrices up to size 32x32 before max number of threads is exceeded. Need to do tiled version next.
function cholesky_kernel!(A)
    n = size(A)[1]
    for col in 1:n
        nthreads = n-col+1
        @metal threads=(nthreads) groups=1 cholesky_update_column!(A, col)
        # Symmetric rank-1 Update
        @metal threads=(nthreads,nthreads) groups=1 cholesky_symmetric_rank1_update!(A, col)
    end
    return
end

function cholesky_update_column!(A, col)
    i = thread_position_in_grid().x
    if i == 1
        A[col,col] = sqrt(A[col,col])
    end
    # Ensure that rest of threads read updated value of A[col,col]
    threadgroup_barrier(Metal.MemoryFlagDevice)
    # Update column
    if i != 1
        A[col+i-1,col] = A[col+i-1,col] / A[col,col]
    end
    return
end

function cholesky_symmetric_rank1_update!(A, col)
    i = thread_position_in_grid().x
    j = thread_position_in_grid().y
    A[col+i,col+j] = A[col+i,col+j] - (A[col+i,col] * A[col+j,col]')
    return
end
