using BenchmarkTools
using Random
using GCPDecompositions

T = Float64

# Test least squares
function test_ls(n, r, alg)
    Random.seed!(0)
    X = rand(n,n,n)
    M = CPD(ones(T, r), rand.(T, size(X), r))
    return @btime gcp($X, $r; init=$M, algorithm=$alg)
end

function main()

    ns = [20, 50, 100, 200, 400]
    r = 5

    # ALS
    for n in ns
        print("Benchmarking ALS size $n x $n x $n:")
        test_ls(n,r,CP_ALS())
    end

    # Fast ALS
    for n in ns[1:3]
        print("Benchmarking Fast-ALS size $n x $n x $n:")
        test_ls(n,r,CP_FastALS())
    end

    # LBFGS
    for n in ns
        print("Benchmarking LBFGS size $n x $n x $n:")
        test_ls(n,r,GCP_LBFGSB())
    end

end

main()