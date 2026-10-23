## Symmetric tensor type

@testitem "Constructors" begin
    using GCPDecompositions: SymArray

    @testset "T=$T" for T in [Float64, Float16, Int]
        # Test valid symmetry patterns: length(vals) = prod over groups of binomial(I_k + n_k - 1, I_k)
        @test SymArray(zeros(T, 3), (1,), (3,)) isa SymArray{T,1,1}
        @test SymArray(zeros(T, 6), (1, 1), (3,)) isa SymArray{T,2,1}
        @test SymArray(zeros(T, 10), (1, 1, 1), (3,)) isa SymArray{T,3,1}
        @test SymArray(zeros(T, 24), (1, 1, 2), (3, 4)) isa SymArray{T,3,2}
        @test SymArray(zeros(T, 9), (1, 2, 2), (3, 2)) isa SymArray{T,3,2}
        @test SymArray(zeros(T, 24), (1, 2, 3), (2, 3, 4)) isa SymArray{T,3,3}
        @test SymArray(zeros(T, 15), (1, 1, 1, 1), (3,)) isa SymArray{T,4,1}
        @test SymArray(zeros(T, 18), (1, 1, 2, 2), (2, 3)) isa SymArray{T,4,2}

        # Fields are stored correctly
        Y = SymArray(collect(T, 1:24), (1, 1, 2), (3, 4))
        @test Y.vals == collect(T, 1:24)
        @test Y.S == (1, 1, 2)
        @test Y.sizes == (3, 4)

        # Explicit type parameters match the outer constructor
        @test SymArray{T,3,2}(zeros(T, 24), (1, 1, 2), (3, 4)) isa SymArray{T,3,2}

        # Groups must be numbered 1,...,K with every group used
        @test_throws ArgumentError SymArray(zeros(T, 9), (2, 2), (3, 3))       # doesn't start at 1
        @test_throws ArgumentError SymArray(zeros(T, 18), (1, 1, 3), (3, 3))   # skips group 2 (K = 2)
        @test_throws ArgumentError SymArray(zeros(T, 54), (1, 1, 3), (3, 3, 3)) # group 2 unused (K = 3)
        @test_throws ArgumentError SymArray(zeros(T, 6), (1, 1), (3, 3))       # K larger than groups in S
        @test_throws ArgumentError SymArray(zeros(T, 9), (0, 1), (3, 3))       # group 0
        @test_throws ArgumentError SymArray(zeros(T, 9), (2, 1), (3, 3))       # groups out of order

        # Groups must be contiguous
        @test_throws ArgumentError SymArray(zeros(T, 18), (1, 2, 1), (3, 3))
        @test_throws ArgumentError SymArray(zeros(T, 36), (1, 2, 1, 2), (3, 3))

        # Length of vals must match the number of unique entries
        @test_throws DimensionMismatch SymArray(zeros(T, 9), (1, 1, 1), (3,))
        @test_throws DimensionMismatch SymArray(zeros(T, 11), (1, 1, 1), (3,))
        @test_throws DimensionMismatch SymArray(zeros(T, 27), (1, 1, 2), (3, 3))  # full (non-packed) size
        @test_throws DimensionMismatch SymArray(zeros(T, 24), (1, 1, 2), (4, 3))  # sizes swapped
    end
end

@testitem "Constructor from dense array" begin
    using GCPDecompositions: SymArray

    # Expected packed values: entries at index representatives in column-major order
    packed(X, S) = [X[I] for I in CartesianIndices(X)
                    if all(n -> S[n] != S[n+1] || I[n] >= I[n+1], 1:length(S)-1)]

    @testset "T=$T, S=$S" for T in [Float64, Float16, Int],
        (S, sizes) in [
            ((1, 1), (3,)),
            ((1, 2), (3, 4)),
            ((1, 1, 1), (3,)),
            ((1, 1, 2), (3, 4)),
            ((1, 2, 2), (3, 2)),
            ((1, 2, 3), (2, 3, 4)),
            ((1, 1, 1, 1), (3,)),
            ((1, 1, 2, 2), (2, 3)),
        ]

        # Dense array with symmetry S, formed from a rank-2 SymCPD
        λ = T.(ones(2))
        U = ntuple(k -> T.(rand(1:5, sizes[k], 2)), length(sizes))
        X = Array(SymCPD(λ, U, S))

        Xsym = SymArray(X, S)
        N, K = length(S), length(sizes)
        @test Xsym isa SymArray{T,N,K}
        @test Xsym.S == S
        @test Xsym.sizes == sizes
        @test Xsym.vals == packed(X, S)
    end

    # Invalid S
    @test_throws ArgumentError SymArray(zeros(3, 3, 3), (1, 1, 3))
    @test_throws ArgumentError SymArray(zeros(3, 3, 3), (1, 2, 1))

    # Modes in the same cell must have the same size
    @test_throws DimensionMismatch SymArray(zeros(3, 4), (1, 1))
    @test_throws DimensionMismatch SymArray(zeros(3, 3, 2, 3), (1, 1, 2, 2))
end

@testitem "symmetrize_tensor" begin
    using GCPDecompositions: SymArray, symmetrize_tensor
    using Combinatorics: permutations

    # Expected packed values: entries at index representatives in column-major order
    packed(X, S) = [X[I] for I in CartesianIndices(X)
                    if all(n -> S[n] != S[n+1] || I[n] >= I[n+1], 1:length(S)-1)]

    # Brute-force symmetrization: average over all mode permutations that stay within each cell
    function brute_symmetrize(X, S)
        N = length(S)
        perms = [p for p in permutations(1:N) if all(n -> S[p[n]] == S[n], 1:N)]
        return map(CartesianIndices(X)) do I
            sum(X[Tuple(I)[p]...] for p in perms) / length(perms)
        end
    end

    patterns = [
        ((1, 1), (3,)),
        ((1, 2), (3, 4)),
        ((1, 1, 1), (3,)),
        ((1, 1, 2), (3, 4)),
        ((1, 2, 2), (3, 2)),
        ((1, 2, 3), (2, 3, 4)),
        ((1, 1, 1, 1), (3,)),
        ((1, 1, 2, 2), (2, 3)),
    ]

    # Symmetric input: symmetrizing should leave the unique entries unchanged
    @testset "symmetric input, T=$T, S=$S" for T in [Float64, Float16, Int], (S, sizes) in patterns
        λ = T.(ones(2))
        U = ntuple(k -> T.(rand(1:5, sizes[k], 2)), length(sizes))
        X = Array(SymCPD(λ, U, S))

        Xsym = symmetrize_tensor(X, S)
        @test Xsym isa SymArray{float(T),length(S),length(sizes)}
        @test Xsym.S == S
        @test Xsym.sizes == sizes
        @test Xsym.vals ≈ SymArray(X, S).vals
    end

    # General (non-symmetric) input: matches brute-force symmetrization
    @testset "general input, T=$T, S=$S" for T in [Float64, Int], (S, sizes) in patterns
        dims = ntuple(n -> sizes[S[n]], length(S))
        X = T == Int ? rand(1:9, dims...) : randn(dims...)

        Xsym = symmetrize_tensor(X, S)
        @test Xsym.vals ≈ packed(brute_symmetrize(X, S), S)

        # Symmetrizing is idempotent
        @test symmetrize_tensor(brute_symmetrize(X, S), S).vals ≈ Xsym.vals
    end

    # No symmetry (all cells of size one): values are just the entries of X
    X = randn(2, 3, 4)
    @test symmetrize_tensor(X, (1, 2, 3)).vals == vec(X)

    # Element type is promoted to floating point
    @test eltype(symmetrize_tensor(rand(1:9, 3, 3), (1, 1)).vals) == Float64
    @test eltype(symmetrize_tensor(rand(Float32, 3, 3), (1, 1)).vals) == Float32
    @test eltype(symmetrize_tensor(rand(Float16, 3, 3), (1, 1)).vals) == Float16

    # Invalid S
    @test_throws ArgumentError symmetrize_tensor(zeros(3, 3, 3), (1, 1, 3))
    @test_throws ArgumentError symmetrize_tensor(zeros(3, 3, 3), (1, 2, 1))

    # Modes in the same cell must have the same size
    @test_throws DimensionMismatch symmetrize_tensor(zeros(3, 4), (1, 1))
    @test_throws DimensionMismatch symmetrize_tensor(zeros(3, 3, 2, 3), (1, 1, 2, 2))
end

@testitem "size" begin
    using GCPDecompositions: SymArray

    @testset "S=$S" for (S, sizes, dims) in [
        ((1,), (3,), (3,)),
        ((1, 1), (3,), (3, 3)),
        ((1, 2), (3, 4), (3, 4)),
        ((1, 1, 1), (3,), (3, 3, 3)),
        ((1, 1, 2), (3, 4), (3, 3, 4)),
        ((1, 2, 2), (3, 2), (3, 2, 2)),
        ((1, 2, 3), (2, 3, 4), (2, 3, 4)),
        ((1, 1, 2, 2), (2, 3), (2, 2, 3, 3)),
        ((1, 1, 2, 3, 3), (2, 4, 3), (2, 2, 4, 3, 3)),
    ]
        # From packed values
        len = prod(k -> binomial(sizes[k] + count(==(k), S) - 1, count(==(k), S)), eachindex(sizes))
        Y = SymArray(zeros(len), S, sizes)
        @test size(Y) == dims
        @test @inferred(size(Y)) isa NTuple{length(S),Int}
        @test ndims(Y) == length(S)
        @test all(n -> size(Y, n) == dims[n], eachindex(dims))
        @test size(Y, length(S) + 1) == 1

        # From a dense array: size matches the original array
        X = zeros(dims...)
        @test size(SymArray(X, S)) == size(X)
    end
end

@testitem "getindex" begin
    using GCPDecompositions: SymArray, _packed_index

    # Index representatives (non-increasing indices within each cell) in column-major order
    representatives(dims, S) = [Tuple(I) for I in CartesianIndices(dims)
                                if all(n -> S[n] != S[n+1] || I[n] >= I[n+1], 1:length(S)-1)]

    @testset "T=$T, S=$S" for T in [Float64, Int],
        (S, sizes) in [
            ((1, 1), (3,)),
            ((1, 2), (3, 4)),
            ((1, 1, 1), (4,)),
            ((1, 1, 2), (3, 4)),
            ((1, 2, 2), (3, 2)),
            ((1, 2, 3), (2, 3, 4)),
            ((1, 1, 1, 1), (3,)),
            ((1, 1, 2, 2), (2, 3)),
            ((1, 1, 2, 3, 3, 3), (2, 4, 3)),
        ]

        # Symmetric dense array (integer-valued entries) from a rank-2 SymCPD
        λ = T.(ones(2))
        U = ntuple(k -> T.(rand(1:5, sizes[k], 2)), length(sizes))
        X = Array(SymCPD(λ, U, S))
        Xsym = SymArray(X, S)

        # Representatives map to packed indices 1, 2, ..., in order
        reps = representatives(size(X), S)
        @test [_packed_index(S, sizes, I) for I in reps] == 1:length(Xsym.vals)

        # Every entry (including permuted indices) matches the dense array
        @test all(I -> Xsym[Tuple(I)...] == X[I], CartesianIndices(X))
        @test all(I -> Xsym[I] == X[I], CartesianIndices(X))
        @test Array(Xsym) == X

        # Single index gives the packed entry
        @test all(i -> Xsym[i] == Xsym.vals[i], eachindex(Xsym.vals))
    end

    # Out of bounds
    Xsym = SymArray(collect(1:12), (1, 1, 2), (2, 4))
    @test_throws BoundsError Xsym[3, 1, 1]
    @test_throws BoundsError Xsym[1, 1, 5]
    @test_throws BoundsError Xsym[0, 1, 1]
    @test_throws BoundsError Xsym[13]
    @test_throws BoundsError Xsym[0]

end
