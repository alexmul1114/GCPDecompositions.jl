## Symmetric Tensor Type

"""
    SymArray{T,N,K} <: AbstractArray{T,N}

`N`-dimensional symmetric tensor with `K` cells of symmetric modes,
    with only unique values stored in packed, column-major order.
Elements can be accessed as for a standard `N`-dimensional array,
    or with a linear index corresponding to the packed storage format.

Fields:
+ `vals::Vector{T}`       : packed vector of values
+ `S::NTuple{N,Int}`      : tuple defining symmetry 
+ `sizes::NTuple{K,Int}`  : tuple of dimensions of each cell
"""
struct SymArray{T,N,K} <: AbstractArray{T,N}
    vals::Vector{T}
    S::NTuple{N,Int}
    sizes::NTuple{K,Int}

    function SymArray{T,N,K}(
        vals::Vector{T},
        S::NTuple{N,Int},
        sizes::NTuple{K,Int}
    ) where {T,N,K}
        _check_sym_pattern(S, K)
        length(vals) == _packed_length(S, sizes) || throw(
            DimensionMismatch(
                "Number of unique elements defined by `S` and `sizes` does not match length of `vals`"
            )
        )
        return new{T,N,K}(vals, S, sizes)
    end
end

"""
    SymArray(vals::Vector{T}, S::NTuple{N,Int}, sizes::NTuple{K,Int})

Construct a `SymArray` with symmetry `S` and cell sizes `sizes`,
given the packed, column-major values `vals`.
"""
SymArray(vals::Vector{T}, S::NTuple{N,Int}, sizes::NTuple{K,Int}) where {T,N,K} =
    SymArray{T,N,K}(vals, S, sizes)

"""
    SymArray(X::AbstractArray{T,N}, S::NTuple{N,Int})

Construct a `SymArray` with symmetry `S` from the dense array `X`.
`X` is assumed to have the symmetry given by `S` (this is not checked).
"""
function SymArray(X::AbstractArray{T,N}, S::NTuple{N,Int}) where {T,N}
    Base.require_one_based_indexing(X)
    K = length(unique(S))
    _check_sym_pattern(S, K)
    sizes = _cell_sizes(X, S, K)

    vals = Vector{T}(undef, _packed_length(S, sizes))
    _pack_sym_vals!(vals, X, S)
    return SymArray(vals, S, sizes)
end

"""
    size(X::SymArray)

Return the dimensions of the dense array represented by `X`.
"""
size(X::SymArray{T,N}) where {T,N} = ntuple(n -> X.sizes[X.S[n]], Val(N))

"""
    getindex(X::SymArray{T,N}, I::Vararg{Int,N})
    getindex(X::SymArray, i::Int)

Return an entry of `X`. 
With `N` indices, returns the entry of the dense array.
With a single index `i`, returns the `i`th entry in the packed storage.
"""
Base.@propagate_inbounds function getindex(X::SymArray{T,N}, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(X, I...)
    return @inbounds X.vals[_packed_index(X.S, X.sizes, I)]
end
Base.@propagate_inbounds getindex(X::SymArray, i::Int) = X.vals[i]

"""
    _packed_index(S::NTuple{N,Int}, sizes::NTuple{K,Int}, I::NTuple{N,Int})

Map the multilinear index `I` to the corresponding index in the packed storage,
for symmetry pattern `S` and cell sizes `sizes`.

Within each cell with `c` modes of size `n`, if `j_1 >= j_2 >= ... >= j_c` are the cell's
indices sorted in non-increasing order, the (zero-based) index within the cell is
`binomial(n + c - 1, c) - 1 - sum(binomial(n - j_t + t - 1, t) for t in 1:c)`.
Cells are combined in column-major order.
"""
function _packed_index(S::NTuple{N,Int}, sizes::NTuple{K,Int}, I::NTuple{N,Int}) where {N,K}
    idx = 1
    cell_offset = 1
    first_mode = 1
    for k in 1:K
        # Cells are contiguous, so cell k is modes first_mode:last_mode
        last_mode = first_mode
        while last_mode < N && S[last_mode+1] == k
            last_mode += 1
        end
        num_modes = last_mode - first_mode + 1
        mode_size = sizes[k]
        total_idxs_cell = binomial(mode_size + num_modes - 1, num_modes)

        # Position t of each index in the cell's non-increasing order (ties broken by mode),
        # found by counting instead of sorting to avoid allocating
        s = 0
        for m in first_mode:last_mode
            t = 1
            for m2 in first_mode:last_mode
                t += (I[m2] > I[m]) | ((I[m2] == I[m]) & (m2 < m))
            end
            s += binomial(mode_size - I[m] + t - 1, t)
        end

        idx += cell_offset * (total_idxs_cell - s - 1)
        cell_offset *= total_idxs_cell
        first_mode = last_mode + 1
    end
    return idx
end

"""
    _pack_sym_vals!(vals::Vector, X::AbstractArray{T,N}, S::NTuple{N,Int})

Fill `vals` with the unique entries of the symmetric tensor `X` in packed, column-major order.
Uses the same loop structure (and so the same ordering) as `fill_reduced_Y_vec!`.
"""
@generated function _pack_sym_vals!(vals::Vector, X::AbstractArray{T,N}, S::NTuple{N,Int}) where {T,N}
    quote
        vec_idx = 1
        @inbounds @nloops(
            $N,
            i,
            k -> (k == $N ? 1 : S[k] == S[k+1] ? i_{k+1} : 1):size(X, k),
            begin
                vals[vec_idx] = @nref $N X i
                vec_idx += 1
            end
        )
        return vals
    end
end

"""
    symmetrize_tensor(X::AbstractArray{T,N}, S::NTuple{N,Int})

Symmetrize the tensor `X` with respect to the symmetry defined by `S`,
returning the result as a `SymArray`.

Each unique entry is the average of `X` over all permutations of the indices
within each symmetric cell. Only the unique entries are computed.
"""
function symmetrize_tensor(X::AbstractArray{T,N}, S::NTuple{N,Int}) where {T,N}
    Base.require_one_based_indexing(X)
    K = length(unique(S))
    _check_sym_pattern(S, K)
    sizes = _cell_sizes(X, S, K)

    groups = [findall(==(k), S) for k in 1:K]
    cell_sizes = length.(groups)

    # Form all permutations of the modes within each cell
    cell_perms = [collect(Combinatorics.permutations(g)) for g in groups]
    all_perms = NTuple{N,Int}[]
    for p in Iterators.product(cell_perms...)
        push!(all_perms, Tuple(vcat(p...)))
    end

    # Symmetric part of integer data may include floats, so need to promote
    TS = float(T)
    constant_factor = TS(1 / prod(factorial.(cell_sizes)))

    vals = Vector{TS}(undef, _packed_length(S, sizes))
    _fill_symmetrized_vals!(vals, X, S, all_perms, constant_factor)
    return SymArray(vals, S, sizes)
end

"""
    _fill_symmetrized_vals!(vals::Vector, X::AbstractArray{T,N}, S::NTuple{N,Int}, all_perms, constant_factor)

Fill `vals` with the unique entries of the symmetrized `X` in packed, column-major order.
Each entry is `constant_factor` times the sum of `X` over the index permutations in `all_perms`.
"""
@generated function _fill_symmetrized_vals!(
    vals::Vector,
    X::AbstractArray{T,N},
    S::NTuple{N,Int},
    all_perms::Vector{NTuple{N,Int}},
    constant_factor
) where {T,N}
    quote
        vec_idx = 1
        @inbounds @nloops(
            $N,
            i,
            k -> (k == $N ? 1 : S[k] == S[k+1] ? i_{k+1} : 1):size(X, k),
            begin
                I = @ntuple $N i
                acc = zero(T)
                for perm in all_perms
                    acc += @nref $N X n -> I[perm[n]]
                end
                vals[vec_idx] = constant_factor * acc
                vec_idx += 1
            end
        )
        return vals
    end
end

# Size of each symmetric cell of X, taken from its first mode; all modes in a cell must match
function _cell_sizes(X::AbstractArray{T,N}, S::NTuple{N,Int}, K::Int) where {T,N}
    sizes = ntuple(k -> size(X, findfirst(==(k), S)), K)
    for n in 1:N
        size(X, n) == sizes[S[n]] || throw(
            DimensionMismatch(
                "All modes in a symmetric cell must have the same size"
            )
        )
    end
    return sizes
end

# Check that the symmetric cells in S are numbered 1,...,K and defined contiguously
function _check_sym_pattern(S::NTuple{N,Int}, K::Int) where {N}
    unique(S) == collect(1:K) || throw(
        ArgumentError(
            "Symmetric Groups must be numbered 1,2,...K"
        )
    )
    for n in 2:N
        S[n] >= S[n-1] || throw(
            ArgumentError(
                "Symmetry cells must be defined contiguously"
            )
        )
    end
    return nothing
end

# Number of unique entries of a tensor with symmetry S and cell sizes `sizes`
_packed_length(S::NTuple{N,Int}, sizes::NTuple{K,Int}) where {N,K} =
    prod(k -> binomial(sizes[k] + count(==(k), S) - 1, count(==(k), S)), 1:K; init = 1)

# Check that the symmetric data tensor X has the same symmetry and size as the model M,
# so that the packed entries of X line up with the loops over unique entries of M
function _check_sym_data(X::SymArray, M::SymCPD)
    X.S == M.S || throw(
        ArgumentError(
            "Symmetry of data tensor must match symmetry of model"
        )
    )
    size(X) == size(M) || throw(
        DimensionMismatch(
            "Size of data tensor must match size of model"
        )
    )
    return nothing
end

# Check that the symmetric tensor y has symmetry S and cell sizes matching the factor matrices Us,
# so that the packed entries of y line up with the loops over unique entries
function _check_sym_factors(y::SymArray, Us::Tuple, S::NTuple)
    y.S == S || throw(
        ArgumentError(
            "Symmetry of tensor must match symmetry `S`"
        )
    )
    y.sizes == map(U -> size(U, 1), Us) || throw(
        DimensionMismatch(
            "Cell sizes of tensor must match number of rows of factor matrices"
        )
    )
    return nothing
end