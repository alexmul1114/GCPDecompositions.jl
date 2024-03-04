## Tensor Kernels

"""
    mttkrp(X, (U1, U2, ..., UN), n)

Compute the Matricized Tensor Times Khatri-Rao Product (MTTKRP)
of an N-way tensor X with the matrices U1, U2, ..., UN along mode n.

See also: `mttkrp!`
"""
mttkrp(X, U, n) = mttkrp!(similar(U[n]), X, U, n)

"""
    mttkrp!(G, X, (U1, U2, ..., UN), n)

Compute the Matricized Tensor Times Khatri-Rao Product (MTTKRP)
of an N-way tensor X with the matrices U1, U2, ..., UN along mode n
and store the result in G.

Algorithm is based on Section III-B of the paper:
> **Fast Alternating LS Algorithms for High Order
>   CANDECOMP/PARAFAC Tensor Factorizations**.
> Anh-Huy Phan, Petr Tichavský, Andrzej Cichocki.
> *IEEE Transactions on Signal Processing*, 2013.
> DOI: 10.1109/TSP.2013.2269903

See also: `mttkrp`
"""
function mttkrp!(G, X, U, n)
    # Dimensions
    Base.require_one_based_indexing(G, X, U...)
    N, I, r = length(U), Tuple(size.(U, 1)), (only ∘ unique)(size.(U, 2))
    (N == ndims(X) && I == size(X)) ||
        throw(DimensionMismatch("`X` and `U` do not have matching dimensions"))
    n in 1:N || throw(DimensionMismatch("`n` must be in `1:ndims(X)`"))
    size(G) == size(U[n]) ||
        throw(DimensionMismatch("Output `G` must have the same size as `U[n]`"))

    # Choose appropriate multiplication order:
    # + n == 1: no splitting required
    # + n == N: no splitting required
    # + 1 < n < N: better to multiply "bigger" side out first
    #   + prod(I[1:n]) > prod(I[n:N]): better to multiply left-to-right
    #   + prod(I[1:n]) < prod(I[n:N]): better to multiply right-to-left
    if n == 1
        buffer_kr = similar(U[2], prod(I[2:N]), r)
        khatrirao!(buffer_kr, U[reverse(2:N)]...)
        mul!(G, reshape(X, I[1], :), buffer_kr)
    elseif n == N
        buffer_kr = similar(U[1], prod(I[1:N-1]), r)
        khatrirao!(buffer_kr, U[reverse(1:N-1)]...)
        mul!(G, transpose(reshape(X, :, I[N])), buffer_kr)
    elseif prod(I[1:n]) > prod(I[n:N])
        # Inner multiplication: left side
        buffer = n == 2 ? (;) : (; kr_inner = similar(U[1], prod(I[1:n-1]), r))
        kr_left = n == 2 ? U[1] : khatrirao!(buffer.kr_inner, U[reverse(1:n-1)]...)
        L = reshape(transpose(reshape(X, :, prod(I[n:N]))) * kr_left, (I[n:N]..., r))

        # Outer multiplication: right side
        buffer = n + 1 == N ? (;) : (; kr_outer = similar(U[n+1], prod(I[n+1:N]), r))
        kr_right = n + 1 == N ? U[N] : khatrirao!(buffer.kr_outer, U[reverse(n+1:N)]...)
        for j in 1:r
            mul!(
                view(G, :, j),
                reshape(selectdim(L, ndims(L), j), I[n], :),
                view(kr_right, :, j),
            )
        end
    else
        # Inner multiplication: right side
        buffer = n + 1 == N ? (;) : (; kr_inner = similar(U[n+1], prod(I[n+1:N]), r))
        kr_right = n + 1 == N ? U[N] : khatrirao!(buffer.kr_inner, U[reverse(n+1:N)]...)
        R = reshape(reshape(X, prod(I[1:n]), :) * kr_right, (I[1:n]..., r))

        # Outer multiplication: left side
        buffer = n == 2 ? (;) : (; kr_outer = similar(U[1], prod(I[1:n-1]), r))
        kr_left = n == 2 ? U[1] : khatrirao!(buffer.kr_outer, U[reverse(1:n-1)]...)
        for j in 1:r
            mul!(
                view(G, :, j),
                transpose(reshape(selectdim(R, ndims(R), j), :, I[n])),
                view(kr_left, :, j),
            )
        end
    end
    return G
end

"""
    khatrirao(A1, A2, ...)

Compute the Khatri-Rao product (i.e., the column-wise Kronecker product)
of the matrices `A1`, `A2`, etc.
"""
function khatrirao(A::Vararg{T,N}) where {T<:AbstractMatrix,N}
    Base.require_one_based_indexing(A...)

    # Special case: N = 1
    if N == 1
        return A[1]
    end

    # General case: N > 1
    I, r = size.(A, 1), (only ∘ unique)(size.(A, 2))
    return khatrirao!(similar(A[1], prod(I), r), A...)
end

"""
    khatrirao!(K, A1, A2, ...)

Compute the Khatri-Rao product (i.e., the column-wise Kronecker product)
of the matrices `A1`, `A2`, etc. and store the result in `K`.
"""
function khatrirao!(K::T, A::Vararg{T,N}) where {T<:AbstractMatrix,N}
    # Check dimensions
    Base.require_one_based_indexing(K, A...)
    I, r = size.(A, 1), (only ∘ unique)(size.(A, 2))
    size(K) == (prod(I), r) || throw(
        DimensionMismatch(
            "Output `K` must have size equal to `(prod(size.(A,1)), size(A[1],2))",
        ),
    )

    # Special case: N = 1
    if N == 1
        K .= A[1]
        return K
    end

    # Base case: N = 2
    if N == 2
        reshape(K, I[2], I[1], r) .= reshape(A[2], :, 1, r) .* reshape(A[1], 1, :, r)
        return K
    end

    # Recursive case: N > 2
    n = argmin(n -> I[n] * I[n+1], 1:N-1)
    return khatrirao!(K, A[1:n-1]..., khatrirao(A[n], A[n+1]), A[n+2:end]...)
end
