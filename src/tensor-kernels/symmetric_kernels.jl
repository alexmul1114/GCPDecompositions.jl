"""
    collect_multinomial_coefficients(S::NTuple{N,Int}, cell_sizes::NTuple{K,Int}, ::Val{N}) where {N,K}

Collect the multinomial coefficient (number of repeated entries) for each
unique value in the tensor with symmetry pattern given by S and mode sizes for each cell
given by cell_sizes.
"""
@generated function collect_multinomial_coefficients(
    S::NTuple{N,Int}, 
    cell_sizes::NTuple{K,Int}, 
    ::Val{N}
) where {N,K}
    quote
        coef_dtype = Float64
        sz = _packed_length(S, cell_sizes)
        num = 1.0
        for k in 1:$K
            num *= factorial(count(S .== k))
        end
        coefs = Vector{coef_dtype}(undef, sz)
        vec_idx = 1
        @nloops $N i k -> (k == $N ? 1 : S[k] == S[k+1] ? i_{k+1} : 1):cell_sizes[S[k]] begin
            if $K == 1 && $N == 2
                α = i_1 == i_2 ? 1.0 : 2.0
            elseif $K == 1 && $N == 3
                α = i_1 == i_2 ? (i_2 == i_3 ? 1.0 : 3.0) : (i_2 == i_3 ? 3.0 : 6.0)
            elseif $K == 1 && $N == 4
                n_equal = (i_1==i_2) + (i_1==i_3) + (i_1==i_4) + (i_2==i_3) + (i_2==i_4) + (i_3==i_4)
                α = n_equal == 0 ? 24.0 : n_equal == 1 ? 12.0 : n_equal == 2 ? 6.0 : n_equal == 3 ? 4.0 : 1.0
            else
                idx = @ntuple $N i
                denom = 1.0
                num_modes = 1
                for m in 2:$N
                    if S[m] == S[m-1] && idx[m] == idx[m-1]
                        num_modes += 1
                    else
                        denom *= factorial(num_modes)
                        num_modes = 1
                    end
                end
                denom *= factorial(num_modes)
                α = num / denom
            end
            coefs[vec_idx] = α
            vec_idx += 1
        end
        return coefs
    end
end

"""
    fill_reduced_Y_vec!(Y_vec::AbstractVector, X::SymArray{TX,N,K}, M::SymCPD, loss, ::Val{N}, ::Val{R}) where {TX,N,K,R}

Forms reduced vectorization of derivative tensor Y with only structually unique entries.
The data `X` is read in its packed order, which matches the order of `Y_vec`.
Computes partial products at loop level 2 for efficiency. May be able to get further improvements
    by saving partial products at all loop levels 2 through N.
"""
@generated function fill_reduced_Y_vec!(Y_vec::AbstractVector, X::SymArray{TX,N,K}, M::SymCPD, loss, ::Val{N}, ::Val{R}) where {TX,N,K,R}
    set_partial_m = map(1:R) do j
        terms = [:(M.U[M.S[$k]][$(Symbol("i_$(k)")), $j]) for k in 2:N]
        :(partial_prod[$j] = M.λ[$j] * *( $(terms...) ))
    end
    pre_body = Expr(:block, set_partial_m...)

    quote
        _check_sym_data(X, M)
        S = M.S
        T = eltype(M.U[1])
        partial_prod = zeros(MVector{$R, T})
        mode1_factors = M.U[S[1]]
        X_vals = X.vals
        vec_idx = 1
        @inbounds @nloops(
            $N,
            i,
            k -> (k == $N ? 1 : S[k] == S[k+1] ? i_{k+1} : 1):size(M.U[S[k]], 1),
            d -> d == 2 ? $pre_body : nothing,
            begin
                x = X_vals[vec_idx]
                m = zero(T)
                for col in 1:$R
                    m = muladd(mode1_factors[i_1, col], partial_prod[col], m)
                end 
                Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
                vec_idx += 1
            end
        )
    end
end

"""
    columnwise_ttv_all_modes!(
        result::Vector{TU},
        y::SymArray{TY,N,K},
        Us::NTuple{K,AbstractMatrix{TU}},
        multinomial_coefs::Vector{TC},
        ::Val{S}, ::Val{N}, ::Val{R}
    ) where {TY,N,K,TU,TC,S,R}

Compute the TTV in all modes (i.e., the weights gradient) given the symmetric
    derivative tensor y (read in its packed order) and factor matrices in Us, using the coefficients in
    multinomial_coefs, for general symmetry given by S, order N, and rank R.
"""
@generated function columnwise_ttv_all_modes!(
    result::AbstractVector{TU},
    y::SymArray{TY,N,K},
    Us::NTuple{K,AbstractMatrix{TU}},
    multinomial_coefs::Vector{TC},
    ::Val{S}, ::Val{N}, ::Val{R}
) where {TY,N,K,TU,TC,S,R}

    # Define functions for different symbols in expressions
    Iv(d) = Symbol(:i_, d);  Partial_Prod(d) = Symbol(:partial_prod_, d)
    UM(d) = Symbol(:Ut_, S[d])        # transposed factor at loop position d

    pre_exprs  = [Any[] for _ in 1:N]
    post_exprs = [Any[] for _ in 1:N]

    # Add pre-expression at each level for partial products
    for d in 2:N
        push!(pre_exprs[d], :(for col in 1:$R
            $(Partial_Prod(d))[col] = $(d == N ? :($(UM(d))[col, $(Iv(d))]) :
                              :($(UM(d))[col, $(Iv(d))] * $(Partial_Prod(d+1))[col]))
        end))
    end
    
    # Add pre/post expressions for zeroing/flushing the accumulator
    push!(pre_exprs[2],  :(fill!(acc, zero($TU))))
    push!(post_exprs[2], :(for col in 1:$R
                               result[col] = muladd(acc[col], partial_prod_2[col], result[col])
                           end))

    chain(exprs) = Expr(:->, :d, Expr(:block, foldr((k, rest) -> isempty(exprs[k]) ? rest :
                 Expr(:if, :(d == $k), Expr(:block, exprs[k]...), rest),
                 1:N; init = :nothing)))
    pre, post = chain(pre_exprs), chain(post_exprs)

    alloc = Expr(:block, (:($(Partial_Prod(d)) = zeros(MVector{$R, $TU})) for d in 2:N)...)
    
    quote
        _check_sym_factors(y, Us, $S)
        Sv = $(S)
        acc = zeros(MVector{$R, TU})
        @nexprs $K k -> (Ut_k = permutedims(Us[k]))
        Ufirst = $(UM(1))
        $alloc
        y_vals = y.vals
        vec_idx = 1
        @inbounds @nloops(
            $N,
            i,
            k -> (k == $N ? 1 : Sv[k] == Sv[k+1] ? i_{k+1} : 1):size(Us[Sv[k]], 1),
            $pre,
            $post,
            begin
                tensor_term = multinomial_coefs[vec_idx] * y_vals[vec_idx]
                for col in 1:$R
                    acc[col] = muladd(tensor_term, Ufirst[col, i_1], acc[col])  # Note the product of terms from modes 2-N is factored out into post-expr for i_2
                end
                vec_idx += 1
            end
        )
        return result
    end
end

"""
    columnwise_ttv_all_modes_except_one!(
        result::AbstractMatrix,
        y::SymArray{TY,N,K},
        Us::NTuple{K,AbstractMatrix{TU}},
        multinomial_coefs::Vector{TC},
        ::Val{S}, ::Val{c}, ::Val{N}, ::Val{R}
    ) where {TY,N,K,TU,TC,S,c,R}

Compute the TTSV in all modes except the first one corresponding to cell c, i.e.,
    the gradient for the factors for cell c minus scaling by diag(λ),
    given the symmetric derivative tensor y (read in its packed order)
    and the factor matrices in Us, with pre-computed multinomial coefficients.
result should be dimensions R x n, where R is the rank and n is the mode size for cell c.
This function currently does a permutedims on the factor matrices in Us, but
    it would be better if there was an option to store the factor matrices transposed from the start.
"""
@generated function columnwise_ttv_all_modes_except_one!(
    result::AbstractMatrix,
    y::SymArray{TY,N,K},
    Us::NTuple{K,AbstractMatrix{TU}},
    multinomial_coefs::Vector{TC},
    ::Val{S}, ::Val{c}, ::Val{N}, ::Val{R}
) where {TY,N,K,TU,TC,S,c,R}

    num_modes_cell = count(==(c), S)
    mode = findfirst(==(c), S)
    last_mode_cell = mode + num_modes_cell - 1
    acc_modes = [j + mode - 1 for j in 1:num_modes_cell if j + mode - 1 >= 2]

    need_pref = last_mode_cell >= 3  # Flag whether we can save multiplies by computing prefix products
    x_needed  = sort!(unique!([collect(2:last_mode_cell-1); collect(mode+1:N)])) # Included U in prefix/suffix products, excluding mode 1
    peel = (mode == 1 && num_modes_cell >= 2)  # Flag whether to peel first i_1 iteration
    
    # Define functions for different symbols in expressions
    Iv(d)=Symbol(:i_,d);  Pv(d)=Symbol(:p_,d);   Av(d)=Symbol(:acc_,d)
    Xv(d)=Symbol(:x_,d);  SUF(d)=Symbol(:suf_,d); SP(d)=Symbol(:sp_,d)
    UM(d)=Symbol(:U_, S[d])  # transposed factor matrix at loop position d, defined in quote
    IDiff(d)=Symbol(:idiff_,d) # tracks whether i_d == i_{d+1}, hoisted to level d to reduce branching in innermost loop
    col_assign_loop(lhs, rhs) = :(for col in 1:$R; $lhs = $rhs; end)

    pre_exprs  = [Any[] for _ in 1:N]
    post_exprs = [Any[] for _ in 1:N]

    # Add pre-expressions to hoist checking whether indices in cell are equal outside of innermost loop.
    # Note that for the case when mode == 1 and last_mode_cell >= 2,
    # we peel the i_1 == i_2 iteration of the innermost loop, so the condition for j == 2 is always true.
    for d in max(mode, 2)+1:last_mode_cell
        push!(pre_exprs[d-1], :($(IDiff(d)) = $(Iv(d)) != $(Iv(d-1))))
    end
    
    # Add pre- and post-expressions for each level
    for d in 1:N
        # Load row i_d of matrix U[S[d]] into x_d (pre)
        d in x_needed && push!(pre_exprs[d],
            col_assign_loop(:($(Xv(d))[col]), :($(UM(d))[col, $(Iv(d))])))

        # Compute suffix product x_d * ... * x_N (pre) if it will be used
        if d > mode
            push!(pre_exprs[d], col_assign_loop(:($(SUF(d))[col]),
                d == N ? :($(Xv(d))[col]) : :($(Xv(d))[col] * $(SUF(d+1))[col])))
        end

        # Compute coefficient p_d, the multiplicity of i_d among i_j for j in d:last_mode_cell (pre).
        if d >= max(mode, 2) && d <= last_mode_cell
            push!(pre_exprs[d], d == last_mode_cell ? :($(Pv(d)) = 1) :
                :($(Pv(d)) = ifelse($(Iv(d)) == $(Iv(d+1)), $(Pv(d+1)) + 1, 1)))
        end

        # For level d = 2, compute i_1 loop-invariant product sp_D = p_D * prod_{k=2...N, k != D} x_k,
        # i.e., the leave-one-out-product excluding x_1, but including the coefficient p_D,
        # for D in {2,..,last_mode_cell} if D >= mode (pre).
        # Also compute prefix product x_2 * ... * x_{last_mode_cell - 1} (pre).
        # For d >= mode + 1, i_d == i_{d-1}, we assign zeros to sp_d. This prevents us from having to 
        # do branching in the innermost loop, at a minimal cost of added computation from adding zeros.
        if d == 2
            need_pref && push!(pre_exprs[2], :(fill!(pref, one($TU))))
            for D in 2:last_mode_cell
                if D >= mode
                    fac = Any[Pv(D)]
                    D > 2 && push!(fac, :(pref[col]))
                    D < N && push!(fac, :($(SUF(D+1))[col]))
                    prod_ex = length(fac) == 1 ? only(fac) : Expr(:call, :*, fac...)
                    rhs = D in max(mode,2)+1:last_mode_cell ? :(ifelse($(IDiff(D)), $prod_ex, zero($TU))) : prod_ex
                    push!(pre_exprs[2], col_assign_loop(:($(SP(D))[col]), rhs))
                    # push!(pre_exprs[2], col_assign_loop(:($(SP(D))[col]),
                    #     length(fac) == 1 ? only(fac) : Expr(:call, :*, fac...)))
                end
                D < last_mode_cell && push!(pre_exprs[2],
                    col_assign_loop(:(pref[col]), :(pref[col] * $(Xv(D))[col])))
            end
        end

        # Zero (pre) and flush (post) accumulators
        if d in acc_modes
            # !(peel && d == 2) && push!(pre_exprs[d], :(fill!($(Av(d)), zero($TU)))) # If peel && d == 2 we directly overwrite acc_2
            # The fill in the above line causes major slowdowns for odd R, the below is better.
            !(peel && d == 2) && push!(pre_exprs[d], :(for col in 1:$R
                                    $(Av(d))[col] = zero($TU)
                                end))    # If peel && d == 2 we directly overwrite acc_2
            push!(post_exprs[d], :(for col in 1:$R
                                    result[col, $(Iv(d))] += $(Av(d))[col]
                                end))
        end
    end

    # Peel off first iteration of i_1 loop (i.e., i_1 = i_2) when mode == 1 and num_modes_cell >= 2 to reduce branching
    # and coefficient computation logic (pre).
    if peel
        push!(pre_exprs[2], :(yw = multinomial_coefs[vec_idx] * y_vals[vec_idx]))
        push!(pre_exprs[2], :(tt = (p_2 + 1) * yw))
        push!(pre_exprs[2], col_assign_loop(:(acc_2[col]), :(tt * suf_2[col])))
        if last_mode_cell >= 3
            push!(pre_exprs[2], col_assign_loop(:(yx1[col]), :(yw * x_2[col])))
            for D in 3:last_mode_cell
                push!(pre_exprs[2], Expr(:if, :($(Iv(D)) != $(Iv(D-1))),
                    col_assign_loop(:($(Av(D))[col]),
                        :(muladd(yx1[col], $(SP(D))[col], $(Av(D))[col])))))
            end
        end
        push!(pre_exprs[2], :(vec_idx += 1))
    end

    chain(exprs) = Expr(:->, :d, Expr(:block, foldr((k, rest) -> isempty(exprs[k]) ? rest :
                 Expr(:if, :(d == $k), Expr(:block, exprs[k]...), rest),
                 1:N; init = :nothing)))
    pre, post = chain(pre_exprs), chain(post_exprs)

    # Expressions for allocations
    alloc = Expr(:block)
    for d in x_needed;  push!(alloc.args, :($(Xv(d))  = zeros(MVector{$R, $TU}))) end
    for d in mode+1:N;  push!(alloc.args, :($(SUF(d)) = zeros(MVector{$R, $TU}))) end
    for D in acc_modes; push!(alloc.args, :($(Av(D))  = zeros(MVector{$R, $TU}))) end
    for D in acc_modes; push!(alloc.args, :($(SP(D))  = zeros(MVector{$R, $TU}))) end
    need_pref           && push!(alloc.args, :(pref = zeros(MVector{$R, $TU})))
    !isempty(acc_modes) && push!(alloc.args, :(yx1  = zeros(MVector{$R, $TU})))

    # Expressions for loop body
    # Compute value of p_1*yw if mode != 1
    tt1_def = peel                ? :(tt1 = yw) :
              mode != 1           ? :nothing :
              num_modes_cell == 1 ? :(tt1 = yw) :
                                    :(tt1 = ifelse(i_1 == i_2, p_2 + 1, 1) * yw)

    yx1_def = isempty(acc_modes)  ? :nothing :
              :(for col in 1:$R
                    yx1[col] = yw * Ufirst[col, i_1]
                end)

    quote
        _check_sym_factors(y, Us, $S)
        Sv = $(S)
        @nexprs $K k -> (U_k = permutedims(Us[k]))
        Ufirst = $(Symbol(:U_, S[1]))
        $alloc
        y_vals = y.vals
        vec_idx = 1
        @inbounds @nloops(
            $N, 
            i, 
            k -> (k == $N ? 1 :
                    k == 1 ? $(peel ? :(i_2 + 1) : (S[1] == S[2] ? :(i_2) : 1)) :
                    Sv[k] == Sv[k+1] ? i_{k+1} : 1):size(Us[Sv[k]], 1),
            $pre, 
            $post, 
            begin   # Body expr
                yw = multinomial_coefs[vec_idx] * y_vals[vec_idx]
                $tt1_def
                $yx1_def
                @nexprs $num_modes_cell j -> begin
                    if j > $(2 - mode)
                        for col in 1:$R
                            acc_{j+$mode-1}[col] = muladd(yx1[col], sp_{j+$mode-1}[col], acc_{j+$mode-1}[col])
                        end
                    else
                        for col in 1:$R
                            result[col, i_1] = muladd(tt1, suf_2[col], result[col, i_1])
                        end
                    end
                    # end
                end
                vec_idx += 1
            end
        )
        return result
    end
end

# Find split points for outer loop such that load across threads is (close) to balanced,
# and starting vec_idxs for each thread
@generated function ttv_threading_plan(
    S::NTuple{N,Int}, 
    cell_sizes::NTuple{K,Int}, 
    num_threads::Int,
    ::Val{N}
) where {N,K}
    quote
        total_sz = _packed_length(S, cell_sizes)
        chunk_iters = div(total_sz, num_threads)
        iN_starts = [1]
        vec_idx_starts = [1]
        vec_idx = 1
        chunk_idx = 1
        @nloops(
            $N,
            i,
            k -> (k == $N ? 1 : S[k] == S[k+1] ? i_{k+1} : 1):cell_sizes[S[k]],
            k -> k == $N ? 
                begin
                    if chunk_idx > chunk_iters
                        push!(iN_starts, i_{$N})
                        push!(vec_idx_starts, vec_idx)
                        chunk_idx = chunk_idx - chunk_iters
                    end
                end
                : nothing,
            begin
                chunk_idx += 1
                vec_idx += 1
            end
        )
        return iN_starts, vec_idx_starts
    end
end

function fill_reduced_Y_vec_multithreaded!(
    Y_vec::AbstractVector, 
    X::SymArray{TX,N,K}, M::SymCPD,
    loss,
    iN_starts::Vector{Int}, vec_idx_starts::Vector{Int},
    num_threads::Int,
) where {TX,N,K}
    _check_sym_data(X, M)

    nchunks = min(num_threads, length(iN_starts)) 
    
    Threads.@threads for t in 1:nchunks
        lo = iN_starts[t]
        hi = t == nchunks ? size(M.U[M.S[N]], 1) : iN_starts[t+1] - 1
        v0 = vec_idx_starts[t]
        _fill_reduced_Y_vec_multithreaded_chunk!(Y_vec, X, M, loss, lo, hi, v0, Val(N), Val(ncomps(M)))
    end

    return Y_vec
end

@generated function _fill_reduced_Y_vec_multithreaded_chunk!(
    partial_Y_vec::AbstractVector,
    X::SymArray{TX,N,K}, M::SymCPD,
    loss,
    iN_lo::Int, iN_hi::Int, vec_idx_start::Int,
    ::Val{N}, ::Val{R}
) where {TX,N,K,R}
    set_partial_m = map(1:R) do j
        terms = [:(M.U[M.S[$k]][$(Symbol("i_$(k)")), $j]) for k in 2:N]
        :(partial_prod[$j] = M.λ[$j] * *( $(terms...) ))
    end
    pre_body = Expr(:block, set_partial_m...)

    quote
        S = M.S
        T = eltype(M.U[1])
        partial_prod = zeros(MVector{$R, T})
        mode1_factors = M.U[S[1]]
        X_vals = X.vals
        vec_idx = vec_idx_start
        @inbounds @nloops(
            $N,
            i,
            k -> (k == $N ? iN_lo : S[k] == S[k+1] ? i_{k+1} : 1):(k == $N ? iN_hi : size(M.U[S[k]], 1)),
            d -> d == 2 ? $pre_body : nothing,
            begin
                x = X_vals[vec_idx]
                m = zero(T)
                for col in 1:$R
                    m = muladd(mode1_factors[i_1, col], partial_prod[col], m)
                end 
                partial_Y_vec[vec_idx] = ismissing(x) ? zero(nonmissingtype(eltype(X))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
                vec_idx += 1
            end
        )
    end
end

function columnwise_ttv_all_modes_multithread!(
    result::AbstractVector,
    y::SymArray{TY,N,K}, Us::NTuple{K,AbstractMatrix{TU}},
    multinomial_coefs::Vector{TC},
    iN_starts::Vector{Int}, vec_idx_starts::Vector{Int},
    num_threads::Int,
    ::Val{S}, ::Val{N}, ::Val{R}
) where {TY,N,K,TU,TC,S,R}
    _check_sym_factors(y, Us, S)

    nchunks = min(num_threads, length(iN_starts)) 

    UsT  = ntuple(k -> permutedims(Us[k]), Val(K))
    partial_results = [t == 1 ? result : zeros(TU, R) for t in 1:nchunks]

    Threads.@threads for t in 1:nchunks
        lo = iN_starts[t]
        hi = t == nchunks ? size(UsT[K], 2) : iN_starts[t+1] - 1
        v0 = vec_idx_starts[t]
        _columnwise_ttv_all_modes_chunk!(partial_results[t], y, UsT, multinomial_coefs, lo, hi, v0,
                    Val(S), Val(N), Val(R))
    end
    for t in 2:nchunks
        @inbounds for I in eachindex(result)
            result[I] += partial_results[t][I]
        end
    end
    return result
end

@generated function _columnwise_ttv_all_modes_chunk!(
    result::AbstractVector{TU},
    y::SymArray{TY,N,K},
    Us_T::NTuple{K,AbstractMatrix{TU}},
    multinomial_coefs::Vector{TC},
    iN_lo::Int, iN_hi::Int, vec_idx_start::Int,
    ::Val{S}, ::Val{N}, ::Val{R}
) where {TY,N,K,TU,TC,S,R}

    # Define functions for different symbols in expressions
    Iv(d) = Symbol(:i_, d);  Partial_Prod(d) = Symbol(:partial_prod_, d)
    UM(d) = Symbol(:Ut_, S[d])        # transposed factor at loop position d

    pre_exprs  = [Any[] for _ in 1:N]
    post_exprs = [Any[] for _ in 1:N]

    # Add pre-expression at each level for partial products
    for d in 2:N
        push!(pre_exprs[d], :(for col in 1:$R
            $(Partial_Prod(d))[col] = $(d == N ? :($(UM(d))[col, $(Iv(d))]) :
                              :($(UM(d))[col, $(Iv(d))] * $(Partial_Prod(d+1))[col]))
        end))
    end
    
    # Add pre/post expressions for zeroing/flushing the accumulator
    push!(pre_exprs[2],  :(fill!(acc, zero($TU))))
    push!(post_exprs[2], :(for col in 1:$R
                               result[col] = muladd(acc[col], partial_prod_2[col], result[col])
                           end))

    chain(exprs) = Expr(:->, :d, Expr(:block, foldr((k, rest) -> isempty(exprs[k]) ? rest :
                 Expr(:if, :(d == $k), Expr(:block, exprs[k]...), rest),
                 1:N; init = :nothing)))
    pre, post = chain(pre_exprs), chain(post_exprs)

    alloc = Expr(:block, (:($(Partial_Prod(d)) = zeros(MVector{$R, $TU})) for d in 2:N)...)
    
    quote
        Sv = $(S)
        acc = zeros(MVector{$R, TU})
        @nexprs $K k -> (Ut_k = Us_T[k])
        Ufirst = $(UM(1))
        $alloc
        y_vals = y.vals
        vec_idx = vec_idx_start
        @inbounds @nloops(
            $N,
            i,
            k -> (k == $N ? iN_lo : Sv[k] == Sv[k+1] ? i_{k+1} : 1):(k == $N ? iN_hi : size(Us_T[Sv[k]], 2)),
            $pre,
            $post,
            begin
                tensor_term = multinomial_coefs[vec_idx] * y_vals[vec_idx]
                for col in 1:$R
                    acc[col] = muladd(tensor_term, Ufirst[col, i_1], acc[col])  # Note the product of terms from modes 2-N is factored out into post-expr for i_2
                end
                vec_idx += 1
            end
        )
        return result
    end
end

function columnwise_ttv_all_modes_except_one_multithread!(
    result::AbstractMatrix,
    y::SymArray{TY,N,K}, Us::NTuple{K,AbstractMatrix{TU}},
    multinomial_coefs::Vector{TC},
    iN_starts::Vector{Int}, vec_idx_starts::Vector{Int},
    num_threads::Int,
    ::Val{S}, ::Val{c}, ::Val{N}, ::Val{R}
) where {TY,N,K,TU,TC,S,c,R}
    _check_sym_factors(y, Us, S)

    nchunks = min(num_threads, length(iN_starts))

    UsT  = ntuple(k -> permutedims(Us[k]), Val(K))
    partial_results = [t == 1 ? result : zeros(TU, R, size(result,2)) for t in 1:nchunks]

    Threads.@threads for t in 1:nchunks
        lo = iN_starts[t]
        hi = t == nchunks ? size(UsT[K], 2) : iN_starts[t+1] - 1
        v0 = vec_idx_starts[t]
        _columnwise_ttv_all_modes_except_one_chunk!(partial_results[t], y, UsT, multinomial_coefs, lo, hi, v0,
                    Val(S), Val(c), Val(N), Val(R))
    end
    for t in 2:nchunks
        @inbounds for I in eachindex(result)
            result[I] += partial_results[t][I]
        end
    end
    return result
end


# TTV in all modes except one with outer loop chunk defined by iN_lo, iN_hi,
# for multithreading across outer loop.
@generated function _columnwise_ttv_all_modes_except_one_chunk!(
    result::AbstractMatrix,
    y::SymArray{TY,N,K},
    Us_T::NTuple{K,AbstractMatrix{TU}},
    multinomial_coefs::Vector{TC},
    iN_lo::Int, iN_hi::Int, vec_idx_start::Int,
    ::Val{S}, ::Val{c}, ::Val{N}, ::Val{R}
) where {TY,N,K,TU,TC,S,c,R}

    num_modes_cell = count(==(c), S)
    mode = findfirst(==(c), S)
    last_mode_cell = mode + num_modes_cell - 1
    acc_modes = [j + mode - 1 for j in 1:num_modes_cell if j + mode - 1 >= 2]

    need_pref = last_mode_cell >= 3  # Flag whether we can save multiplies by computing prefix products
    x_needed  = sort!(unique!([collect(2:last_mode_cell-1); collect(mode+1:N)])) # Included U in prefix/suffix products, excluding mode 1
    peel = (mode == 1 && num_modes_cell >= 2)  # Flag whether to peel first i_1 iteration
    
    # Define functions for different symbols in expressions
    Iv(d)=Symbol(:i_,d);  Pv(d)=Symbol(:p_,d);   Av(d)=Symbol(:acc_,d)
    Xv(d)=Symbol(:x_,d);  SUF(d)=Symbol(:suf_,d); SP(d)=Symbol(:sp_,d)
    UM(d)=Symbol(:U_, S[d])  # transposed factor matrix at loop position d, defined in quote
    IDiff(d)=Symbol(:idiff_,d) # tracks whether i_d == i_{d+1}, hoisted to level d to reduce branching in innermost loop
    col_assign_loop(lhs, rhs) = :(for col in 1:$R; $lhs = $rhs; end)

    pre_exprs  = [Any[] for _ in 1:N]
    post_exprs = [Any[] for _ in 1:N]

    # Add pre-expressions to hoist checking whether indices in cell are equal outside of innermost loop.
    # Note that for the case when mode == 1 and last_mode_cell >= 2,
    # we peel the i_1 == i_2 iteration of the innermost loop, so the condition for j == 2 is always true.
    for d in max(mode, 2)+1:last_mode_cell
        push!(pre_exprs[d-1], :($(IDiff(d)) = $(Iv(d)) != $(Iv(d-1))))
    end
    
    # Add pre- and post-expressions for each level
    for d in 1:N
        # Load row i_d of matrix U[S[d]] into x_d (pre)
        d in x_needed && push!(pre_exprs[d],
            col_assign_loop(:($(Xv(d))[col]), :($(UM(d))[col, $(Iv(d))])))

        # Compute suffix product x_d * ... * x_N (pre) if it will be used
        if d > mode
            push!(pre_exprs[d], col_assign_loop(:($(SUF(d))[col]),
                d == N ? :($(Xv(d))[col]) : :($(Xv(d))[col] * $(SUF(d+1))[col])))
        end

        # Compute coefficient p_d, the multiplicity of i_d among i_j for j in d:last_mode_cell (pre)
        if d >= max(mode, 2) && d <= last_mode_cell
            push!(pre_exprs[d], d == last_mode_cell ? :($(Pv(d)) = 1) :
                :($(Pv(d)) = ifelse($(Iv(d)) == $(Iv(d+1)), $(Pv(d+1)) + 1, 1)))
        end

        # For level d = 2, compute i_1 loop-invariant product sp_D = p_D * prod_{k=2...N, k != D} x_k,
        # i.e., the leave-one-out-product excluding x_1, but including the coefficient p_D,
        # for D in {2,..,last_mode_cell} if D >= mode (pre).
        # Also compute prefix product x_2 * ... * x_{last_mode_cell - 1} (pre).
        # For d >= mode + 1, i_d == i_{d-1}, we assign zeros to sp_d. This prevents us from having to 
        # do branching in the innermost loop, at a minimal cost of added computation from adding zeros.
        if d == 2
            need_pref && push!(pre_exprs[2], :(fill!(pref, one($TU))))
            for D in 2:last_mode_cell
                if D >= mode
                    fac = Any[Pv(D)]
                    D > 2 && push!(fac, :(pref[col]))
                    D < N && push!(fac, :($(SUF(D+1))[col]))
                    prod_ex = length(fac) == 1 ? only(fac) : Expr(:call, :*, fac...)
                    rhs = D in max(mode,2)+1:last_mode_cell ? :(ifelse($(IDiff(D)), $prod_ex, zero($TU))) : prod_ex
                    push!(pre_exprs[2], col_assign_loop(:($(SP(D))[col]), rhs))
                    # push!(pre_exprs[2], col_assign_loop(:($(SP(D))[col]),
                    #     length(fac) == 1 ? only(fac) : Expr(:call, :*, fac...)))
                end
                D < last_mode_cell && push!(pre_exprs[2],
                    col_assign_loop(:(pref[col]), :(pref[col] * $(Xv(D))[col])))
            end
        end

        # Zero (pre) and flush (post) accumulators
        if d in acc_modes
            # !(peel && d == 2) && push!(pre_exprs[d], :(fill!($(Av(d)), zero($TU)))) # If peel && d == 2 we directly overwrite acc_2
            # The fill in the above line causes major slowdowns for odd R, the below is better.
            !(peel && d == 2) && push!(pre_exprs[d], :(for col in 1:$R
                                    $(Av(d))[col] = zero($TU)
                                end))    # If peel && d == 2 we directly overwrite acc_2
            push!(post_exprs[d], :(for col in 1:$R
                                    result[col, $(Iv(d))] += $(Av(d))[col]
                                end))
        end
    end

    # Peel off first iteration of i_1 loop (i.e., i_1 = i_2) when mode == 1 and num_modes_cell >= 2 to reduce branching
    # and coefficient computation logic (pre).
    if peel
        push!(pre_exprs[2], :(yw = multinomial_coefs[vec_idx] * y_vals[vec_idx]))
        push!(pre_exprs[2], :(tt = (p_2 + 1) * yw))
        push!(pre_exprs[2], col_assign_loop(:(acc_2[col]), :(tt * suf_2[col])))
        if last_mode_cell >= 3
            push!(pre_exprs[2], col_assign_loop(:(yx1[col]), :(yw * x_2[col])))
            for D in 3:last_mode_cell
                push!(pre_exprs[2], Expr(:if, :($(Iv(D)) != $(Iv(D-1))),
                    col_assign_loop(:($(Av(D))[col]),
                        :(muladd(yx1[col], $(SP(D))[col], $(Av(D))[col])))))
            end
        end
        push!(pre_exprs[2], :(vec_idx += 1))
    end

    chain(exprs) = Expr(:->, :d, Expr(:block, foldr((k, rest) -> isempty(exprs[k]) ? rest :
                 Expr(:if, :(d == $k), Expr(:block, exprs[k]...), rest),
                 1:N; init = :nothing)))
    pre, post = chain(pre_exprs), chain(post_exprs)

    # Expressions for allocations
    alloc = Expr(:block)
    for d in x_needed;  push!(alloc.args, :($(Xv(d))  = zeros(MVector{$R, $TU}))) end
    for d in mode+1:N;  push!(alloc.args, :($(SUF(d)) = zeros(MVector{$R, $TU}))) end
    for D in acc_modes; push!(alloc.args, :($(Av(D))  = zeros(MVector{$R, $TU}))) end
    for D in acc_modes; push!(alloc.args, :($(SP(D))  = zeros(MVector{$R, $TU}))) end
    need_pref           && push!(alloc.args, :(pref = zeros(MVector{$R, $TU})))
    !isempty(acc_modes) && push!(alloc.args, :(yx1  = zeros(MVector{$R, $TU})))

    # Expressions for loop body
    # Compute value of p_1*yw if mode != 1
    tt1_def = peel                ? :(tt1 = yw) :
              mode != 1           ? :nothing :
              num_modes_cell == 1 ? :(tt1 = yw) :
                                    :(tt1 = ifelse(i_1 == i_2, p_2 + 1, 1) * yw)

    yx1_def = isempty(acc_modes)  ? :nothing :
              :(for col in 1:$R
                    yx1[col] = yw * Ufirst[col, i_1]
                end)

    quote
        Sv = $(S)
        #@nexprs $K k -> (U_k = permutedims(Us[k])) - permutedims in calling function instead
        @nexprs $K k -> (U_k = Us_T[k])
        Ufirst = $(Symbol(:U_, S[1]))
        $alloc
        y_vals = y.vals
        vec_idx = vec_idx_start
        @inbounds @nloops(
            $N, 
            i, 
            k -> (k == $N ? iN_lo :
                    k == 1 ? $(peel ? :(i_2 + 1) : (S[1] == S[2] ? :(i_2) : 1)) :
                    Sv[k] == Sv[k+1] ? i_{k+1} : 1) :
                    (k == $N ? iN_hi : size(Us_T[Sv[k]], 2)),
            $pre, 
            $post, 
            begin   # Body expr
                yw = multinomial_coefs[vec_idx] * y_vals[vec_idx]
                $tt1_def
                $yx1_def
                @nexprs $num_modes_cell j -> begin
                    # if (j == 1 ? true : i_{j+$mode-1} != i_{j+$mode-2})
                    if j > $(2 - mode)
                        for col in 1:$R
                            acc_{j+$mode-1}[col] = muladd(yx1[col], sp_{j+$mode-1}[col], acc_{j+$mode-1}[col])
                        end
                    else
                        for col in 1:$R
                            result[col, i_1] = muladd(tt1, suf_2[col], result[col, i_1])
                        end
                    end
                    # end
                end
                vec_idx += 1
            end
        )
        return result
    end
end