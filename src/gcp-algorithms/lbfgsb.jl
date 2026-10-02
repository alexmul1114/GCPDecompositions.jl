## Algorithm: LBFGSB

"""
    LBFGSB

**L**imited-memory **BFGS** with **B**ox constraints.

Brief description of algorithm parameters:

  - `m::Int`         : max number of variable metric corrections (default: `10`)
  - `factr::Float64` : function tolerance in units of machine epsilon (default: `1e7`)
  - `pgtol::Float64` : (projected) gradient tolerance (default: `1e-5`)
  - `maxfun::Int`    : max number of function evaluations (default: `15000`)
  - `maxiter::Int`   : max number of iterations (default: `15000`)
  - `iprint::Int`    : verbosity (default: `-1`)
      + `iprint < 0` means no output
      + `iprint = 0` prints only one line at the last iteration
      + `0 < iprint < 99` prints `f` and `|proj g|` every `iprint` iterations
      + `iprint = 99` prints details of every iteration except n-vectors
      + `iprint = 100` also prints the changes of active set and final `x`
      + `iprint > 100` prints details of every iteration including `x` and `g`

See documentation of [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl) for more details.
"""
Base.@kwdef struct LBFGSB <: AbstractAlgorithm
    m::Int         = 10
    factr::Float64 = 1e7
    pgtol::Float64 = 1e-5
    maxfun::Int    = 15000
    maxiter::Int   = 15000
    iprint::Int    = -1
end

function _gcp(
    X::Array{TX,N},
    r,
    loss,
    constraints::Tuple{Vararg{GCPConstraints.LowerBound}},
    algorithm::GCPAlgorithms.LBFGSB,
    init,
) where {TX,N}
    # T = promote_type(nonmissingtype(TX), Float64)
    T = Float64    # LBFGSB.jl seems to only support Float64

    # Compute lower bound from constraints
    lower = maximum(constraint.value for constraint in constraints; init = T(-Inf))

    # Error for unsupported loss/constraint combinations
    dom = GCPLosses.domain(loss)
    if dom == Interval(-Inf, +Inf)
        lower in (-Inf, 0.0) || error(
            "only lower bound constraints of `-Inf` or `0` are (currently) supported for loss functions with a domain of `-Inf .. Inf`",
        )
    elseif dom == Interval(0.0, +Inf)
        lower == 0.0 || error(
            "only lower bound constraints of `0` are (currently) supported for loss functions with a domain of `0 .. Inf`",
        )
    else
        error(
            "only loss functions with a domain of `-Inf .. Inf` or `0 .. Inf` are (currently) supported",
        )
    end

    # Initialization
    M0 = deepcopy(init)
    u0 = vcat(vec.(M0.U)...)

    # Setup vectorized objective function and gradient
    vec_cutoffs = (0, cumsum(r .* size(X))...)
    vec_ranges = ntuple(k -> vec_cutoffs[k]+1:vec_cutoffs[k+1], Val(N))
    function f(u)
        U = map(range -> reshape(view(u, range), :, r), vec_ranges)
        return GCPLosses.objective(CPD(ones(T, r), U), X, loss)
    end
    function g!(gu, u)
        U = map(range -> reshape(view(u, range), :, r), vec_ranges)
        GU = map(range -> reshape(view(gu, range), :, r), vec_ranges)
        GCPLosses.grad_U!(GU, CPD(ones(T, r), U), X, loss)
        return gu
    end

    # Run LBFGSB
    lbfgsopts = (; (pn => getproperty(algorithm, pn) for pn in propertynames(algorithm))...)
    u = lbfgsb(f, g!, u0; lb = fill(lower, length(u0)), lbfgsopts...)[2]
    U = map(range -> reshape(u[range], :, r), vec_ranges)
    return CPD(ones(T, r), U)
end

"""
    _symgcp(X::Array{TX,N}, S::NTuple{N,Int}, loss, constraints::Tuple{Vararg{GCPConstraints.LowerBound}}, 
                        algorithm::GCPAlgorithms.LBFGSB, init, γ; sym_data=false, symmetrize_data=false, sym_grad_threads=1) 

Run LBFGS to compute a symmetric decomposition using SymGCP.
If `sym_data` is `true`, the data is assumed be symmetric (with the same symmetry as specified by S),
and we use the efficient symmetric gradient and objective algorithms.
If `sym_data` is `false` and `symmetrize_data` is true, we first create a symmetrized copy of the data,
and use the symmetrized data for the efficient gradient and objective algorithms (assuming loss function is affine in data).
If `sym_data` and `symmetrize_data` are both false, we use the nonsymmetric gradient and objective algorithms.
If `sym_grad_threads` is greater than 1, multithreaded version of symmetric gradient function is used.
"""
function _symgcp(
    X::Array{TX,N},
    S::NTuple{N,Int},
    loss,
    constraints::Tuple{Vararg{GCPConstraints.LowerBound}},
    algorithm::GCPAlgorithms.LBFGSB,
    init,
    γ;
    sym_data=false,
    symmetrize_data=false,
    sym_grad_threads=1,
) where {TX,N}

    # Error for unsupported combination of keyword args
    sym_grad_threads > 1 && !(sym_data || symmetrize_data) && error(
        "multithreaded symmetric gradient requires sym_data=true or symmetrize_data=true"
    )

    # Warn when number of threads is greater than number of available threads
    sym_grad_threads <= Threads.nthreads() || @warn "sym_grad_threads is greater than the total number of available threads"

    T = Float64    # LBFGSB.jl seems to only support Float64
    r = ncomps(init)

    # Compute lower bound from constraints
    lower = maximum(constraint.value for constraint in constraints; init = T(-Inf))

    # Error for unsupported loss/constraint combinations
    dom = GCPLosses.domain(loss)
    if dom == Interval(-Inf, +Inf)
        lower in (-Inf, 0.0) || error(
            "only lower bound constraints of `-Inf` or `0` are (currently) supported for loss functions with a domain of `-Inf .. Inf`",
        )
    elseif dom == Interval(0.0, +Inf)
        lower == 0.0 || error(
            "only lower bound constraints of `0` are (currently) supported for loss functions with a domain of `0 .. Inf`",
        )
    else
        error(
            "only loss functions with a domain of `-Inf .. Inf` or `0 .. Inf` are (currently) supported",
        )
    end

    use_symmetric_algs = sym_data || symmetrize_data

    # Begin timing
    # t0 = time_ns()

    # Symmetrize data if selected
    Xsym = !sym_data && symmetrize_data ? symmetrize_tensor(X, S) : sym_data ? X : nothing

    # Initialization
    M0 = deepcopy(init)
    u_λ_0 = vcat(vec.(M0.U)..., M0.λ)
    K = ngroups(M0)

    # Create gradient buffers
    grad_buffers = use_symmetric_algs ? GCPLosses.create_symgcp_sym_grad_buffers(M0) : GCPLosses.create_symgcp_nonsym_grad_buffers(X, M0)

    if use_symmetric_algs
        cell_sizes = ntuple(k -> size(M0.U[k],1), K)
        multinomial_coefs = collect_multinomial_coefficients(M0.S, cell_sizes, Val(N))
    end

    # Setup vectorized objective function and gradient
    vec_cutoffs = (0, (cumsum(r .* tuple([size(M0.U[k])[1] for k in 1:K]...))...), sum((length(M0.U[k]) for k in 1:K)) + r)
    vec_ranges = ntuple(k -> vec_cutoffs[k]+1:vec_cutoffs[k+1], Val(K+1))

    if sym_grad_threads > 1
        iN_starts, vec_idx_starts = ttv_threading_plan(M0.S, cell_sizes, sym_grad_threads, Val(ndims(X)));
    end

    setting = use_symmetric_algs ? (sym_grad_threads > 1 ? Val(:SymThreaded) : Val(:Sym)) : Val(:NonSym)
    data = sym_grad_threads > 1 ? (Xsym, multinomial_coefs, sym_grad_threads, iN_starts, vec_idx_starts) : use_symmetric_algs ? (Xsym, multinomial_coefs) : (X,)
    Mfinal = _symgcp_lbfgsb(setting, data, grad_buffers, u_λ_0, loss, γ, lower, algorithm, vec_ranges, r, S, Val(r))
    
    # elapsed_time = (time_ns() - t0) / 1e9  # Return total time in seconds
    # final_loss = GCPLosses.objective_nonsymdata(Mfinal, X, loss, γ, Val(r))

    # return (M=Mfinal, loss=final_loss, time=elapsed_time)
    return Mfinal
end

function _symgcp_lbfgsb(setting, data, grad_buffers, u_λ_0, loss, γ, lower, algorithm, vec_ranges, r, S, ::Val{R}) where {R}
    function f(u_λ)
        U = map(range -> reshape(view(u_λ, range), :, r), vec_ranges[1:length(vec_ranges)-1])
        λ = view(u_λ, vec_ranges[length(vec_ranges)])
        return _objective(setting, SymCPD(λ, U, S), data, loss, γ, Val(R))
    end
    function g!(gu_λ, u_λ)
        U = map(range -> reshape(view(u_λ, range), :, r), vec_ranges[1:length(vec_ranges)-1])
        λ = view(u_λ, vec_ranges[length(vec_ranges)])
        GU = map(range -> reshape(view(gu_λ, range), :, r), vec_ranges[1:length(vec_ranges)-1])
        Gλ = view(gu_λ, vec_ranges[length(vec_ranges)])
        _grad!(setting, (GU..., Gλ), SymCPD(λ, U, S), data, loss, γ, grad_buffers)
        return gu_λ 
    end

    # Run LBFGSB
    lbfgsopts = (; (pn => getproperty(algorithm, pn) for pn in propertynames(algorithm))...)
    u_λ = lbfgsb(f, g!, u_λ_0; lb = fill(lower, length(u_λ_0)), lbfgsopts...)[2]

    U = map(range -> reshape(u_λ[range], :, r), vec_ranges[1:length(vec_ranges)-1])
    λ = u_λ[vec_ranges[length(vec_ranges)]]

    return SymCPD(λ, U, S)
end
_objective(::Val{:NonSym}, M, (X,), loss, γ, valR) = GCPLosses.objective_nonsymdata(M, X, loss, γ, valR)
_objective(::Union{Val{:Sym},Val{:SymThreaded}}, M, (Xsym, coefs), loss, γ, valR) = GCPLosses.objective_symdata(M, Xsym, loss, γ, coefs, valR)
_grad!(::Val{:NonSym}, GUλ, M, (X,), loss, γ, buffers) = 
    GCPLosses.symgcp_nonsym_mttkrp_grad!(GUλ, M, X, loss, γ; sym_data=false, buffers=buffers)
_grad!(::Val{:Sym}, GUλ, M, (Xsym, coefs), loss, γ, buffers) = 
    GCPLosses.symgcp_symmetric_grad!(GUλ, M, Xsym, loss, coefs, γ; buffers=buffers)
_grad!(::Val{:SymThreaded}, GUλ, M, (Xsym, coefs, nthreads, iN_starts, vec_idx_starts), loss, γ, buffers) = 
    GCPLosses.symgcp_symmetric_grad_multithread!(GUλ, M, Xsym, loss, coefs, γ, nthreads, iN_starts, vec_idx_starts; buffers=buffers)