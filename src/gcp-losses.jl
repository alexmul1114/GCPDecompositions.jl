## Loss function types

"""
Loss functions for Generalized CP Decomposition.
"""
module GCPLosses

using ..GCPDecompositions
using ..TensorKernels: mttkrps!, mttkrp, mttkrp!, sparse_mttkrp!, sparse_mttkrps!, checksym, khatrirao
using ..TensorKernels: fill_reduced_Y_vec!, columnwise_ttv_all_modes!, columnwise_ttv_all_modes_except_one!
using ..TensorKernels: fill_reduced_Y_vec_multithreaded!, columnwise_ttv_all_modes_multithread!, columnwise_ttv_all_modes_except_one_multithread!
using IntervalSets: Interval
using LinearAlgebra: mul!, rmul!, Diagonal, norm, dot
using SparseArrayKit: SparseArray, nonzero_keys, nonzero_values
using StaticArrays: MVector, SVector
using Base.Cartesian: @nloops, @ntuple, @ncall, @nexprs, @nref
using Combinatorics: permutations
import ForwardDiff

# Abstract type

"""
    AbstractLoss

Abstract type for GCP loss functions ``f(x,m)``,
where ``x`` is the data entry and ``m`` is the model entry.

Concrete types `ConcreteLoss <: AbstractLoss` should implement:

  - `value(loss::ConcreteLoss, x, m)` that computes the value of the loss function ``f(x,m)``
  - `deriv(loss::ConcreteLoss, x, m)` that computes the value of the partial derivative ``\\partial_m f(x,m)`` with respect to ``m``
  - `domain(loss::ConcreteLoss)` that returns an `Interval` from IntervalSets.jl defining the domain for ``m``
"""
abstract type AbstractLoss end

"""
    value(loss, x, m)

Compute the value of the (entrywise) loss function `loss`
for data entry `x` and model entry `m`.
"""
function value end

"""
    deriv(loss, x, m)

Compute the derivative of the (entrywise) loss function `loss`
at the model entry `m` for the data entry `x`.
"""
function deriv end

"""
    domain(loss)

Return the domain of the (entrywise) loss function `loss`.
"""
function domain end

# Objective function and gradients

"""
    objective(M::CPD, X::AbstractArray, loss)

Compute the GCP objective function for the model tensor `M`, data tensor `X`,
and loss function `loss`.
"""
function objective(M::CPD{T,N}, X::Array{TX,N}, loss) where {T,TX,N}
    return sum(value(loss, X[I], M[I]) for I in CartesianIndices(X) if !ismissing(X[I]))
end

"""
    objective(M::SymCPD, X::Array, loss)

Compute the symmetric GCP objective function for the symmetric model tensor `M`, 
nonsymmetric data tensor `X`, and loss function `loss`, with regularization parameter γ.
"""
@generated function objective_nonsymdata(
    M::SymCPD{TM,N,K}, 
    X::Array{TX,N}, 
    loss, γ,
    ::Val{R}
) where {TM,N,K,TX,R}
    set_partial_m = map(1:R) do j
        terms = [:(M.U[M.S[$k]][$(Symbol("i_$(k)")), $j]) for k in 2:N]
        :(partial_prod[$j] = M.λ[$j] * *( $(terms...) ))
    end
    pre_body = Expr(:block, set_partial_m...)

    quote
        partial_prod = zeros(MVector{$R, TM})
        mode1_factors = M.U[M.S[1]]
        f = zero(promote_type(TM, TX))
        @inbounds @nloops(
            $N,
            i,
            k -> 1:size(M.U[M.S[k]], 1),
            d -> d == 2 ? $pre_body : nothing,
            begin
                x = @nref $N X i
                m = zero(TM)
                for col in 1:$R
                    m = muladd(mode1_factors[i_1, col], partial_prod[col], m)
                end 
                # f += m
                f += value(loss, x, m)
            end
        )
        reg = zero(TX)
        @inbounds for k in 1:K
            for col in 1:R
                reg += (norm(@view M.U[k][:, col])^2 - 1)^2
            end
        end
        f += γ * reg
        return f
    end
end

"""
    objective_symdata(M::SymCPD{T,N,K}, X::Array{TX,N}, loss, γ, multinomial_coefs, ::Val{R}) where {T,N,K,TX,R}

Compute the symmetric GCP objective function for the symmetric model tensor `M`, 
symmetric data tensor `X`, and loss function `loss`,
using multinomial_coefs to rescale individual terms, with regularization parameter γ.
"""
@generated function objective_symdata(
    M::SymCPD{TM,N,K}, 
    X::Array{TX,N}, 
    loss, γ,
    multinomial_coefs,
    ::Val{R}
) where {TM,N,K,TX,R}
    set_partial_m = map(1:R) do j
        terms = [:(M.U[M.S[$k]][$(Symbol("i_$(k)")), $j]) for k in 2:N]
        :(partial_prod[$j] = M.λ[$j] * *( $(terms...) ))
    end
    pre_body = Expr(:block, set_partial_m...)

    quote
        S = M.S
        partial_prod = zeros(MVector{$R, TM})
        mode1_factors = M.U[M.S[1]]
        f = zero(promote_type(TM, TX))
        vec_idx = 1
        @inbounds @nloops(
            $N,
            i,
            k -> (k == $N ? 1 : S[k] == S[k+1] ? i_{k+1} : 1):size(M.U[S[k]], 1),
            d -> d == 2 ? $pre_body : nothing,
            begin
                x = @nref $N X i
                m = zero(TM)
                for col in 1:$R
                    m = muladd(mode1_factors[i_1, col], partial_prod[col], m)
                end 
                f += multinomial_coefs[vec_idx] * value(loss, x, m)
                vec_idx += 1
            end
        )
        reg = zero(TX)
        @inbounds for k in 1:K
            for col in 1:R
                reg += (norm(@view M.U[k][:, col])^2 - 1)^2
            end
        end
        f += γ * reg
        return f
    end
end

"""
    grad_U!(GU, M::CPD, X::AbstractArray, loss)

Compute the GCP gradient with respect to the factor matrices `U = (U[1],...,U[N])`
for the model tensor `M`, data tensor `X`, and loss function `loss`, and store
the result in `GU = (GU[1],...,GU[N])`.
"""
function grad_U!(
    GU::NTuple{N,TGU},
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
) where {T,TX,N,TGU<:AbstractMatrix{T}}
    Y = [
        ismissing(X[I]) ? zero(nonmissingtype(eltype(X))) : deriv(loss, X[I], M[I]) for
        I in CartesianIndices(X)
    ]
    mttkrps!(GU, Y, M.U)
    for k in 1:N
        rmul!(GU[k], Diagonal(M.λ))
    end
    return GU
end

# Option for sym_data=true is provided for benchmarking time for single vs. multiple MTTKRPs.
# In practice, should just use the symmetric grad algorithm for symmetric data.
function symgcp_nonsym_mttkrp_grad!(
	GU_λ::NTuple{V, AbstractArray},
	M::SymCPD{TM,N,K},
	X::Array{TX,N},
	loss,
    γ;
	sym_data=false,
    buffers = create_symgcp_nonsym_grad_buffers(X, M),
) where {V,TM,TX,N,K}

    missing_or_deriv(x, m) = ismissing(x) ? zero(nonmissingtype(typeof(x))) : GCPDecompositions.GCPLosses.deriv(loss, x, m)
    copy!(buffers.Y, convertCPD(M); buffers=buffers.M_array_buffers)
    buffers.Y .= missing_or_deriv.(X, buffers.Y)

    # Weights gradient
    mul!(
        GU_λ[K+1], 
        GCPDecompositions.TensorKernels.khatrirao!(buffers.weight_kr_buffer, [M.U[k] for k in reverse(M.S)]...)', 
        vec(buffers.Y)
    )

	# Factor matrix gradients
    for cell in 1:K
        if sym_data
			buffer_mode = findfirst(M.S .== cell)
            GCPDecompositions.TensorKernels.mttkrp!(
                GU_λ[cell], buffers.Y, 
                tuple([M.U[k] for k in M.S]...), 
                findall(M.S .== cell)[1], 
                buffers.mttkrp_buffers[buffer_mode]
            )
			rmul!(GU_λ[cell], count(M.S .== cell))
	        rmul!(GU_λ[cell], Diagonal(M.λ))
        else
            for (index, mode) in enumerate(findall(M.S .== cell))
                if index == 1  # Overwrite
                    GCPDecompositions.TensorKernels.mttkrp!(
                        GU_λ[cell], buffers.Y, 
                        tuple([M.U[k] for k in M.S]...), 
                        mode, buffers.mttkrp_buffers[mode]
                    )
                else  # Add in-place
                    added_factor = similar(GU_λ[cell])
                    GCPDecompositions.TensorKernels.mttkrp!(
                        added_factor, buffers.Y, 
                        tuple([M.U[k] for k in M.S]...), 
                        mode, buffers.mttkrp_buffers[mode]
                    )
                    GU_λ[cell] .= GU_λ[cell] + added_factor
                end
            end
			rmul!(GU_λ[cell], Diagonal(M.λ))
        end       
        GU_λ[cell] .+= mapslices(x -> 4γ * (norm(x)^2 - 1) * x, M.U[cell]; dims=1)
    end

    return GU_λ
end

function symgcp_symmetric_grad!(
	GU_λ::NTuple{V, AbstractArray}, 
	M::SymCPD{TM,N,K},
    X::Array{TX,N}, 
	loss,
	multinomial_coefs::AbstractVector,
    γ;
    buffers = create_symgcp_sym_grad_buffers(M),
) where {V,TM,TX,N,K}
	
	# Fill reduced derivative tensor
	fill_reduced_Y_vec!(buffers.Y_vec, X, M, loss, Val(N), Val(ncomps(M)))
    
	# Weights gradient
	fill!(GU_λ[K+1], zero(TM))
    columnwise_ttv_all_modes!(GU_λ[K+1], buffers.Y_vec, M.U, multinomial_coefs, Val(M.S), Val(N), Val(ncomps(M)))

	# Factor matrix gradients
	for cell in 1:K
		GU_T = zeros(TM, size(GU_λ[cell], 2), size(GU_λ[cell], 1))
	    columnwise_ttv_all_modes_except_one!(GU_T, buffers.Y_vec, M.U, multinomial_coefs, Val(M.S), Val(cell), Val(N), Val(ncomps(M)))
	    GU_λ[cell] .= permutedims(GU_T)
	    rmul!(GU_λ[cell], Diagonal(M.λ))
        GU_λ[cell] .+= mapslices(x -> 4γ * (norm(x)^2 - 1) * x, M.U[cell]; dims=1)
	end
	
    return GU_λ
end

function symgcp_symmetric_grad_multithread!(
	GU_λ::NTuple{V, AbstractArray}, 
	M::SymCPD{TM,N,K},
    X::Array{TX,N}, 
	loss,
	multinomial_coefs::AbstractVector,
    γ,
    nthreads,
    iN_starts,
    vec_idx_starts;
    buffers = create_symgcp_sym_grad_buffers(M),
) where {V,TM,TX,N,K}
	
	# Fill reduced derivative tensor
	fill_reduced_Y_vec_multithreaded!(buffers.Y_vec, X, M, loss, iN_starts, vec_idx_starts, nthreads)
    
	# Weights gradient
	fill!(GU_λ[K+1], zero(TM))
    columnwise_ttv_all_modes_multithread!(GU_λ[K+1], buffers.Y_vec, M.U, multinomial_coefs, iN_starts, vec_idx_starts, nthreads, Val(M.S), Val(N), Val(ncomps(M)))
    
	# Factor matrix gradients
	for cell in 1:K
		GU_T = zeros(TM, size(GU_λ[cell], 2), size(GU_λ[cell], 1))
        columnwise_ttv_all_modes_except_one_multithread!(GU_T, buffers.Y_vec, M.U, multinomial_coefs, iN_starts, vec_idx_starts, nthreads, Val(M.S), Val(cell), Val(N), Val(ncomps(M)))
	    GU_λ[cell] .= permutedims(GU_T)
	    rmul!(GU_λ[cell], Diagonal(M.λ))
        GU_λ[cell] .+= mapslices(x -> 4γ * (norm(x)^2 - 1) * x, M.U[cell]; dims=1)
	end
	
    return GU_λ
end

function create_symgcp_nonsym_grad_buffers(
    X::AbstractArray{TX,N},
    M::SymCPD{TM,N,K}
) where {TX,N,TM,K}
    # Allocate buffers
    return (;
        Y = similar(M.U[1], size(X)),
        M_array_buffers = GCPDecompositions.create_copy_buffers(convertCPD(M)),
        mttkrp_buffers = [GCPDecompositions.TensorKernels.create_mttkrp_buffer(X, tuple([M.U[k] for k in M.S]...), mode) for mode in 1:N],
        weight_kr_buffer = similar(M.U[1], length(X), ncomps(M))
    )
end

function create_symgcp_sym_grad_buffers(
    M::SymCPD{TM,N,K}
) where {TM,N,K}
    vec_size = prod(k -> prod(i -> size(M.U[k],1)+i-1, 1:count(M.S .== k))÷factorial(count(M.S .== k)), unique(M.S))
    # Allocate buffers
    return (;
        Y_vec = similar(M.U[1], vec_size)
    )
end

"""
    stochastic_grad_U_λ!(GU_λ, M::SymCPD, X::AbstractArray, loss, B)

Compute the SymGCP gradient with respect to the factor matrices `U = (U[1],...,U[N])` and the 
weights `λ` for the model tensor `M`, elements of the data tensor `X` with indices given by B, and loss function `loss`, and store
the result in `GU_λ = (GU[1],...,GU[K], Gλ)`. Simplify gradients for symmetry of model tensor matching 
symmetry of data tensor if sym_data is true. γ controls the strength of the (column-norm - 1) regularization.
    p - number of nonzero elements in batch
    q - number of zero elements in batch
"""
function stochastic_grad_U_λ!(
    GU_λ::Tuple,
    M::SymCPD{T,N,K},
    X::Array{TX,N},
    loss,
    sym_data,
    γ,
    B,
    sampling_strategy;
    p=1,
    q=1
) where {T,TX,N,K}
    
    η = count(!iszero, X)
    ζ = length(X) - η
    ω = length(X)

    # Initialize sparse subsampled derivative tensor
    inds = unique(B)
    Y = SparseArray{T,N}(Dict([(idx, zero(T)) for idx in inds]), size(X))

    # Compute bias-corrected derivatives
    for (i,idx) in enumerate(B)
        if sampling_strategy == "uniform"
            Y[idx] += (ω / length(B)) * deriv(loss, X[idx], M[idx])
        elseif sampling_strategy == "stratified"
            # First p entries of B are nonzeros, remaining q entries are zeros
            if i <= p
                Y[idx] += (η / p) * deriv(loss, X[idx], M[idx])
            else
                Y[idx] += (ζ / q) * deriv(loss, X[idx], M[idx])
            end
        elseif sampling_strategy == "semi-stratified"
            # First p entries of B are nonzeros, remaining q entries are possible zeros
            if i <= p
                Y[idx] += (η / p) * (deriv(loss, X[idx], M[idx]) - deriv(loss, zero(T), M[idx]))
            else
                Y[idx] += (ω / q) * deriv(loss, zero(T), M[idx])
            end
        else
            error(
                "The only supported sampling strategies are uniform and stratified",
            )
        end
    end

    # Factor matrix gradients
    Us = tuple([M.U[k] for k in M.S]...)

    # Compute mttkrp for each mode
    mode_GUs = similar.(Us)
    sparse_mttkrps!(mode_GUs, Y, Us)

    for j in 1:K
        if sym_data
            first_n = findall(M.S .== j)[1]
            GU_λ[j] .= mode_GUs[first_n]
            rmul!(GU_λ[j], count(M.S .== j))
        else
            for (index, mode) in enumerate(findall(M.S .== j))
                if index == 1  # Overwrite
                    GU_λ[j] .= mode_GUs[mode]
                else  # Add in-place
                    GU_λ[j] .+= mode_GUs[mode]
                end
            end
        end
        rmul!(GU_λ[j], Diagonal(M.λ))
        if !iszero(γ)
            GU_λ[j] .+= mapslices(x -> 4γ * (norm(x)^2 - 1) * x, M.U[j]; dims=1)
        end
    end

    # Weights gradient
    inds, vals = nonzero_keys(Y), nonzero_values(Y)
	Uh = reduce(.*, Us[k][getindex.(inds, k), :] for k in eachindex(Us))
    mul!(GU_λ[K+1], Uh', collect(vals))

    return GU_λ
end

# Statistically motivated losses

"""
    LeastSquares()

Loss corresponding to conventional CP decomposition.
Corresponds to a statistical assumption of Gaussian data `X`
with mean given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\mathcal{N}(\\mu_i, \\sigma)``
  - **Link function:** ``m_i = \\mu_i``
  - **Loss function:** ``f(x,m) = (x-m)^2``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct LeastSquares <: AbstractLoss end
value(::LeastSquares, x, m) = (x - m)^2
deriv(::LeastSquares, x, m) = 2 * (m - x)
domain(::LeastSquares) = Interval(-Inf, +Inf)

"""
    NonnegativeLeastSquares()

Loss corresponding to nonnegative CP decomposition.
Corresponds to a statistical assumption of Gaussian data `X`
with nonnegative mean given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\mathcal{N}(\\mu_i, \\sigma)``
  - **Link function:** ``m_i = \\mu_i``
  - **Loss function:** ``f(x,m) = (x-m)^2``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct NonnegativeLeastSquares <: AbstractLoss end
value(::NonnegativeLeastSquares, x, m) = (x - m)^2
deriv(::NonnegativeLeastSquares, x, m) = 2 * (m - x)
domain(::NonnegativeLeastSquares) = Interval(0.0, Inf)

"""
    Poisson(eps::Real = 1e-10)

Loss corresponding to a statistical assumption of Poisson data `X`
with rate given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\operatorname{Poisson}(\\lambda_i)``
  - **Link function:** ``m_i = \\lambda_i``
  - **Loss function:** ``f(x,m) = m - x \\log(m + \\epsilon)``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct Poisson{T<:Real} <: AbstractLoss
    eps::T
    Poisson{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "Poisson loss requires nonnegative `eps`"))
end
Poisson(eps::T = 1e-10) where {T<:Real} = Poisson{T}(eps)
value(loss::Poisson, x, m) = m - x * log(m + loss.eps)
deriv(loss::Poisson, x, m) = one(m) - x / (m + loss.eps)
domain(::Poisson) = Interval(0.0, +Inf)

"""
    PoissonLog()

Loss corresponding to a statistical assumption of Poisson data `X`
with log-rate given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\operatorname{Poisson}(\\lambda_i)``
  - **Link function:** ``m_i = \\log \\lambda_i``
  - **Loss function:** ``f(x,m) = e^m - x m``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct PoissonLog <: AbstractLoss end
value(::PoissonLog, x, m) = exp(m) - x * m
deriv(::PoissonLog, x, m) = exp(m) - x
domain(::PoissonLog) = Interval(-Inf, +Inf)

"""
    Gamma(eps::Real = 1e-10)

Loss corresponding to a statistical assumption of Gamma-distributed data `X`
with scale given by the low-rank model tensor `M`.

- **Distribution:** ``x_i \\sim \\operatorname{Gamma}(k, \\sigma_i)``
- **Link function:** ``m_i = k \\sigma_i``
- **Loss function:** ``f(x,m) = \\frac{x}{m + \\epsilon} + \\log(m + \\epsilon)``
- **Domain:** ``m \\in [0, \\infty)``
"""
struct Gamma{T<:Real} <: AbstractLoss
    eps::T
    Gamma{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "Gamma loss requires nonnegative `eps`"))
end
Gamma(eps::T = 1e-10) where {T<:Real} = Gamma{T}(eps)
value(loss::Gamma, x, m) = x / (m + loss.eps) + log(m + loss.eps)
deriv(loss::Gamma, x, m) = -x / (m + loss.eps)^2 + inv(m + loss.eps)
domain(::Gamma) = Interval(0.0, +Inf)

"""
    Rayleigh(eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Rayleigh data `X`
with sacle given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{Rayleigh}(\\theta_i)``
  - **Link function:** ``m_i = \\sqrt{\\frac{\\pi}{2}\\theta_i}``
  - **Loss function:** ``f(x, m) = 2\\log(m + \\epsilon) + \\frac{\\pi}{4}(\\frac{x}{m + \\epsilon})^2``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct Rayleigh{T<:Real} <: AbstractLoss
    eps::T
    Rayleigh{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "Rayleigh loss requires nonnegative `eps`"))
end
Rayleigh(eps::T = 1e-10) where {T<:Real} = Rayleigh{T}(eps)
value(loss::Rayleigh, x, m) = 2 * log(m + loss.eps) + (pi / 4) * ((x / (m + loss.eps))^2)
deriv(loss::Rayleigh, x, m) = 2 / (m + loss.eps) - (pi / 2) * (x^2 / (m + loss.eps)^3)
domain(::Rayleigh) = Interval(0.0, +Inf)

"""
    BernoulliOdds(eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Bernouli data `X`
with odds-sucess rate given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{Bernouli}(\\rho_i)``
  - **Link function:** ``m_i = \\frac{\\rho_i}{1 - \\rho_i}``
  - **Loss function:** ``f(x, m) = \\log(m + 1) - x\\log(m + \\epsilon)``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct BernoulliOdds{T<:Real} <: AbstractLoss
    eps::T
    BernoulliOdds{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "BernoulliOdds requires nonnegative `eps`"))
end
BernoulliOdds(eps::T = 1e-10) where {T<:Real} = BernoulliOdds{T}(eps)
value(loss::BernoulliOdds, x, m) = log(m + 1) - x * log(m + loss.eps)
deriv(loss::BernoulliOdds, x, m) = 1 / (m + 1) - (x / (m + loss.eps))
domain(::BernoulliOdds) = Interval(0.0, +Inf)

"""
    BernoulliLogit(eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Bernouli data `X`
with log odds-success rate given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{Bernouli}(\\rho_i)``
  - **Link function:** ``m_i = \\log(\\frac{\\rho_i}{1 - \\rho_i})``
  - **Loss function:** ``f(x, m) = \\log(1 + e^m) - xm``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct BernoulliLogit{T<:Real} <: AbstractLoss
    eps::T
    BernoulliLogit{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "BernoulliLogitsLoss requires nonnegative `eps`"))
end
BernoulliLogit(eps::T = 1e-10) where {T<:Real} = BernoulliLogit{T}(eps)
value(::BernoulliLogit, x, m) = log(1 + exp(m)) - x * m
deriv(::BernoulliLogit, x, m) = exp(m) / (1 + exp(m)) - x
domain(::BernoulliLogit) = Interval(-Inf, +Inf)

"""
    NegativeBinomialOdds(r::Integer, eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Negative Binomial
data `X` with log odds failure rate given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{NegativeBinomial}(r, \\rho_i) ``
  - **Link function:** ``m = \\frac{\\rho}{1 - \\rho}``
  - **Loss function:** ``f(x, m) = (r + x) \\log(1 + m) - x\\log(m + \\epsilon) ``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct NegativeBinomialOdds{S<:Integer,T<:Real} <: AbstractLoss
    r::S
    eps::T
    function NegativeBinomialOdds{S,T}(r::S, eps::T) where {S<:Integer,T<:Real}
        eps >= zero(eps) ||
            throw(DomainError(eps, "NegativeBinomialOdds requires nonnegative `eps`"))
        r >= zero(r) ||
            throw(DomainError(r, "NegativeBinomialOdds requires nonnegative `r`"))
        return new(r, eps)
    end
end
NegativeBinomialOdds(r::S, eps::T = 1e-10) where {S<:Integer,T<:Real} =
    NegativeBinomialOdds{S,T}(r, eps)
value(loss::NegativeBinomialOdds, x, m) = (loss.r + x) * log(1 + m) - x * log(m + loss.eps)
deriv(loss::NegativeBinomialOdds, x, m) = (loss.r + x) / (1 + m) - x / (m + loss.eps)
domain(::NegativeBinomialOdds) = Interval(0.0, +Inf)

"""
    Huber(Δ::Real)

  Huber Loss for given Δ

  - **Loss function:** ``f(x, m) = (x - m)^2 if \\abs(x - m)\\leq\\Delta, 2\\Delta\\abs(x - m) - \\Delta^2 otherwise``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct Huber{T<:Real} <: AbstractLoss
    Δ::T
    Huber{T}(Δ::T) where {T<:Real} =
        Δ >= zero(Δ) ? new(Δ) : throw(DomainError(Δ, "Huber requires nonnegative `Δ`"))
end
Huber(Δ::T) where {T<:Real} = Huber{T}(Δ)
value(loss::Huber, x, m) =
    abs(x - m) <= loss.Δ ? (x - m)^2 : 2 * loss.Δ * abs(x - m) - loss.Δ^2
deriv(loss::Huber, x, m) =
    abs(x - m) <= loss.Δ ? -2 * (x - m) : -2 * sign(x - m) * loss.Δ * x
domain(::Huber) = Interval(-Inf, +Inf)

"""
    BetaDivergence(β::Real, eps::Real)

    BetaDivergence Loss for given β

  - **Loss function:** ``f(x, m; β) = \\frac{1}{\\beta}m^{\\beta} - \\frac{1}{\\beta - 1}xm^{\\beta - 1}
                          if \\beta \\in \\mathbb{R}  \\{0, 1\\},
                            m - x\\log(m) if \\beta = 1,
                            \\frac{x}{m} + \\log(m) if \\beta = 0``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct BetaDivergence{S<:Real,T<:Real} <: AbstractLoss
    β::T
    eps::T
    BetaDivergence{S,T}(β::S, eps::T) where {S<:Real,T<:Real} =
        eps >= zero(eps) ? new(β, eps) :
        throw(DomainError(eps, "BetaDivergence requires nonnegative `eps`"))
end
BetaDivergence(β::S, eps::T = 1e-10) where {S<:Real,T<:Real} = BetaDivergence{S,T}(β, eps)
function value(loss::BetaDivergence, x, m)
    if loss.β == 0
        return x / (m + loss.eps) + log(m + loss.eps)
    elseif loss.β == 1
        return m - x * log(m + loss.eps)
    else
        return 1 / loss.β * m^loss.β - 1 / (loss.β - 1) * x * m^(loss.β - 1)
    end
end
function deriv(loss::BetaDivergence, x, m)
    if loss.β == 0
        return -x / (m + loss.eps)^2 + 1 / (m + loss.eps)
    elseif loss.β == 1
        return 1 - x / (m + loss.eps)
    else
        return m^(loss.β - 1) - x * m^(loss.β - 2)
    end
end
domain(::BetaDivergence) = Interval(0.0, +Inf)

# User-defined loss
"""
    UserDefined

Type for user-defined loss functions ``f(x,m)``,
where ``x`` is the data entry and ``m`` is the model entry.

Contains three fields:

 1. `func::Function`   : function that evaluates the loss function ``f(x,m)``
 2. `deriv::Function`  : function that evaluates the partial derivative ``\\partial_m f(x,m)`` with respect to ``m``
 3. `domain::Interval` : `Interval` from IntervalSets.jl defining the domain for ``m``

The constructor is `UserDefined(func; deriv, domain)`.
If not provided,

  - `deriv` is automatically computed from `func` using forward-mode automatic differentiation
  - `domain` gets a default value of `Interval(-Inf, +Inf)`
"""
struct UserDefined <: AbstractLoss
    func::Function
    deriv::Function
    domain::Interval
    function UserDefined(
        func::Function;
        deriv::Function = (x, m) -> ForwardDiff.derivative(m -> func(x, m), m),
        domain::Interval = Interval(-Inf, Inf),
    )
        hasmethod(func, Tuple{Real,Real}) ||
            error("`func` must accept two inputs `(x::Real, m::Real)`")
        hasmethod(deriv, Tuple{Real,Real}) ||
            error("`deriv` must accept two inputs `(x::Real, m::Real)`")
        return new(func, deriv, domain)
    end
end
value(loss::UserDefined, x, m) = loss.func(x, m)
deriv(loss::UserDefined, x, m) = loss.deriv(x, m)
domain(loss::UserDefined) = loss.domain

end