## Loss function types

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

# Statistically motivated losses

"""
    LeastSquaresLoss()

Loss corresponding to conventional CP decomposition.
Corresponds to a statistical assumption of Gaussian data `X`
with mean given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\mathcal{N}(\\mu_i, \\sigma)``
  - **Link function:** ``m_i = \\mu_i``
  - **Loss function:** ``f(x,m) = (x-m)^2``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct LeastSquaresLoss <: AbstractLoss end
value(::LeastSquaresLoss, x, m) = (x - m)^2
deriv(::LeastSquaresLoss, x, m) = 2 * (m - x)
domain(::LeastSquaresLoss) = Interval(-Inf, +Inf)

"""
    NonnegativeLeastSquaresLoss()

Loss corresponding to nonnegative CP decomposition.
Corresponds to a statistical assumption of Gaussian data `X`
with nonnegative mean given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\mathcal{N}(\\mu_i, \\sigma)``
  - **Link function:** ``m_i = \\mu_i``
  - **Loss function:** ``f(x,m) = (x-m)^2``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct NonnegativeLeastSquaresLoss <: AbstractLoss end
value(::NonnegativeLeastSquaresLoss, x, m) = (x - m)^2
deriv(::NonnegativeLeastSquaresLoss, x, m) = 2 * (m - x)
domain(::NonnegativeLeastSquaresLoss) = Interval(0.0, Inf)

"""
    PoissonLoss(eps::Real = 1e-10)

Loss corresponding to a statistical assumption of Poisson data `X`
with rate given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\operatorname{Poisson}(\\lambda_i)``
  - **Link function:** ``m_i = \\lambda_i``
  - **Loss function:** ``f(x,m) = m - x \\log(m + \\epsilon)``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct PoissonLoss{T<:Real} <: AbstractLoss
    eps::T
    PoissonLoss{T}(eps::T) where {T<:Real} =
        eps >= zero(eps) ? new(eps) :
        throw(DomainError(eps, "Poisson loss requires nonnegative `eps`"))
end
PoissonLoss(eps::T = 1e-10) where {T<:Real} = PoissonLoss{T}(eps)
value(loss::PoissonLoss, x, m) = m - x * log(m + loss.eps)
deriv(loss::PoissonLoss, x, m) = one(m) - x / (m + loss.eps)
domain(::PoissonLoss) = Interval(0.0, +Inf)

"""
    PoissonLogLoss()

Loss corresponding to a statistical assumption of Poisson data `X`
with log-rate given by the low-rank model tensor `M`.

  - **Distribution:** ``x_i \\sim \\operatorname{Poisson}(\\lambda_i)``
  - **Link function:** ``m_i = \\log \\lambda_i``
  - **Loss function:** ``f(x,m) = e^m - x m``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct PoissonLogLoss <: AbstractLoss end
value(::PoissonLogLoss, x, m) = exp(m) - x * m
deriv(::PoissonLogLoss, x, m) = exp(m) - x
domain(::PoissonLogLoss) = Interval(-Inf, +Inf)

"""
    GammaLoss(eps::Real = 1e-10)

Loss corresponding to a statistical assumption of Gamma-distributed data `X`
with scale given by the low-rank model tensor `M`.

- **Distribution:** ``x_i \\sim \\operatorname{Gamma}(k, \\sigma_i)``
- **Link function:** ``m_i = k \\sigma_i``
- **Loss function:** ``f(x,m) = \\frac{x}{m + \\epsilon} + \\log(m + \\epsilon)``
- **Domain:** ``m \\in [0, \\infty)``
"""
struct GammaLoss{T<:Real} <: AbstractLoss 
  eps::T
  GammaLoss{T}(eps::T) where {T<:Real} =
    eps >= zero(eps) ? new(eps) :
    throw(DomainError(eps, "Gamma loss requires nonnegative `eps`"))
end
GammaLoss(eps::T = 1e-10) where {T<:Real} = GammaLoss{T}(eps)
value(loss::GammaLoss, x, m) = x / (m + loss.eps) + log(m + loss.eps)
deriv(loss::GammaLoss, x, m) = -x / (m + loss.eps)^2 + inv(m + loss.eps)
domain(::GammaLoss) = Interval(0.0, +Inf)

"""
    RayleighLoss(eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Rayleigh data `X`
with sacle given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{Rayleigh}(\\theta_i)``
  - **Link function:** ``m_i = \\sqrt{\\frac{\\pi}{2}\\theta_i}``
  - **Loss function:** ``f(x, m) = 2\\log(m + \\epsilon) + \\frac{\\pi}{4}(\\frac{x}{m + \\epsilon})^2``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct RayleighLoss{T<:Real} <: AbstractLoss 
  eps::T
  RayleighLoss{T}(eps::T) where {T<:Real} =
    eps >= zero(eps) ? new(eps) :
    throw(DomainError(eps, "Rayleigh loss requires nonnegative `eps`"))
end
RayleighLoss(eps::T = 1e-10) where {T<:Real} = RayleighLoss{T}(eps)
value(loss::RayleighLoss, x, m) = 2*log(m + loss.eps) + (pi / 4) * ((x/(m + loss.eps))^2)
deriv(loss::RayleighLoss, x, m) = 2/(m + loss.eps) - (pi / 2) * (x^2 / (m + loss.eps)^3)
domain(::RayleighLoss) = Interval(0.0, +Inf)

"""
    BernoulliOddsLoss(eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Bernouli data `X`
with success rate given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{Bernouli}(\\rho_i)``
  - **Link function:** ``m_i = \\rho / (1 - \\rho)``
  - **Loss function:** ``f(x, m) = \\log(m + 1) - x\\log(m + \\epsilon)``
  - **Domain:** ``m \\in [0, \\infty)``
"""
struct BernoulliOddsLoss{T<:Real} <: AbstractLoss 
  eps::T
  BernoulliOddsLoss{T}(eps::T) where {T<:Real} =
    eps >= zero(eps) ? new(eps) :
    throw(DomainError(eps, "BernoulliOddsLoss loss requires nonnegative `eps`"))
end
BernoulliOddsLoss(eps::T = 1e-10) where {T<:Real} = BernoulliOddsLoss{T}(eps)
value(loss::BernoulliOddsLoss, x, m) = log(m + 1) - x * log(m + loss.eps)
deriv(loss::BernoulliOddsLoss, x, m) = 1 / (m + 1) - (x / (m + loss.eps))
domain(::BernoulliOddsLoss) = Interval(0.0, +Inf)


"""
    BernoulliLogitLoss(eps::Real = 1e-10)

Loss corresponding to the statistical assumption of Bernouli data `X`
with log-success rate given by the low-rank model tensor `M`

  - **Distribution:** ``x_i \\sim \\operatorname{Bernouli}(\\rho_i)``
  - **Link function:** ``m_i = \\log(\\rho_i / (1 - \\rho_i))``
  - **Loss function:** ``f(x, m) = \\log(1 + e^m) - xm``
  - **Domain:** ``m \\in \\mathbb{R}``
"""
struct BernoulliLogitLoss{T<:Real} <: AbstractLoss 
  eps::T
  BernoulliLogitLoss{T}(eps::T) where {T<:Real} =
    eps >= zero(eps) ? new(eps) :
    throw(DomainError(eps, "BernoulliLogitsLoss loss requires nonnegative `eps`"))
end
BernoulliLogitLoss(eps::T = 1e-10) where {T<:Real} = BernoulliLogitLoss{T}(eps)
value(::BernoulliLogitLoss, x, m) = log(1 + exp(m)) - x * m
deriv(::BernoulliLogitLoss, x, m) = exp(m) / (1 + exp(m)) - x
domain(::BernoulliLogitLoss) = Interval(-Inf, +Inf)


# User-defined loss

"""
    UserDefinedLoss

Type for user-defined loss functions ``f(x,m)``,
where ``x`` is the data entry and ``m`` is the model entry.

Contains three fields:

 1. `func::Function`   : function that evaluates the loss function ``f(x,m)``
 2. `deriv::Function`  : function that evaluates the partial derivative ``\\partial_m f(x,m)`` with respect to ``m``
 3. `domain::Interval` : `Interval` from IntervalSets.jl defining the domain for ``m``

The constructor is `UserDefinedLoss(func; deriv, domain)`.
If not provided,

  - `deriv` is automatically computed from `func` using forward-mode automatic differentiation
  - `domain` gets a default value of `Interval(-Inf, +Inf)`
"""
struct UserDefinedLoss <: AbstractLoss
    func::Function
    deriv::Function
    domain::Interval
    function UserDefinedLoss(
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
value(loss::UserDefinedLoss, x, m) = loss.func(x, m)
deriv(loss::UserDefinedLoss, x, m) = loss.deriv(x, m)
domain(loss::UserDefinedLoss) = loss.domain
