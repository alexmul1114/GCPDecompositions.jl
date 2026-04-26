## Stochastic GCP objective and gradient functions: Full sampler

"""
    FullGCPSampler(numsamples::Int)

Samples all entries in data tensor. Workaround to make a GPU-compatible gradient descent
algorithm for now, due to issues with LBFGS-B.
"""
struct FullGCPSampler <: AbstractGCPSampler end

function gcp_stoch_objective(
    rng::AbstractRNG,
    M::CPD{T,N},
    X::MtlArray{TX,N},
    loss,
    sampler::FullGCPSampler,
) where {T,TX,N}
    return gcp_stoch_objective(rng, M, X, loss, GCPSampleOnce(X, sampler))
end

GCPSampleOnce(X::MtlArray, sampler::FullGCPSampler) =
    GCPSampleOnce(sampler, Vector{NTuple{ndims(X),Int}}())

GCPSampleOnce(X::Array, sampler::FullGCPSampler) =
    GCPSampleOnce(sampler, Vector{NTuple{ndims(X),Int}}())

function gcp_stoch_objective(
    rng::AbstractRNG,
    M::CPD{T,N},
    X::MtlArray{TX,N},
    loss,
    (; sampler, cache)::GCPSampleOnce{<:FullGCPSampler},
) where {T,TX,N} 
    #return sum(value(loss, X[I], M[I]) for I in CartesianIndices(X) if !ismissing(X[I]))
    return sum(value(loss, X[I], M[I]) for I in CartesianIndices(X))
    #return mapreduce(I -> value(loss, X[I], M[I]), +, CartesianIndices(X))
end

function gcp_stoch_grad_U!(
    rng::AbstractRNG,
    GU::NTuple{N,TGU},
    M::CPD{T,N},
    X::Array{TX,N},
    loss,
    sampler::FullGCPSampler,
) where {T,TX,N,TGU<:AbstractMatrix{T}}
    n, ω, s = size(X), length(X), sampler.numsamples
    inds = sample!(rng, CartesianIndices(n), Vector{NTuple{ndims(X),Int}}(undef, s))
    vals = [
        ismissing(X[CartesianIndex(I)]) ? zero(nonmissingtype(eltype(X))) :
        (ω / s) * deriv(loss, X[CartesianIndex(I)], M[CartesianIndex(I)]) for I in inds
    ]
    Yt = SparseArrayCOO(n, inds, vals)
    mttkrps!(GU, Yt, M.U)
    for k in 1:N
        rmul!(GU[k], Diagonal(M.λ))
    end
    return GU
end