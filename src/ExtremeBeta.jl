import Distributions: Beta, ContinuousUnivariateDistribution
import Random
import StatsBase: sample, Weights


"""
    ExtremeBeta(μ, ϕ, k0, k1)

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/animation_ExtremeBeta.gif?raw=true)

# Examples
```jldoctest
julia> ExtremeBeta()
ExtremeBeta{Float64}(
μ: 0.5
ϕ: 1.0
k0: 0.0
k1: 0.0
beta_dist: Distributions.Beta{Float64}(α=1.0, β=1.0)
)
```
"""
struct ExtremeBeta{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    ϕ::T
    k0::T
    k1::T
    beta_dist::Beta{T}

    function ExtremeBeta{T}(μ::T, ϕ::T, k0::T, k1::T) where {T<:Real}
        @assert ϕ > 0 "ϕ must be > 0"
        @assert k0 >= 0 "k0 must be >= 0"
        @assert k1 >= 0 "k1 must be >= 0"
        new{T}(μ, ϕ, k0, k1, Beta(μ * 2 * ϕ, 2 * ϕ * (1 - μ)))
    end
end

ExtremeBeta(μ::T, ϕ::T, k0::T, k1::T) where {T<:Real} = ExtremeBeta{T}(μ, ϕ, k0, k1)

function ExtremeBeta(μ::Real=0.5, ϕ::Real=1, k0::Real=0, k1::Real=0)
    T = promote_type(typeof(μ), typeof(ϕ), typeof(k0), typeof(k1))
    ExtremeBeta(T(μ), T(ϕ), T(k0), T(k1))
end

# Methods ------------------------------------------------------------------------------------------
params(d::ExtremeBeta) = (d.μ, d.ϕ, d.k0, d.k1)
minimum(::ExtremeBeta) = 0
maximum(::ExtremeBeta) = 1
insupport(::ExtremeBeta, x::Real) = 0 ≤ x ≤ 1

# This is probably incorrect as it doesn't take into account the zeros and ones:
# mean(d::ExtremeBeta) = logistic(d.μ)
# var(d::ExtremeBeta) = logistic(d.μ) * (1 - logistic(d.μ)) / (1 + d.ϕ)


# Random -------------------------------------------------------------------------------------------
function _logistic(x::Real)
    return 1.0 / (1.0 + exp(-x))
end

function _invlogistic(y::Real)
    if y <= 0 || y >= 1
        error("Input must be between 0 and 1 (exclusive).")
    end
    return log(y / (1.0 - y))
end


# μ=0.5; ϕ=1; k0=0.01; k1=0.01
# d = ExtremeBeta(μ, ϕ, k0, k1)
# y=0.99

function Random.rand(rng::Random.AbstractRNG, d::ExtremeBeta)
    μ, ϕ, k0, k1 = params(d)
    y = Random.rand(rng)

    # Compute Probabilities
    # -------------------------------------
    # P(y=0)
    if k0 == 0
        p0 = 0.0
    else
        p0 = _logistic(_invlogistic(k0) - _invlogistic(y))
    end

    # P(y=1)
    if k1 == 0
        p1 = 0.0
    else
        p1 = _logistic(_invlogistic(y) - _invlogistic(1 - k1))
    end

    # P(y in ]0, 1[)
    p = 1 - (p0 + p1) # P(y in ]0, 1[)

    return sample([0, 1, Random.rand(rng, d.beta_dist)], Weights([p0, p1, p]))
end

Random.rand(d::ExtremeBeta) = rand(Random.GLOBAL_RNG, d)
Random.rand(rng::Random.AbstractRNG, d::ExtremeBeta, n::Int) = [rand(rng, d) for _ in 1:n]
Random.rand(d::ExtremeBeta, n::Int) = rand(Random.GLOBAL_RNG, d, n)

sampler(d::ExtremeBeta) = d