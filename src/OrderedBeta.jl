using Distributions
using Random

# Internal definitions
_log1pexp(x::Real) = log1p(exp(x))
_logbeta(a::Real, b::Real) = lgamma(a) + lgamma(b) - lgamma(a + b)
_logistic(x::Real) = 1 / (1 + exp(-x))


"""
    OrderedBeta(μ, ϕ, k1, k2)

The distribution is defined on the interval [0, 1] with additional point masses at 0 and 1.

# Arguments
- `μ`: location parameter on the scale 0-1
- `ϕ`: precision parameter (must be positive)
- `k1`: first cutpoint (`curzero`)
- `k2`: log of the difference between the second and first cutpoints (`cutone`)

# Details

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/animation_OrderedBeta.gif?raw=true)

The figure above shows the parameter space for *k1* and *k2*, showing the regions that produce a large proportion of zeros and ones (in red).
Understanding this is important to set appropriate priors on these parameters.

# Examples
```jldoctest
julia> OrderedBeta(0.5, 1, -6, 4)
OrderedBeta{Float64}(μ=0.5, ϕ=1.0, k1=-6.0, k2=4.0)
```
"""
struct OrderedBeta{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    ϕ::T
    k1::T
    k2::T

    function OrderedBeta{T}(μ::T, ϕ::T, k1::T, k2::T) where {T<:Real}
        @assert ϕ > 0 "ϕ must be positive"
        new{T}(μ, ϕ, k1, k2)
    end
end

OrderedBeta(μ::T, ϕ::T, k1::T, k2::T) where {T<:Real} = OrderedBeta{T}(μ, ϕ, k1, k2)

function OrderedBeta(μ::Real, ϕ::Real, k1::Real, k2::Real)
    OrderedBeta(promote(μ, ϕ, k1, k2)...)
end

OrderedBeta(; μ::Real=0.5, ϕ::Real=1, k1::Real=-6, k2::Real=4) = OrderedBeta(μ, ϕ, k1, k2)

# Basic ------------------------------------------------------------------------------------------
Distributions.params(d::OrderedBeta) = (d.μ, d.ϕ, d.k1, d.k2)
Distributions.minimum(::OrderedBeta) = 0
Distributions.maximum(::OrderedBeta) = 1
Distributions.insupport(::OrderedBeta, x::Real) = 0 ≤ x ≤ 1

function Distributions.mean(d::OrderedBeta)
    μ, ϕ, k1, k2 = Distributions.params(d)
    thresh = [k1, k1 + exp(k2)]

    # Probabilities for 0, 1, and (0, 1)
    p_0 = 1 - _logistic(μ - thresh[1])
    p_1 = _logistic(μ - thresh[2])
    p_middle = _logistic(μ - thresh[1]) - _logistic(μ - thresh[2])

    # Mean of the Beta distribution
    beta_mean = μ

    # Weighted mean
    return p_0 * 0 + p_middle * beta_mean + p_1 * 1
end

function Distributions.var(d::OrderedBeta)
    μ, ϕ, k1, k2 = Distributions.params(d)
    thresh = [k1, k1 + exp(k2)]

    # Probabilities for 0, 1, and (0, 1)
    p_0 = 1 - _logistic(μ - thresh[1])
    p_1 = _logistic(μ - thresh[2])
    p_middle = _logistic(μ - thresh[1]) - _logistic(μ - thresh[2])

    # Parameters of the Beta distribution using BetaPhi2
    beta_dist = BetaPhi2(μ, ϕ)
    beta_mean = mean(beta_dist)
    beta_var = var(beta_dist)

    # Mean of the OrderedBeta distribution
    orderedbeta_mean = Distributions.mean(d)

    # Variance of the OrderedBeta distribution
    var_orderedbeta = p_0 * (0 - orderedbeta_mean)^2 +
                      p_1 * (1 - orderedbeta_mean)^2 +
                      p_middle * (beta_mean - orderedbeta_mean)^2 +
                      p_middle * beta_var

    return var_orderedbeta
end

# Random -------------------------------------------------------------------------------------------
function Random.rand(rng::Random.AbstractRNG, d::OrderedBeta)
    μ, ϕ, k1, k2 = Distributions.params(d)
    thresh = [k1, k1 + exp(k2)]
    u = Random.rand(rng)

    p_0 = 1 - _logistic(μ - thresh[1])
    p_1 = _logistic(μ - thresh[2])
    p_middle = _logistic(μ - thresh[1]) - _logistic(μ - thresh[2])

    if u < p_0
        return 0.0
    elseif u < p_0 + p_middle
        return Random.rand(rng, BetaPhi2(μ, ϕ))
    else
        return 1.0
    end
end

Random.rand(d::OrderedBeta) = Random.rand(Random.GLOBAL_RNG, d)
Random.rand(rng::Random.AbstractRNG, d::OrderedBeta, n::Int) = [Random.rand(rng, d) for _ in 1:n]
Random.rand(d::OrderedBeta, n::Int) = rand(Random.GLOBAL_RNG, d, n)
Distributions.sampler(d::OrderedBeta) = d

# PDF -------------------------------------------------------------------------------------------
function Distributions.logpdf(d::OrderedBeta, x::Real)
    μ, ϕ, k1, k2 = Distributions.params(d)
    thresh = [k1, k1 + exp(k2)]

    if x == 0
        return log1p(-_logistic(μ - thresh[1]))
    elseif x == 1
        return log(_logistic(μ - thresh[2]))
    elseif 0 < x < 1
        log_p_middle = log(_logistic(μ - thresh[1]) - _logistic(μ - thresh[2]))
        return log_p_middle + logpdf(BetaPhi2(μ, ϕ), x)
    else
        return -Inf
    end
end

Distributions.pdf(d::OrderedBeta, x::Real) = exp(Distributions.logpdf(d, x))
Distributions.loglikelihood(d::OrderedBeta, x::Real) = Distributions.logpdf(d, x)

function Distributions.cdf(d::OrderedBeta, x::Real)
    μ, ϕ, k1, k2 = Distributions.params(d)
    thresh = [k1, k1 + exp(k2)]

    if x <= 0
        return zero(μ)
    elseif x >= 1
        return one(μ)
    else
        p_0 = 1 - _logistic(μ - thresh[1])
        p_middle = _logistic(μ - thresh[1]) - _logistic(μ - thresh[2])
        return p_0 + p_middle * Distributions.cdf(BetaPhi2(μ, ϕ), x)
    end
end

function Distributions.quantile(d::OrderedBeta, q::Real)
    0 <= q <= 1 || throw(DomainError(q, "quantile must be in [0, 1]"))
    μ, ϕ, k1, k2 = Distributions.params(d)
    thresh = [k1, k1 + exp(k2)]

    p_0 = 1 - _logistic(μ - thresh[1])
    p_1 = _logistic(μ - thresh[2])

    if q <= p_0
        return 0.0
    elseif q >= 1 - p_1
        return 1.0
    else
        p_middle = _logistic(μ - thresh[1]) - _logistic(μ - thresh[2])
        q_adjusted = (q - p_0) / p_middle
        return Distributions.quantile(BetaPhi2(μ, ϕ), q_adjusted)
    end
end
