using Distributions
using Random

# Internal definitions
_logistic(x::Real) = 1 / (1 + exp(-x))
_logit(x::Real) = log(x / (1 - x))

"""
    OrderedBeta(μ, ϕ, k1, k2)

This distribution was introduced by [Kubinec (2023)](https://doi.org/10.1017/pan.2022.20) as an appropriate and parsimonious way of describing data commonly observed in psychological science (such as from slider scales). 
It is defined with a Beta distribution on the interval ]0, 1[ with additional point masses at 0 and 1.
The Beta distribution is specified using the [`BetaPhi2`](@ref) parametrization.

# Arguments
- `μ`: location parameter on the scale 0-1
- `ϕ`: precision parameter (must be positive). Note that this parameter is based on the [`BetaPhi2`](@ref) reparametrization of the Beta distribution,
    which corresponds to half the precision of the traditional Beta distribution as implemented in for example the `ordbetareg` package.
- `k1`: first cutpoint (`cutzero`), likely lower than 0.5.
- `k2`: second cutpoint (`cutone`), likely higher than 0.5. Must be greater than `k1`.

# Details

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/animation_OrderedBeta.gif?raw=true)

The figure above shows the parameter space for *k1* and *k2*, showing the regions that produce a large proportion of zeros and ones (in red).
Understanding this is important to set appropriate priors on these parameters.

Compared to the `ordbetareg` R package, the main difference is that:
- *phi* ϕ (Julia version) = *phi* ϕ (R version) / 2
- *k1* and *k2* are specified on the raw scale [0, 1], and independently (in the R package, they are specified on the logit scale and k2 is expressed as a difference from k1).


# Examples
```jldoctest
julia> OrderedBeta(0.5, 1, 0.1, 0.9)
OrderedBeta{Float64}(μ=0.5, ϕ=1.0, k1=0.1, k2=0.9)
```

# References
- Kubinec, R. (2023). Ordered beta regression: a parsimonious, well-fitting model for continuous data with lower and upper bounds. Political analysis, 31(4), 519-536.
"""
struct OrderedBeta{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    ϕ::T
    k1::T
    k2::T

    function OrderedBeta{T}(μ::T, ϕ::T, k1::T, k2::T) where {T<:Real}
        if (ϕ <= 0)
            throw(DomainError(ϕ, "ϕ must be > 0"))
        end
        new{T}(μ, ϕ, k1, k2)
    end
end

OrderedBeta(μ::T, ϕ::T, k1::T, k2::T) where {T<:Real} = OrderedBeta{T}(μ, ϕ, k1, k2)

function OrderedBeta(μ::Real, ϕ::Real, k1::Real, k2::Real)
    OrderedBeta(promote(μ, ϕ, k1, k2)...)
end

OrderedBeta(; μ::Real=0.5, ϕ::Real=1, k1::Real=0.1, k2::Real=0.9) = OrderedBeta(μ, ϕ, k1, k2)

# Basic ------------------------------------------------------------------------------------------
Distributions.params(d::OrderedBeta) = (d.μ, d.ϕ, d.k1, d.k2)
Distributions.minimum(::OrderedBeta) = 0
Distributions.maximum(::OrderedBeta) = 1
Distributions.insupport(::OrderedBeta, x::Real) = 0 ≤ x ≤ 1

function Distributions.mean(d::OrderedBeta)
    μ, ϕ, k1, k2 = Distributions.params(d)
    mu_ql = _logit(μ)
    k1_logit = _logit(k1)
    k2_logit = _logit(k2)

    # Probabilities for 0, 1, and (0, 1)
    p_0 = 1 - _logistic(mu_ql - k1_logit)
    p_1 = _logistic(mu_ql - k2_logit)
    p_middle = _logistic(mu_ql - k1_logit) - _logistic(mu_ql - k2_logit)

    # Mean of the Beta distribution
    beta_mean = μ

    # Weighted mean
    return p_0 * 0 + p_middle * beta_mean + p_1 * 1
end

function Distributions.var(d::OrderedBeta)
    μ, ϕ, k1, k2 = Distributions.params(d)
    mu_ql = _logit(μ)
    k1_logit = _logit(k1)
    k2_logit = _logit(k2)

    # Probabilities for 0, 1, and (0, 1)
    p_0 = 1 - _logistic(mu_ql - k1_logit)
    p_1 = _logistic(mu_ql - k2_logit)
    p_middle = _logistic(mu_ql - k1_logit) - _logistic(mu_ql - k2_logit)

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
    mu_ql = _logit(μ)
    k1_logit = _logit(k1)
    k2_logit = _logit(k2)
    u = Random.rand(rng)

    p_0 = 1 - _logistic(mu_ql - k1_logit)
    p_1 = _logistic(mu_ql - k2_logit)
    p_middle = _logistic(mu_ql - k1_logit) - _logistic(mu_ql - k2_logit)

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
    mu_ql = _logit(μ)
    k1_logit = _logit(k1)
    k2_logit = _logit(k2)

    # Calculate probabilities for the three categories
    p_0 = 1 - _logistic(mu_ql - k1_logit)
    p_middle = _logistic(mu_ql - k1_logit) - _logistic(mu_ql - k2_logit)
    p_1 = _logistic(mu_ql - k2_logit)

    if x == 0
        return log(p_0)
    elseif x == 1
        return log(p_1)
    elseif 0 < x < 1
        if p_middle < 0
            return -Inf
        end
        return log(p_middle) + logpdf(BetaPhi2(μ, ϕ), x)
    else
        return -Inf
    end
end

Distributions.pdf(d::OrderedBeta, x::Real) = exp(Distributions.logpdf(d, x))
Distributions.loglikelihood(d::OrderedBeta, x::Real) = Distributions.logpdf(d, x)

function Distributions.cdf(d::OrderedBeta, x::Real)
    μ, ϕ, k1, k2 = Distributions.params(d)
    mu_ql = _logit(μ)
    k1_logit = _logit(k1)
    k2_logit = _logit(k2)

    if x <= 0
        return zero(μ)
    elseif x >= 1
        return one(μ)
    else
        p_0 = 1 - _logistic(mu_ql - k1_logit)
        p_middle = _logistic(mu_ql - k1_logit) - _logistic(mu_ql - k2_logit)
        return p_0 + p_middle * Distributions.cdf(BetaPhi2(μ, ϕ), x)
    end
end

function Distributions.quantile(d::OrderedBeta, q::Real)
    0 <= q <= 1 || throw(DomainError(q, "quantile must be in [0, 1]"))
    μ, ϕ, k1, k2 = Distributions.params(d)
    mu_ql = _logit(μ)
    k1_logit = _logit(k1)
    k2_logit = _logit(k2)

    p_0 = 1 - _logistic(mu_ql - k1_logit)
    p_1 = _logistic(mu_ql - k2_logit)

    if q <= p_0
        return 0.0
    elseif q >= 1 - p_1
        return 1.0
    else
        p_middle = _logistic(mu_ql - k1_logit) - _logistic(mu_ql - k2_logit)
        q_adjusted = (q - p_0) / p_middle
        return Distributions.quantile(BetaPhi2(μ, ϕ), q_adjusted)
    end
end