import Distributions
import StatsBase
import Random

"""
    BetaPhi2(μ, ϕ)

Construct a Beta distribution with parameters mean `μ` and precision `ϕ`.
It is defined as `Beta(μ * 2ϕ, (1 - μ) * 2ϕ)`.


# Arguments
- `μ`: Location parameter (range: \$]0, 1[\$).
- `ϕ`: Precision parameter (must be \$> 0\$).

# Details
*Beta Phi2* is a variant of the traditional *Mu-Phi* parametrization defined as \$Beta(μ * ϕ, (1 - μ) * ϕ)\$ in which, when μ is at its center (i.e., 0.5), a ϕ equal to 1 results in a flat prior (i.e., \$Beta(1, 1)\$).
It is useful to set priors for ϕ on the log scale in regression models, so that a prior of \$Normal(0, 1)\$ assigns the most probability on a flat distribution (ϕ=1).

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/animation_BetaPhi2.gif?raw=true)

The red area shows the region where the distribution assigns the highest probability to extreme values (towards 0 and/or 1).
The blue area shows the region where the distribution is "convex" and peaks within the \$]0, 1[\$ interval.


# Examples
```jldoctest
julia> BetaPhi2(0.5, 1)
Distributions.Beta{Float64}(α=1.0, β=1.0)

julia> BetaPhi2(μ=0.7, ϕ=3.0)
Distributions.Beta{Float64}(α=4.199999999999999, β=1.8000000000000003)
```
"""
struct BetaPhi2{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    μ::T
    ϕ::T
    # beta_dist::Distributions.Beta{T}

    function BetaPhi2{T}(μ::T, ϕ::T) where {T<:Real}
        if (ϕ <= 0)
            throw(DomainError(ϕ, "ϕ must be > 0"))
        end

        # The test below allows for μ to be == 0 or 1 to prevent logpdf from throwing an error
        if (μ < 0) || (μ > 1)
            throw(DomainError(μ, "μ must be > 0 and < 1"))
        end
        new{T}(μ, ϕ)
    end
end

BetaPhi2(μ::T, ϕ::T) where {T<:Real} = BetaPhi2{T}(μ, ϕ)

function BetaPhi2(μ::Real, ϕ::Real)
    T = promote_type(typeof(μ), typeof(ϕ))
    return BetaPhi2(T(μ), T(ϕ))
end

BetaPhi2(; μ::Real=0.5, ϕ::Real=1) = BetaPhi2(μ, ϕ)

# Definition ----------------------------------------------------------------------------------------
function _BetaPhi2(μ::Real, ϕ::Real)
    return Distributions.Beta(μ * 2 * ϕ, 2 * ϕ * (1 - μ))
end

# Basic ------------------------------------------------------------------------------------------
Distributions.params(d::BetaPhi2) = (d.μ, d.ϕ)
Distributions.minimum(::BetaPhi2) = 0
Distributions.maximum(::BetaPhi2) = 1
Distributions.insupport(::BetaPhi2, x::Real) = 0 ≤ x ≤ 1
Distributions.mean(d::BetaPhi2) = Distributions.mean(_BetaPhi2(d.μ, d.ϕ))
Distributions.var(d::BetaPhi2) = Distributions.var(_BetaPhi2(d.μ, d.ϕ))

# Random -------------------------------------------------------------------------------------------
Distributions.sampler(d::BetaPhi2) = Distributions.sampler(_BetaPhi2(d.μ, d.ϕ))
Random.rand(rng::Random.AbstractRNG, d::BetaPhi2) = Distributions.rand(rng, _BetaPhi2(d.μ, d.ϕ))
Random.rand(d::BetaPhi2) = rand(Random.GLOBAL_RNG, d)
Random.rand(rng::Random.AbstractRNG, d::BetaPhi2, n::Int) = [rand(rng, d) for _ in 1:n]
Random.rand(d::BetaPhi2, n::Int) = rand(Random.GLOBAL_RNG, d, n)

# PDF -------------------------------------------------------------------------------------------
Distributions.pdf(d::BetaPhi2, x::Real) = Distributions.pdf(_BetaPhi2(d.μ, d.ϕ), x)
function Distributions.logpdf(d::BetaPhi2, x::Real)
    if (d.μ <= eps()) | (d.μ >= 1 - eps())
        return -Inf
    end
    return Distributions.logpdf(_BetaPhi2(d.μ, d.ϕ), x)
end

Distributions.cdf(d::BetaPhi2, x::Real) = Distributions.cdf(_BetaPhi2(d.μ, d.ϕ), x)



