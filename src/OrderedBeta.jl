import Distributions: Beta, ContinuousUnivariateDistribution
import Random: rand


"""
    OrderedBeta(μ, ϕ, cut0, cut1)

The distribution is defined on the interval [0, 1] with additional point masses at 0 and 1.

# Parameters
- `μ`: location parameter ]0, 1[
- `ϕ`: precision parameter (must be positive)
- `cut0`: first cutpoint
- `cut1`: log of the difference between the second and first cutpoints


# Examples
```jldoctest
julia> OrderedBeta(0.5, 2, 0, 1)
```
"""
struct OrderedBeta{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    ϕ::T
    cut0::T
    cut1::T
    beta_dist::Beta{T}

    function OrderedBeta{T}(μ::T, ϕ::T, cut0::T, cut1::T) where {T<:Real}
        @assert ϕ > 0 "ϕ must be positive"
        @assert cut0 < cut1 "cut0 must be less than cut1"
        new{T}(μ, ϕ, cut0, cut1, Beta(μ * ϕ, (1 - μ) * ϕ))
    end
end

OrderedBeta(μ::T, ϕ::T, cut0::T, cut1::T) where {T<:Real} = OrderedBeta{T}(μ, ϕ, cut0, cut1)

function OrderedBeta(μ::Real, ϕ::Real, cut0::Real, cut1::Real)
    T = promote_type(typeof(μ), typeof(ϕ), typeof(cut0), typeof(cut1))
    OrderedBeta(T(μ), T(ϕ), T(cut0), T(cut1))
end

# Methods ------------------------------------------------------------------------------------------
params(d::OrderedBeta) = (d.μ, d.ϕ, d.cut0, d.cut1)
minimum(::OrderedBeta) = 0
maximum(::OrderedBeta) = 1
insupport(::OrderedBeta, x::Real) = 0 ≤ x ≤ 1

# This is probably incorrect as it doesn't take into account the zeros and ones:
# mean(d::OrderedBeta) = logistic(d.μ)
# var(d::OrderedBeta) = logistic(d.μ) * (1 - logistic(d.μ)) / (1 + d.ϕ)


# Random -----
function Random.rand(rng::Random.AbstractRNG, d::OrderedBeta)
    μ, ϕ, cut0, cut1 = params(d)
    thresh = [cut0, cut0 + exp(cut1)]
    u = Random.rand(rng)

    if u <= 1 - logistic(μ - thresh[1])
        return zero(μ)
    elseif u >= 1 - logistic(μ - thresh[2])
        return one(μ)
    else
        return Random.rand(rng, d.beta_dist)
    end
end

# Random.rand(d::OrderedBeta) = rand(Random.GLOBAL_RNG, d)
# Random.rand(rng::Random.AbstractRNG, d::OrderedBeta, n::Int) = [rand(rng, d) for _ in 1:n]
# Random.rand(d::OrderedBeta, n::Int) = rand(Random.GLOBAL_RNG, d, n)

# sampler(d::OrderedBeta) = d