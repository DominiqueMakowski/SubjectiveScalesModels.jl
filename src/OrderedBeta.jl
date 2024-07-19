import Distributions: Beta, ContinuousUnivariateDistribution
import Random


"""
    OrderedBeta(μ, ϕ, k1, k2)

The distribution is defined on the interval [0, 1] with additional point masses at 0 and 1.
It is defined as a mixture of a *Beta Phi2* distribution and two point masses at 0 and 1.

# Parameters
- `μ`: location parameter ]0, 1[
- `ϕ`: precision parameter (must be positive)
- `k1`: first cutpoint
- `k2`: Difference between the second and first cutpoints


# Examples
```jldoctest
julia> OrderedBeta(0.5, 1)
OrderedBeta{Float64}(
μ: 0.5
ϕ: 1.0
k1: -6.0
k2: 12.0
beta_dist: Distributions.Beta{Float64}(α=1.0, β=1.0)
)
```
"""
struct OrderedBeta{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    ϕ::T
    k1::T
    k2::T
    beta_dist::Beta{T}

    function OrderedBeta{T}(μ::T, ϕ::T, k1::T, k2::T) where {T<:Real}
        @assert ϕ > 0 "ϕ must be positive"
        @assert k1 < k2 "k1 must be less than k2"
        new{T}(μ, ϕ, k1, k2, Beta(μ * 2 * ϕ, 2 * ϕ * (1 - μ)))
    end
end

OrderedBeta(μ::T, ϕ::T, k1::T, k2::T) where {T<:Real} = OrderedBeta{T}(μ, ϕ, k1, k2)

function OrderedBeta(μ::Real=0.5, ϕ::Real=1, k1::Real=-6, k2::Real=2 * abs(k1))
    T = promote_type(typeof(μ), typeof(ϕ), typeof(k1), typeof(k2))
    OrderedBeta(T(μ), T(ϕ), T(k1), T(k2))
end

# Methods ------------------------------------------------------------------------------------------
params(d::OrderedBeta) = (d.μ, d.ϕ, d.k1, d.k2)
minimum(::OrderedBeta) = 0
maximum(::OrderedBeta) = 1
insupport(::OrderedBeta, x::Real) = 0 ≤ x ≤ 1

# This is probably incorrect as it doesn't take into account the zeros and ones:
# mean(d::OrderedBeta) = logistic(d.μ)
# var(d::OrderedBeta) = logistic(d.μ) * (1 - logistic(d.μ)) / (1 + d.ϕ)


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



# k1 = 1
# _invlogistic(_logistic(k1))
# k2 = 2 * abs(k1)
# _logistic(k1 + k2)

# function Random.rand(rng::Random.AbstractRNG, d::OrderedBeta)
#     μ, ϕ, k1, k2 = params(d)
#     thresh = [_logistic(k1), _logistic(k1 + k2)]
#     u = Random.rand(rng)

#     if u <= 1 - (μ - thresh[1])
#         return zero(μ)
#     elseif u >= 1 - (μ - thresh[2])
#         return one(μ)
#     else
#         return Random.rand(rng, d.beta_dist)
#     end
# end

# Random.rand(d::OrderedBeta) = rand(Random.GLOBAL_RNG, d)
# Random.rand(rng::Random.AbstractRNG, d::OrderedBeta, n::Int) = [rand(rng, d) for _ in 1:n]
# Random.rand(d::OrderedBeta, n::Int) = rand(Random.GLOBAL_RNG, d, n)

# sampler(d::OrderedBeta) = d