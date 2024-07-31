import Distributions
import StatsBase
import Random

"""
    Choco(p1, μ0, ϕ0, μ1, ϕ1; p_mid, ϕ_mid)

Construct a Choice-Confidence (Choco) model distribution. 
It is defined as a mixture of two Beta distributions, one for each side of the scale, and a third (optional) non-scaled Beta distribution for the middle of the scale (allowing for scores clustered in the center of the scale).
The Beta distributions are defined using the [`BetaPhi2`](@ref) parametrization.

# Arguments
- `p1`: Overall probability of the answers being on the right half (i.e., answers between 0.5 and 1) relative to the left half (i.e., answers between 0 and 0.5).
  Default is 0.5, which means that both sides (i.e., "choices") are equally probable.
- `μ0`, `μ1`: Mean of the Beta distributions for the left and right halves, respectively.
- `ϕ0`, `ϕ1`: Precision of the Beta distributions for the left and right halves, respectively.
- `p_mid`: Probability of the answers being in the middle of the scale (i.e., answers around 0.5). 
  Default is 0, which means that the model is a simple mixture of the two other Beta distributions.
- `ϕ_mid`: Precision of the Beta distribution for the middle of the scale (relevant if `p_mid` > 0). 
  Default to 100. This parameter should probably never be as low as 1, as it would be a flat distribution, 
  rendering the distribution unidentifiable (since the same pattern could be observed with another combination of parameters).

See [`BetaPhi2`](@ref) for more details about the parameters.

# Details
*Beta Phi2* is a variant of the traditional *Mu-Phi* location-precision parametrization. 
The modification - scaling ϕ by a factor of 1/2 - creates in a Beta distribution in which, when μ is at its center (i.e., 0.5), a parameter ϕ equal to 1 results in a flat prior (i.e., \$Beta(1, 1)\$).
It is useful to set priors for ϕ on the log scale in regression models, so that a prior of \$Normal(0, 1)\$ assigns the most probability on a flat distribution (ϕ=1).

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/animation_Choco1.gif?raw=true)

In the case of responses clustered in the middle of the scale (at 0.5), in this possible to add a third (non-scaled) Beta distribution centered around 0.5.

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/animation_Choco2.gif?raw=true)


# Examples
```jldoctest
julia> Choco(p1=0.5, μ0=0.7, ϕ0=2, μ1=0.7, ϕ1=2)
Choco{Float64}(p1=0.5, μ0=0.7, ϕ0=2.0, μ1=0.7, ϕ1=2.0, p_mid=0.0, ϕ_mid=100.0)
```
"""
struct Choco{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    p1::T
    μ0::T
    ϕ0::T
    μ1::T
    ϕ1::T
    p_mid::T
    ϕ_mid::T

    function Choco{T}(p1::T, μ0::T, ϕ0::T, μ1::T, ϕ1::T, p_mid::T, ϕ_mid::T) where {T<:Real}
        new{T}(p1, μ0, ϕ0, μ1, ϕ1, p_mid, ϕ_mid)
    end
end


Choco(p1::T, μ0::T, ϕ0::T, μ1::T, ϕ1::T, p_mid::T, ϕ_mid::T) where {T<:Real} = Choco{T}(p1, μ0, ϕ0, μ1, ϕ1, p_mid, ϕ_mid)

function Choco(p1::Real, μ0::Real, ϕ0::Real, μ1::Real, ϕ1::Real, p_mid::Real, ϕ_mid::Real)
    T = promote_type(typeof(p1), typeof(μ0), typeof(ϕ0), typeof(μ1), typeof(ϕ1), typeof(p_mid), typeof(ϕ_mid))
    return Choco(T(p1), T(μ0), T(ϕ0), T(μ1), T(ϕ1), T(p_mid), T(ϕ_mid))
end

Choco(; p1::Real=0.5, μ0::Real=0.5, ϕ0::Real=1, μ1::Real=0.5, ϕ1::Real=1, p_mid::Real=0, ϕ_mid::Real=100) = Choco(p1, μ0, ϕ0, μ1, ϕ1, p_mid, ϕ_mid)
Choco(p1::Real, μ0::Real, ϕ0::Real, μ1::Real, ϕ1::Real; p_mid::Real=0, ϕ_mid::Real=10) = Choco(p1, μ0, ϕ0, μ1, ϕ1, p_mid, ϕ_mid)



# Definition ----------------------------------------------------------------------------------------
function _Choco(p1::Real, μ0::Real, ϕ0::Real, μ1::Real, ϕ1::Real, p_mid::Real, ϕ_mid::Real)
    p_tot = 1 - p_mid
    return Distributions.MixtureModel(
        [
            0.5 + (-0.5 * BetaPhi2(μ0, ϕ0)),
            0.5 + (0.5 * BetaPhi2(μ1, ϕ1)),
            BetaPhi2(0.5, ϕ_mid),
        ], [p_tot - p1 * p_tot, p_tot * p1, p_mid])
end

# Basic ------------------------------------------------------------------------------------------
Distributions.params(d::Choco) = (d.p1, d.μ0, d.ϕ0, d.μ1, d.ϕ1, d.p_mid, d.ϕ_mid)
Distributions.minimum(::Choco) = 0
Distributions.maximum(::Choco) = 1
Distributions.insupport(::Choco, x::Real) = 0 ≤ x ≤ 1
Distributions.mean(d::Choco) = Distributions.mean(_Choco(d.p1, d.μ0, d.ϕ0, d.μ1, d.ϕ1, d.p_mid, d.ϕ_mid))
Distributions.var(d::Choco) = Distributions.var(_Choco(d.p1, d.μ0, d.ϕ0, d.μ1, d.ϕ1, d.p_mid, d.ϕ_mid))


# Random -------------------------------------------------------------------------------------------
Distributions.sampler(d::Choco) = Distributions.sampler(_Choco(d.p1, d.μ0, d.ϕ0, d.μ1, d.ϕ1, d.p_mid, d.ϕ_mid))
Random.rand(rng::Random.AbstractRNG, d::Choco) = Distributions.rand(rng, _Choco(d.p1, d.μ0, d.ϕ0, d.μ1, d.ϕ1, d.p_mid, d.ϕ_mid))
Random.rand(d::Choco) = rand(Random.GLOBAL_RNG, d)
Random.rand(rng::Random.AbstractRNG, d::Choco, n::Int) = [rand(rng, d) for _ in 1:n]
Random.rand(d::Choco, n::Int) = rand(Random.GLOBAL_RNG, d, n)


# PDF -------------------------------------------------------------------------------------------
Distributions.pdf(d::Choco, x::Real) = Distributions.pdf(_Choco(d.p1, d.μ0, d.ϕ0, d.μ1, d.ϕ1, d.p_mid, d.ϕ_mid), x)
Distributions.logpdf(d::Choco, x::Real) = Distributions.logpdf(_Choco(d.p1, d.μ0, d.ϕ0, d.μ1, d.ϕ1, d.p_mid, d.ϕ_mid), x)
Distributions.cdf(d::Choco, x::Real) = Distributions.cdf(_Choco(d.p1, d.μ0, d.ϕ0, d.μ1, d.ϕ1, d.p_mid, d.ϕ_mid), x)


# function Choco(; p0::Real=0.5, μ0::Real=0.5, ϕ0::Real=1, μ1::Real=0.5, ϕ1::Real=1)
#     @assert 0 <= p0 <= 1 "p0 must be in [0, 1]"
#     return MixtureModel(
#         [
#             0.5 + (-0.5 * Beta(μ0 * 2 * ϕ0, 2 * ϕ0 * (1 - μ0))),
#             0.5 + (0.5 * Beta(μ1 * 2 * ϕ1, 2 * ϕ1 * (1 - μ1)))
#         ], [p0, 1 - p0])
# end


# function Choco(p0::Real, μ0::Real, ϕ0::Real, μ1::Real, ϕ1::Real)
#     return Choco(p0=p0, μ0=μ0, ϕ0=ϕ0, μ1=μ1, ϕ1=ϕ1)
# end

# TODO: should we write a new type for this?