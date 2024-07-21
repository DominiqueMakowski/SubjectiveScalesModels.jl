import Distributions: Beta


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
function BetaPhi2(; μ::Real=0.5, ϕ::Real=1)
    return Beta(μ * 2 * ϕ, 2 * ϕ * (1 - μ))
end

function BetaPhi2(μ::Real, ϕ::Real)
    return BetaPhi2(μ=μ, ϕ=ϕ)
end

# TODO: should we write a new type for this?