import Distributions: Beta


"""
    BetaMuPhi(μ, ϕ)

Construct a Beta distribution with parameters mean `μ` and precision `ϕ`.
It is defined as `Beta(μ * ϕ, (1 - μ) * ϕ)`.

# Arguments
- `μ`: Location parameter (range: ]0, 1[)
- `ϕ`: Precision parameter (must be > 0)

# Details
Note that when `μ=0.5` and `ϕ=2` (i.e., Beta(1, 1)), the distribution is flat (uniform).

# Examples
```jldoctest
julia> BetaMuPhi(0.5, 2)
Distributions.Beta{Float64}(α=1.0, β=1.0)
```
"""
function BetaMuPhi(μ::Real, ϕ::Real)
    return Beta(μ * ϕ, (1 - μ) * ϕ)
end