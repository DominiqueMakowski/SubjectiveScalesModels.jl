import Distributions: Beta


"""
    BetaMuPhi(μ, ϕ)

Construct a Beta distribution with parameters mean `μ` and precision `ϕ`.

# Examples
```jldoctest
julia> BetaMuPhi(0.5, 2)
Beta{Float64}(α=1.0, β=1.0)
```
"""
function BetaMuPhi(μ::Number, ϕ::Number)
    return Beta(μ * ϕ, (1 - μ) * ϕ)
end