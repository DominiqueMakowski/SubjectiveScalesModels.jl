import Distributions: Beta


"""
    BetaMuPhi(μ, ϕ)

Construct a Beta distribution with parameters mean `μ` and precision `ϕ`.


"""
function BetaMuPhi(μ::Number, ϕ::Number)
    return Beta(μ * ϕ, (1 - μ) * ϕ)
end