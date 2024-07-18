import Distributions: Beta



function BetaMuPhi(μ::Number, ϕ::Number)
    return Beta(μ * ϕ, (1 - μ) * ϕ)
end