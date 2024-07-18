import Distributions: Beta



function BetaMuPhi(μ, ϕ)
    return Beta(μ * ϕ, (1 - μ) * ϕ)
end