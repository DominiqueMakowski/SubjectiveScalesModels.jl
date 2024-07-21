using Distributions: Beta, MixtureModel

"""
    Choco(p0, μ0, ϕ0, μ1, ϕ1)

Construct a Choice-Confidence (Choco) model distribution.


# Examples
```jldoctest
julia> Choco(p0=0.5, μ0=0.7, ϕ0=2, μ1=0.7, ϕ1=2)
MixtureModel{Distributions.LocationScale{Float64, Distributions.Continuous, Distributions.Beta{Float64}}}(K = 2)
components[1] (prior = 0.5000): Distributions.LocationScale{Float64, Distributions.Continuous, Distributions.Beta{Float64}}(
μ: 0.5
σ: -0.5
ρ: Distributions.Beta{Float64}(α=2.8, β=1.2000000000000002)
)

components[2] (prior = 0.5000): Distributions.LocationScale{Float64, Distributions.Continuous, Distributions.Beta{Float64}}(
μ: 0.5
σ: 0.5
ρ: Distributions.Beta{Float64}(α=2.8, β=1.2000000000000002)
)
```
"""
function Choco(; p0::Real=0.5, μ0::Real=0.5, ϕ0::Real=1, μ1::Real=0.5, ϕ1::Real=1)
    @assert 0 <= p0 <= 1 "p0 must be in [0, 1]"
    return MixtureModel(
        [
            0.5 + (-0.5 * Beta(μ0 * 2 * ϕ0, 2 * ϕ0 * (1 - μ0))),
            0.5 + (0.5 * Beta(μ1 * 2 * ϕ1, 2 * ϕ1 * (1 - μ1)))
        ], [p0, 1 - p0])
end


function Choco(p0::Real, μ0::Real, ϕ0::Real, μ1::Real, ϕ1::Real)
    return Choco(p0=p0, μ0=μ0, ϕ0=ϕ0, μ1=μ1, ϕ1=ϕ1)
end

# TODO: should we write a new type for this?