using Distributions: Beta, MixtureModel

"""
    Choco(p0, μ0, ϕ0, μ1, ϕ1)

Construct a Choice-Confidence (Choco) model distribution.


# Examples
```jldoctest
julia> # Choco(0.5, 0.7, 2, 0.7, 2)
```
"""
Choco = function (p0::Real=0.5, μ0::Real=0.5, ϕ0::Real=1, μ1::Real=0.5, ϕ1::Real=1)
    @assert 0 <= p0 <= 1 "p0 must be in [0, 1]"
    return MixtureModel(
        [
            0.5 + (-0.5 * Beta(μ0 * 2 * ϕ0, 2 * ϕ0 * (1 - μ0))),
            0.5 + (0.5 * Beta(μ1 * 2 * ϕ1, 2 * ϕ1 * (1 - μ1)))
        ], [p0, 1 - p0])
end

# TODO: should we write a new type for this?