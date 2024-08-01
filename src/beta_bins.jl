import Base: minimum, maximum
import StatsBase: iqr

"""
    beta_bins(n::Int=10)

A Simple function that produces a vector of bins boundaries, useful for histograms.
It adds bins on the left and right tails for extreme values (0 and 1), convenient for data suited for [`OrderedBeta`](@ref) models. 

```jldoctest
julia> beta_bins(3)
6-element Vector{Float64}:
 -0.3333333333333333
  2.220446049250313e-16
  0.3333333333333335
  0.6666666666666667
  1.0
  1.3333333333333333

julia> beta_bins(rand(100))
7-element Vector{Float64}:
 -0.25
  2.220446049250313e-16
  0.25000000000000017
  0.5000000000000001
  0.75
  1.0
  1.25
```
"""
function beta_bins(n::Int=10)
    return vcat([-1 / n], range(eps(), 1, n + 1), [1 + 1 / n])
end

function beta_bins(x::Vector{<:Real})
    # https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
    bw = 2 * iqr(x) / length(x)^(1 / 3)
    return beta_bins(Int(round(1 / bw)))
end