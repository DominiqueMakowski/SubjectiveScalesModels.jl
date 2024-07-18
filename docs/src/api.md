# Functions

```@docs
BetaMuPhi(μ::T, ϕ::T) where {T<:Number}
```

```@docs
data_rescale(x::Vector{T}; old_range::Vector{T}=[minimum(x), maximum(x)], new_range::Vector{T}=[0, 1]) where {T<:Number}
```