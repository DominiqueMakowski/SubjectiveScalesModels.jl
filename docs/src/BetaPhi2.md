# BetaPhi2

Regression models with a **Beta** distribution can be useful to predict scores from bounded variables (that has an upper and lower limit), such as that of scales.

The `SubjectiveScalesModels.jl` package defines the the `BetaPhi2()` function that can be used to generate or model this type of data.

```@docs
BetaPhi2
```

## Usage

### Simulate Data

!!! tip "TLDR"
    You can use `rand(dist, n)` to generate *n* observations from a `BetaPhi2()` distribution.

Let's generate some data from a `BetaPhi2()` distribution with known parameters that we are going to try to recover using Bayesian modelling.


```@example betaphi1
using DataFrames
using Random
using Turing
using CairoMakie
using StatsFuns: logistic
using SubjectiveScalesModels
```

```@example betaphi1
Random.seed!(123)

y = rand(BetaPhi2(μ=0.7, ϕ=3.0), 1000)

hist(y, bins=100, color=:darkred)
```


### Prior Specification

Expressing parameters on the logit scale for `μ` and the log scale for `ϕ` can be useful to define priors that are more interpretable and easier to specify (and to avoid computational issues caused by the bounded nature of the parameters).

```@raw html
<details><summary>See code</summary>
```

```@example betaphi1
μ = Normal(0, 1.0)
ϕ = Normal(0, 1.0)

fig =  Figure(size = (1000, 700))
ax1 = Axis(fig[1, 1], 
    xlabel="Prior on the logit scale",
    ylabel="Prior on μ",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)

xaxis1 = range(-10, 10, 1000)

lines!(ax1, xaxis1, pdf.(μ, xaxis1), color=:purple, linewidth=2, label="μ ~ Normal(0, 1)")
axislegend(ax1; position=:rt)

ax2 = Axis(fig[1, 2], 
    xlabel="Prior after logistic transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax2, logistic.(xaxis1), pdf.(μ, xaxis1), color=:purple, linewidth=2, label="μ")

ax3 = Axis(fig[2, 1], 
    xlabel="Prior on the log scale",
    ylabel="Prior on ϕ",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax3, xaxis1, pdf.(ϕ, xaxis1), color=:green, linewidth=2, label="ϕ ~ Normal(0, 1)")
axislegend(ax3; position=:rt)

ax4 = Axis(fig[2, 2], 
    xlabel="Prior after exponential transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
vlines!(ax4, [1], color=:black, linestyle=:dash, linewidth=1)
lines!(ax4, exp.(xaxis1), pdf.(ϕ, xaxis1), color=:green, linewidth=2, label="ϕ")
xlims!(ax4, -0.5, 15)
fig;
```
```@raw html
</details>
```

```@example betaphi1
fig  # hide
```

### Bayesian Model with Turing

```@repl betaphi1
@model function model_beta(y)
    μ ~ Normal(0, 1)
    ϕ ~ Normal(0, 1)

    for i in 1:length(y)
        μ_raw = logistic(μ)
        # if (μ_raw <= eps()) | (μ_raw >= 1 - eps())
        #     Turing.@addlogprob! -Inf
        #     return nothing
        # end
        y[i] ~ BetaPhi2(μ_raw, exp(ϕ))
    end
end

fit = model_beta(y)
posteriors = sample(fit, NUTS(), 500)
```

