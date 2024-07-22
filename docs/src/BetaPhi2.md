# BetaPhi2



Regression models with a **Beta** distribution can be useful to predict scores from bounded variables (that has an upper and lower limit), such as that of scales.

The `SubjectiveScalesModels.jl` package defines the the `BetaPhi2()` function that can be used to generate or model this type of data.

## Function

```@docs
BetaPhi2
```

## Usage

### Simulate Data

!!! tip "Summary"
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

!!! tip "Summary"
    Expressing *μ* on the logit scale and *ϕ* on the log scale is recommended, with default priors as $Normal(0, 1)$.


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

We can easily use this distribution to fit a **Beta regression** model using the `Turing` package.

```@example betaphi1
@model function model_beta(y)
    # Priors
    μ ~ Normal(0, 1)
    ϕ ~ Normal(0, 1)

    # Inference
    for i in 1:length(y)
        y[i] ~ BetaPhi2(logistic(μ), exp(ϕ))
    end
end

fit = model_beta(y)
posteriors = sample(fit, NUTS(), 500)

# 95% CI
hpd(posteriors)
```

### Recover Parameters

!!! tip "Summary"
    Use the `logistic()` (in the `StatsFuns` package) and `exp()` functions to transform the model parameters back to the original scale.

Let us compare the parameters estimated by the model (the mean of the posteriors) with the true values used to generate the data (μ=0.7, ϕ=3.0).

```@example betaphi1
means = DataFrame(mean(posteriors))

table = DataFrame(
    Parameter = means.parameters,
    PosteriorMean = means.mean,
    Estimate = [logistic(means.mean[1]), exp(means.mean[2])],
    TrueValue = [0.7, 3.0]
)
```

Mission accomplished! 