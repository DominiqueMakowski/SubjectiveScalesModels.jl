# Choco Model

A **Choice-Confidence** scale is a subjective scale in which the left and right halves can be conceptualized as **two different choices** (e.g., True/False, Agree/Disagree, etc.), and the **magnitude** of the response (how much the cursor is placed towards he extremes) as the **confidence** in the corresponding choice.

This type of data can be modeled using a mixture of two scaled $Beta$ distributions expressing the confidence for each choice, each choice occurring with a certain probability.

The `SubjectiveScalesModels.jl` package defines the the `Choco()` function that can be used to generate or model data from choice-confidence scales.

```@docs
Choco
```


## Demonstration

### Generate Data

Let's generate some data from a `Choco()` distribution with known parameters that we are going to try to recover using Bayesian modelling.

```@example choco1
using DataFrames
using Random
using Turing
using CairoMakie
using StatsFuns: logistic
using SubjectiveScalesModels
```

```@example choco1
Random.seed!(123)

y = rand(Choco(p0=0.3, μ0=0.7, ϕ0=1, μ1=0.3, ϕ1=3.0), 1000)

hist(y, bins=100, color=:darkred)
```

### Decide on Priors

Deciding on priors requires a good understanding of the meaning of the parameters of the [`BetaPhi2`](@ref) distribution on which the Choco model is based.

The parameters of the `Choco()` distribution have the following requirements:

- `p0`, `μ0` and `μ1`: Must be in the interval 0-1.
- `ϕ0` and `ϕ1`: Must be positive (with a special value at 1 where the distribution is flat when μ is at 0.5).

Because of these specificities, it this convenient to express priors on a different scale (the logit scale for `p0`, `μ0` and `μ1`, and the log scale for `ϕ0` and `ϕ1`) and then transform them using a logistic or exponential link functions.

```@raw html
<details><summary>See code</summary>
```

```@example choco1
fig =  Figure(size = (1000, 700))
ax1 = Axis(fig[1, 1], 
    xlabel="Prior on the logit scale",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)

p0 =  Normal(0, 3)
μ0 = Normal(0, 1.5)
μ1 = Normal(0, 1.0)
ϕ0 = Normal(0, 1.0)
ϕ1 = Normal(0, 0.8)

xaxis1 = range(-10, 10, 1000)

lines!(ax1, xaxis1, pdf.(p0, xaxis1), color=:purple, linewidth=2, label="p0 ~ Normal(0, 3)")
axislegend(ax1; position=:rt)

ax2 = Axis(fig[1, 2], 
    xlabel="Prior after logistic transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax2, logistic.(xaxis1), pdf.(p0, xaxis1), color=:purple, linewidth=2, label="p0")

ax3 = Axis(fig[2, 1], 
    xlabel="Prior on the logit scale",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax3, xaxis1, pdf.(μ0, xaxis1), color=:blue, linewidth=2, label="μ0 ~ Normal(0, 1.5)")
lines!(ax3, xaxis1, pdf.(μ1, xaxis1), color=:red, linewidth=2, label="μ1 ~ Normal(0, 1.0)")
axislegend(ax3; position=:rt)

ax4 = Axis(fig[2, 2], 
    xlabel="Prior after logistic transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax4, logistic.(xaxis1), pdf.(μ0, xaxis1), color=:blue, linewidth=2, label="μ0")
lines!(ax4, logistic.(xaxis1), pdf.(μ1, xaxis1), color=:red, linewidth=2, label="μ1")

ax5 = Axis(fig[3, 1], 
    xlabel="Prior on the log scale",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax5, xaxis1, pdf.(ϕ0, xaxis1), color=:green, linewidth=2, label="ϕ0 ~ Normal(0, 1)")
lines!(ax5, xaxis1, pdf.(ϕ1, xaxis1), color=:orange, linewidth=2, label="ϕ1 ~ Normal(0, 0.8)")
axislegend(ax5; position=:rt)

ax6 = Axis(fig[3, 2], 
    xlabel="Prior after exponential transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
vlines!(ax6, [1], color=:black, linestyle=:dash, linewidth=1)
lines!(ax6, exp.(xaxis1), pdf.(ϕ0, xaxis1), color=:green, linewidth=2, label="ϕ0")
lines!(ax6, exp.(xaxis1), pdf.(ϕ1, xaxis1), color=:orange, linewidth=2, label="ϕ1")
xlims!(ax6, 0, 10)
fig
```
```@raw html
</details>
```


### Specify Turing Model


```@example choco1
@model function model_choco(y)
    p0 ~ Normal(0, 3)
    μ0 ~ truncated(Normal(0, 1.5), -10, 10)
    μ1 ~ truncated(Normal(0, 1.0), -10, 10)
    ϕ0 ~ Normal(0, 1)
    ϕ1 ~ Normal(0, 0.8)

    for i in 1:length(y)
        y[i] ~ Choco(logistic(p0), logistic(μ0), exp(ϕ0), logistic(μ1), exp(ϕ1))
    end
end

fit = model_choco(y)
posteriors = sample(fit, NUTS(), 500)
```

!!! tip
    It can be useful to truncate the priors for $\mu$ to avoid the model to explore regions to close to the boundaries 0 and 1 (after transformation), as it might lead to convergence errors




```@example choco1
posterior_mean = DataFrame(mean(posteriors))

# Format
results = DataFrame(
    Parameter = posterior_mean.parameters,
    Posterior_Mean = round.(posterior_mean.mean; digits=2),
    Estimate = round.([
        logistic(posterior_mean.mean[1]), 
        logistic(posterior_mean.mean[2]),
        logistic(posterior_mean.mean[3]),
        exp(posterior_mean.mean[4]),
        exp(posterior_mean.mean[5])
        ]; digits=2),
    Truth = [0.3, 0.7, 1, 0.3, 3]
)

results
```

## Real Data Example

### Download Data 


```@example choco2
using DataFrames, CSV, Downloads
using Random
using Turing
using CairoMakie
using StatsFuns: logistic
using SubjectiveScalesModels
```

```@example choco2
Random.seed!(123)

df = CSV.read(Downloads.download("https://raw.githubusercontent.com/RealityBending/FakeFace/main/data/data.csv"), DataFrame)
df = df[:, [:Participant, :Stimulus, :Real, :Attractive]]


hist(df.Real, bins=50, color=:darkred)
```

Many zeros and ones, which will create problems with the simple Choco model.

```@example choco2
df = df[(df.Real .> 0.001) .& (df.Real .< 0.999) .& (df.Real .!= 0.5), :]
```

In order to decrease the duration of the sampling (for demonstration), we will also keep only the first 2000 rows.

```@example choco2
df = df[1:1000, :]

hist(df.Real, bins=50, color=:darkblue)
```

### Basic Model 

Fit model:

```@example choco2
@model function model_choco(Real)
    p0 ~ Normal(0, 2)
    μ0 ~ truncated(Normal(0, 1.0), -10, 10)
    μ1 ~ truncated(Normal(0, 1.0), -10, 10)
    ϕ0 ~ Normal(0, 1)
    ϕ1 ~ Normal(0, 1.0)

    for i in 1:length(Real)
        Real[i] ~ Choco(logistic(p0), logistic(μ0), exp(ϕ0), logistic(μ1), exp(ϕ1))
    end
end

fit = model_choco(df.Real)
posteriors = sample(fit, NUTS(), 500)
hpd(posteriors; alpha=0.95)
```

```@example choco2
posterior_mean = DataFrame(mean(posteriors))
results = DataFrame(
    Parameter = posterior_mean.parameters,
    Estimate = round.([
        logistic(posterior_mean.mean[1]), 
        logistic(posterior_mean.mean[2]),
        logistic(posterior_mean.mean[3]),
        exp(posterior_mean.mean[4]),
        exp(posterior_mean.mean[5])
        ]; digits=2)
)
```

```@example choco2
pred = predict(model_choco([(missing) for _ in 1:nrow(df)]), posteriors)
pred = Array(pred)

fig = hist(df.Real, normalization=:pdf, color=:darkblue, bins=50)
for i in 1:length(posteriors)
    density!(pred[i, :], color=(:black, 0), strokecolor=(:orange, 0.1), strokewidth=3, boundary=(0, 1))
end
fig
```


### Effect of Attractiveness


```@example choco2
@model function model_choco2(Real, Attractive)
    # Priors at intercept
    p0_intercept ~ Normal(0, 2)
    μ0_intercept ~ truncated(Normal(0, 1.0), -10, 10)
    μ1_intercept ~ truncated(Normal(0, 1.0), -10, 10)
    ϕ0_intercept ~ Normal(0, 1)
    ϕ1_intercept ~ Normal(0, 1.0)

    # Priors over attractiveness effect
    p0_attractiveness ~ Normal(0, 0.5)
    μ0_attractiveness ~ Normal(0, 0.5)
    μ1_attractiveness ~ Normal(0, 0.5)
    ϕ0_attractiveness ~ Normal(0, 0.1)
    ϕ1_attractiveness ~ Normal(0, 0.1)

    # Inference
    for i in 1:length(Real)
        p0 = logistic(p0_intercept + p0_attractiveness * Attractive[i])
        μ0 = logistic(μ0_intercept + μ0_attractiveness * Attractive[i])
        μ1 = logistic(μ1_intercept + μ1_attractiveness * Attractive[i])
        ϕ0 = exp(ϕ0_intercept + ϕ0_attractiveness * Attractive[i])
        ϕ1 = exp(ϕ1_intercept + ϕ1_attractiveness * Attractive[i])
        
        Real[i] ~ Choco(p0, μ0, ϕ0, μ1, ϕ1)
    end
end

fit = model_choco2(df.Real, df.Attractive)
posteriors = sample(fit, NUTS(), 500)
hpd(posteriors; alpha=0.95)
```

```@example choco2
pred = predict(model_choco2(
    [(missing) for _ in 1:1000],
    repeat([0, 1]; inner=500)
), posteriors)
pred = Array(pred)

fig = hist(df.Real, normalization=:pdf, color=:grey, bins=50)
for i in 1:length(posteriors)
    density!(pred[i, 1:500], color=(:black, 0), strokecolor=(:brown, 0.1), strokewidth=3, boundary=(0, 1))
    density!(pred[i, 501:1000], color=(:black, 0), strokecolor=(:blue, 0.1), strokewidth=3, boundary=(0, 1))
end
fig
```