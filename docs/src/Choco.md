# Choice-Confidence (Choco) Model

A **Choice-Confidence** scale is a subjective scale in which the left and right halves can be conceptualized as **two different choices** (e.g., True/False, Agree/Disagree, etc.), and the **magnitude** of the response (how much the cursor is set towards he extremes) as the **confidence** in the corresponding choice.

This type of data can be modeled using a "Choice-Confidence" model consisting of a mixture of two scaled $Beta$ distributions expressing the confidence for each choice, each choice occurring with a certain probability. This model assumes that participant's behaviour when faced with a scale with a psychology distinct left and right halves can be conceptualized as a decision between two discrete choices associated to a degree confidence in said-choice (rather than a continuous degree of one category as assumed with regular *Beta* models).

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/choco_illustration.png?raw=true)

The `SubjectiveScalesModels.jl` package defines the the `Choco()` function that can be used to generate or model data from choice-confidence scales.

## Function

```@docs
Choco
```


## Usage

### Simulate Data

!!! tip "Summary"
    You can use `rand(dist, n)` to generate *n* observations from a `Choco()` distribution with pre-specified parameters.

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

y = rand(Choco(p1=0.3, μ0=0.7, ϕ0=3, μ1=0.4, ϕ1=2), 10000)

hist(y, bins=100,  normalization=:pdf, color=:darkorange)
```

### Prior Specification

Deciding on priors requires a good understanding of the meaning of the parameters of the [`BetaPhi2`](@ref) distribution on which the **Choco** model is based. Make sure you first read the [documentation page](https://dominiquemakowski.github.io/SubjectiveScalesModels.jl/dev/BetaPhi2/#Prior-Specification) about priors of the `BetaPhi2()` distribution.

The parameters of the `Choco()` distribution have the following requirements:

- `p0`, `μ0` and `μ1`: Must be in the interval 0-1.
- `ϕ0` and `ϕ1`: Must be positive (with a special value at 1 where the distribution is flat when μ is at 0.5).

Because of these specificities, it this convenient to express priors on a different scale (the logit scale for `p0`, `μ0` and `μ1`, and the log scale for `ϕ0` and `ϕ1`) and then transform them using a logistic or exponential link functions.

```@raw html
<details><summary>See code</summary>
```

```@example choco1
p1 =  Normal(0, 2.0)
μ0 = Normal(0, 1.0)
μ1 = Normal(0, 0.8)
ϕ0 = Normal(0, 1.0)
ϕ1 = Normal(0, 0.5)

fig =  Figure(size = (850, 600))

ax1 = Axis(fig[1, 1], 
    xlabel="Prior on the logit scale",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)

xaxis1 = range(-10, 10, 1000)

lines!(ax1, xaxis1, pdf.(p1, xaxis1), color=:purple, linewidth=2, label="p1 ~ Normal(0, 2)")
axislegend(ax1; position=:rt)

ax2 = Axis(fig[1, 2], 
    xlabel="Prior after logistic transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax2, logistic.(xaxis1), pdf.(p1, xaxis1), color=:purple, linewidth=2, label="p1")

ax3 = Axis(fig[2, 1], 
    xlabel="Prior on the logit scale",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax3, xaxis1, pdf.(μ0, xaxis1), color=:blue, linewidth=2, label="μ0 ~ Normal(0, 1)")
lines!(ax3, xaxis1, pdf.(μ1, xaxis1), color=:red, linewidth=2, label="μ1 ~ Normal(0, 0.8)")
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
lines!(ax5, xaxis1, pdf.(ϕ1, xaxis1), color=:orange, linewidth=2, label="ϕ1 ~ Normal(0, 0.5)")
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

fig[0, :] = Label(fig, "Priors for Choco Models", fontsize=20, color=:black, font=:bold)
fig;
```
```@raw html
</details>
```

```@example choco1
fig  # hide
```


### Bayesian Model with Turing

```@example choco1
@model function model_choco(y)
    p1 ~ Normal(0, 2)
    μ0 ~ Normal(0, 1)
    μ1 ~ Normal(0, 0.8)
    ϕ0 ~ Normal(0, 1)
    ϕ1 ~ Normal(0, 0.5)

    for i in 1:length(y)
        y[i] ~ Choco(logistic(p1), logistic(μ0), exp(ϕ0), logistic(μ1), exp(ϕ1))
    end
end

fit = model_choco(y)
posteriors = sample(fit, NUTS(), 500)

# 95% CI
hpd(posteriors)
```

Let us do a **Posterior Predictive Check** which involves the generation of predictions from the model to compare the predicted distribution against the actual observed data.

```@example choco1
# Make predictions
pred = predict(model_choco([missing for _ in 1:length(y)]), posteriors)
pred = Array(pred)

fig = hist(y, bins=100, color=:darkorange, normalization=:pdf)
for i in 1:size(pred, 1) # Iterate over each draw
    density!(pred[i, :], color=(:black, 0), strokecolor=(:dodgerblue, 0.05), strokewidth=3)
end
xlims!(0, 1)
fig
```

### Recover Parameters


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

### Data Preprocessing 


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

hist(df.Real, bins=30,  normalization=:pdf, color=:darkred)
```

Many zeros and ones, which will create problems with the simple Choco model.

```@example choco2
df = df[(df.Real .> 0.001) .& (df.Real .< 0.999), :];
```

In order to decrease the duration of the sampling (for demonstration), we will also keep only the first 1000 rows.

```@example choco2
df = df[1:1000, :]

hist(df.Real, bins=30,  normalization=:pdf, color=:crimson)
```

### Basic Model 

Fit model:

```@example choco2
@model function model_choco(y)
    p1 ~ Normal(0, 2)
    μ0 ~ Normal(0, 1)
    μ1 ~ Normal(0, 1)
    ϕ0 ~ Normal(0, 1)
    ϕ1 ~ Normal(0, 1)

    for i in 1:length(y)
        y[i] ~ Choco(logistic(p1), logistic(μ0), exp(ϕ0), logistic(μ1), exp(ϕ1))
    end
end

fit = model_choco(y)
posteriors = sample(fit, NUTS(), 500)

# 95% CI
hpd(posteriors)
```
