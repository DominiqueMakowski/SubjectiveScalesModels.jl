# Choice-Confidence (Choco) Model

A **Choice-Confidence** scale is a subjective scale in which the left and right halves can be conceptualized as **two different choices** (e.g., True/False, Agree/Disagree, etc.), and the **magnitude** of the response (how much the cursor is set towards he extremes) as the **confidence** in the corresponding choice.

This type of data can be modeled using a "Choice-Confidence" model consisting of a mixture of two scaled Ordered Beta distributions (see [`OrderedBeta`](@ref)) expressing the confidence for each choice, each choice occurring with a certain probability (*p0* and *p1*). 
This model assumes that participant's behaviour when faced with a scale with a psychologically distinct left and right halves can be conceptualized as a decision between two discrete categories associated to a degree confidence in that choice (rather than a continuous degree of one category - e.g., "Agreement" - as assumed with regular *Beta* models).

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
using StatsFuns
using SubjectiveScalesModels

Random.seed!(123)

y = rand(Choco(p1=0.3, μ0=0.7, ϕ0=3, μ1=0.4, ϕ1=2), 1000)

hist(y, bins=beta_bins(30),  normalization=:pdf, color=:darkorange)
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
ϕ1 = Normal(0, 1.2)

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
lines!(ax5, xaxis1, pdf.(ϕ1, xaxis1), color=:orange, linewidth=2, label="ϕ1 ~ Normal(0, 1.2)")
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

fig[0, :] = Label(fig, "Priors Example for Choco Models", fontsize=20, color=:black, font=:bold)
fig;
```
```@raw html
</details>
```

```@example choco1
fig  # hide
```


### Bayesian Choco Model with Turing

```@example choco1
@model function model_choco(y)
    p1 ~ Normal(0, 2)
    μ0 ~ Normal(0, 1)
    μ1 ~ Normal(0, 0.8)
    ϕ0 ~ Normal(0, 1)
    ϕ1 ~ Normal(0, 1.2)

    for i in 1:length(y)
        y[i] ~ Choco(logistic(p1), logistic(μ0), exp(ϕ0), logistic(μ1), exp(ϕ1))
    end
end

fit_choco = model_choco(y)
posteriors = sample(fit_choco, NUTS(), 500)

# 95% CI
hpd(posteriors)
```

Let us do a **Posterior Predictive Check** which involves the generation of predictions from the model to compare the predicted distribution against the actual observed data.

```@example choco1
# Make predictions
pred = predict(model_choco([missing for _ in 1:length(y)]), posteriors)
pred = Array(pred)

fig = hist(y, bins=beta_bins(30), color=:darkorange, normalization=:pdf)
for i in 1:size(pred, 1) # Iterate over each draw
    hist!(pred[i, :], bins=beta_bins(30), color=(:dodgerblue, 0.005), normalization=:pdf)
end
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


## Full Choco (with zero, half and one inflation) 

```@example choco2
using DataFrames
using Random
using Turing
using CairoMakie
using StatsFuns
using SubjectiveScalesModels

Random.seed!(123)

y = rand(Choco(p1=0.3, μ0=0.7, ϕ0=3, μ1=0.4, ϕ1=2, p_mid=0.15, ϕ_mid=200, k0=0.1, k1=0.95), 10_000)

hist(y, bins=beta_bins(31),  normalization=:pdf, color=:darkblue)
```

### Prior Specification

!!! tip "Summary"
    We recommend the following priors:
    - **p1 ~ Normal(0, 2)**: On the logit scale.
    - **μ0, μ1 ~ Normal(0, 1)**: On the logit scale. Assign maximum mass to the middle of the choices.
    - **ϕ0, ϕ1 ~ Normal(0, 1)**: On the log scale. Assign maximum mass to 1 after exponential transformation (flat distribution).
    - **p_mid ~ Normal(-3, 1)**: On the logit scale.  Assign more mass to low probabilities of mid-point responses.
    - **ϕ_mid ~ Gamma(22, 0.22)**: On the log scale. Gamma distribution (> 0) that prevents any values below 1 (which would lead to an unidentifiable distribution). A **Gamma(22, 0.22)** that has a mode of ~100 after exponential transformation is an alternative in case truncated priors are not supported.
    - **k0 ~ -Gamma(3, 3)**:  On the logit scale. Prevents values below 0.5 (after logistic transformation).
    - **k1 ~ Gamma(3, 3)**: On the logit scale. Prevents values above 0.5 (after logistic transformation).

```@raw html
<details><summary>See code</summary>
```

```@example choco2
p1 =  Normal(0, 2.0)
μ0 = Normal(0, 1.0)
μ1 = Normal(0, 1.0)
ϕ0 = Normal(0, 1.0)
ϕ1 = Normal(0, 1.0)
p_mid = Normal(-3, 1.0)
ϕ_mid = truncated(Normal(5, 0.5); lower=0)
ϕ_mid2 = Gamma(22, 0.22)

k0 = -Gamma(3, 3)
k1 = Gamma(3, 3)

fig =  Figure(size = (1000, 1000))

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
lines!(ax3, xaxis1, pdf.(μ1, xaxis1), color=:red, linewidth=2, linestyle=:dash, label="μ1 ~ Normal(0, 1)")
axislegend(ax3; position=:rt)

ax4 = Axis(fig[2, 2], 
    xlabel="Prior after logistic transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax4, logistic.(xaxis1), pdf.(μ0, xaxis1), color=:blue, linewidth=2, label="μ0")
lines!(ax4, logistic.(xaxis1), pdf.(μ1, xaxis1), color=:red, linestyle=:dash, linewidth=2, label="μ1")

ax5 = Axis(fig[3, 1], 
    xlabel="Prior on the log scale",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax5, xaxis1, pdf.(ϕ0, xaxis1), color=:green, linewidth=2, label="ϕ0 ~ Normal(0, 1)")
lines!(ax5, xaxis1, pdf.(ϕ1, xaxis1), color=:orange, linestyle=:dash, linewidth=2, label="ϕ1 ~ Normal(0, 1)")
axislegend(ax5; position=:rt)

ax6 = Axis(fig[3, 2], 
    xlabel="Prior after exponential transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
vlines!(ax6, [1], color=:black, linestyle=:dash, linewidth=1)
lines!(ax6, exp.(xaxis1), pdf.(ϕ0, xaxis1), color=:green, linewidth=2, label="ϕ0")
lines!(ax6, exp.(xaxis1), pdf.(ϕ1, xaxis1), color=:orange, linestyle=:dash, linewidth=2, label="ϕ1")
xlims!(ax6, 0, 10)

ax7 = Axis(fig[4, 1], 
    xlabel="Prior on the logit scale",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax7, xaxis1, pdf.(p_mid, xaxis1), color=:brown, linewidth=2, label="p_mid ~ Normal(-3, 1)")
axislegend(ax7; position=:rt)

ax8 = Axis(fig[4, 2], 
    xlabel="Prior after logistic transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax8, logistic.(xaxis1), pdf.(p_mid, xaxis1), color=:brown, linewidth=2, label="p_mid")

ax9 = Axis(fig[5, 1], 
    xlabel="Prior on the log scale",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax9, xaxis1, pdf.(ϕ_mid, xaxis1), color=:brown, linewidth=2, label="ϕ_mid ~ truncated(Normal(5, 0.5); lower=0)")
lines!(ax9, xaxis1, pdf.(ϕ_mid2, xaxis1), color=:orange, linestyle=:dash, linewidth=2, label="ϕ_mid ~ Gamma(22, 0.22)")
axislegend(ax9; position=:lt)

ax10 = Axis(fig[5, 2], 
    xlabel="Prior after exponential transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
vlines!(ax10, [1], color=:black, linestyle=:dash, linewidth=1)
lines!(ax10, exp.(xaxis1), pdf.(ϕ_mid, xaxis1), color=:brown, linewidth=2, label="ϕ_mid")
lines!(ax10, exp.(xaxis1), pdf.(ϕ_mid2, xaxis1), color=:orange, linestyle=:dash, linewidth=2, label="ϕ_mid")
xlims!(ax10, 0, 300)

xaxis2 = range(-20, 20, 10_000)
ax11 = Axis(fig[6, 1], 
    xlabel="Prior on the log scale",
    ylabel="Distribution",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
lines!(ax11, xaxis2, pdf.(k0, xaxis2), color=:purple, linewidth=2, label="k0 ~ -Gamma(3, 3)")
lines!(ax11, xaxis2, pdf.(k1, xaxis2), color=:orange, linewidth=2, label="k1 ~ Gamma(3, 3)")
axislegend(ax11; position=:rt)
xlims!(ax11, -20, 20)

ax12 = Axis(fig[6, 2], 
    xlabel="Prior after logistic transformation",
    yticksvisible=false,
    xticksvisible=false,
    yticklabelsvisible=false)
vlines!(ax12, [1], color=:black, linestyle=:dash, linewidth=1)
lines!(ax12, logistic.(xaxis2), pdf.(k0, xaxis2), color=:purple, linewidth=2, label="k0")
lines!(ax12, logistic.(xaxis2), pdf.(k1, xaxis2), color=:orange, linewidth=2, label="k1")


fig[0, :] = Label(fig, "Recommended Priors for Choco Models", fontsize=20, color=:black, font=:bold)
fig;
```
```@raw html
</details>
```

```@example choco2
fig  # hide
```

### Bayesian Full Choco Model with Turing

```@example choco2
@model function model_choco(y)
    p1 ~ Normal(0, 2)
    μ0 ~ Normal(0, 1)
    μ1 ~ Normal(0, 1)
    ϕ0 ~ Normal(0, 1)
    ϕ1 ~ Normal(0, 1)
    # p_mid ~ Normal(-3, 1)
    # ϕ_mid ~ truncated(Normal(5, 0.5); lower=0)
    k0 ~ -Gamma(3, 3)
    k1 ~ Gamma(3, 3)

    for i in 1:length(y)
        y[i] ~ Choco(logistic(p1), logistic(μ0), exp(ϕ0), logistic(μ1), exp(ϕ1), 0.0, 200, logistic(k0), logistic(k1))
    end
end

fit_choco = model_choco(y)
posteriors = sample(fit_choco, NUTS(), 500)
```

Making inference on *p_mid* and *ϕ_mid* is challenging and requires a lot of data.
It currently **fails** if we set *p_mid* to anything else than zero.

```@example choco2
# Make predictions
pred = predict(model_choco([missing for _ in 1:length(y)]), posteriors)
pred = Array(pred)

fig = hist(y, bins=beta_bins(31), color=:darkblue, normalization=:pdf)
for i in 1:size(pred, 1) # Iterate over each draw
    hist!(pred[i, :], bins=beta_bins(31), color=(:darkorange, 0.01), normalization=:pdf)
end
fig
```