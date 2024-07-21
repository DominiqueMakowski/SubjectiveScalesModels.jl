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
using Random
using SubjectiveScalesModels
using Turing
using CairoMakie
using StatsFuns: logistic
```

```@example choco1
y = rand(Choco(p0=0.3, μ0=0.7, ϕ0=1, μ1=0.3, ϕ1=3.0), 1000)

hist(y, bins=100, color=:darkred)
```

### Specify Turing Model

#### Priors

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
μ0 = Normal(0, 2)
μ1 = Normal(0, 1.5)
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
lines!(ax3, xaxis1, pdf.(μ0, xaxis1), color=:blue, linewidth=2, label="μ0 ~ Normal(0, 2)")
lines!(ax3, xaxis1, pdf.(μ1, xaxis1), color=:red, linewidth=2, label="μ1 ~ Normal(0, 1.5)")
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

```@example choco1
# @model function model_choco(y)
#     p0 ~ Beta(10, 10)
#     μ0 ~ Normal(0, 3)
#     ϕ0 ~ truncated(Normal(0.1, 1); lower=0)
#     μ1 ~ Normal(0, 3)
#     ϕ1 ~ truncated(Normal(0.1, 1); lower=0)

#     for i in 1:length(y)
#         y[i] ~ Choco(p0, logistic(μ0), ϕ0, logistic(μ1), ϕ1)
#     end
# end
```