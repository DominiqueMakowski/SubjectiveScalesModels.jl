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

```@example choco1
fig =  Figure()
ax1 = Axis(fig[1, 1])

lines!(ax1, range(0, 1, 100), pdf.(Beta(5, 5), range(0, 1, 100)), color=:purple, linewidth=2, label="p0")

ax2 = Axis(fig[1, 2])
# lines!(ax2, range(0, 1, 100), pdf.(Beta(5, 5), range(0, 1, 100)), color=:blue, linewidth=2, label="μ0")
fig
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