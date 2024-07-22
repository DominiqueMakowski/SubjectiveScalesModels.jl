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
