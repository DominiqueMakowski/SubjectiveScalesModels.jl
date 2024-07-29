# Ordered Beta Regressions

Data from subjective scales often exhibit clustered responses at the extremes (e.g., zeros and ones). 
This makes it challenging to model with regular Beta regressions.
The Ordered Beta distribution allows for the presence of zeros and ones in an convenient way.

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/illustration_slider.gif?raw=true)

## Function

```@docs
OrderedBeta
```


## Usage

Data with clustered extreme responses are common in psychology and cognitive neuroscience. 
Ordered Beta models are a convenient and parsimonious way of modelling such data.

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/illustration_orderedbeta.png?raw=true)

The model is based on a distribution with 4 parameters, 2 of which are the parameters of the [`BetaPhi2`](@ref) distribution (modeling data in between the extremes), and *k1* and *k2* delimiting fuzzy boundaries by which the probability of extreme values increases.

Because these 4 parameters come with their own constraints (i.e., *phi* $\phi$ must be positive, *mu* $\mu$, *k1* and *k2* must be between 0 and 1), it is convenient to express them on a transformed scale (in which they become unconstrained and can adopt any values).

In particular, *mu* $\mu$, *k1* and *k2* are typically expressed on the **logit** scale, and *phi* $\phi$ is expressed on the log scale.

```@example ordbeta1
# using RDatasets
# using CairoMakie
# using Turing
# using StatsFuns: logistic
# using SubjectiveScalesModels


data = dataset("datasets", "iris")
data.y = data.PetalWidth .- minimum(data.PetalWidth)
data.y = data.y ./ maximum(data.y)
data.x = data.PetalLength ./ maximum(data.PetalLength)

# Inflate zeros and ones
data = vcat(data, data[(data.y .== 0) .| (data.y .== 1), :])
data = vcat(data, data[(data.y .== 0) .| (data.y .== 1), :])

println("N-zero: ", sum(data.y .== 0) ,  ", N-one: ", sum(data.y .== 1))
```


## Validation against R Implementation

### R Output

Let's start by making some toy data (the data itself doesn't matter, as we are only interested in getting the same results) and fit an Ordered Beta model using the `ordbetareg` package.

```r
library(ordbetareg)

data <- iris 
data$y  <- data$Petal.Width - min(data$Petal.Width)
data$y <- data$y / max(data$y)
data$x <- data$Petal.Length / max(data$Petal.Length)
# Inflate zeros and ones
data <- rbind(data, data[(data$y==0) | (data$y == 1), ])
data <- rbind(data, data[(data$y==0) | (data$y == 1), ])

ordbetareg(formula=y ~ x, data=data)
```

```
 Family: ord_beta_reg 
  Links: mu = identity; phi = identity; cutzero = identity; cutone = identity 
Formula: y ~ x 
   Data: data (Number of observations: 174) 
  Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
         total post-warmup draws = 4000

Regression Coefficients:
          Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
Intercept    -3.82      0.14    -4.09    -3.55 1.00     2701     2494
x             6.42      0.21     6.00     6.83 1.00     2999     2808

Further Distributional Parameters:
        Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS
phi        22.33      2.56    17.61    27.69 1.00     4220     3026
cutzero    -3.41      0.27    -3.95    -2.88 1.00     3202     3248
cutone      1.87      0.06     1.75     2.00 1.00     3323     3298

Draws were sampled using sampling(NUTS). For each parameter, Bulk_ESS
and Tail_ESS are effective sample size measures, and Rhat is the potential
scale reduction factor on split chains (at convergence, Rhat = 1).
```

### Julia Implementation

Let us now do the same thing in Julia and Turing.

```@example ordbeta2
using RDatasets
using CairoMakie
using Turing
using StatsFuns: logistic
using SubjectiveScalesModels


data = dataset("datasets", "iris")
data.y = data.PetalWidth .- minimum(data.PetalWidth)
data.y = data.y ./ maximum(data.y)
data.x = data.PetalLength ./ maximum(data.PetalLength)

# Inflate zeros and ones
data = vcat(data, data[(data.y .== 0) .| (data.y .== 1), :])
data = vcat(data, data[(data.y .== 0) .| (data.y .== 1), :])

println("N-zero: ", sum(data.y .== 0) ,  ", N-one: ", sum(data.y .== 1))
```



```@example ordbeta2
@model function model_ordbeta(y, x)
    μ_intercept ~ Normal(0, 3)
    μ_x ~ Normal(0, 3)

    ϕ ~ Normal(0, 3)
    cutzero ~ Normal(0, 3)
    cutone ~ Normal(0, 3)

    for i in 1:length(y)
        μ = μ_intercept + μ_x * x[i]
        y[i] ~ OrderedBeta(logistic(μ), exp(ϕ), logistic(cutzero), logistic(cutone))
    end
end


fit = model_ordbeta(data.y, data.x)
posteriors = sample(fit, NUTS(), 1000)

# Mean posterior
mean(posteriors)
```

!!! danger "Important"
    Note that due to *Stan* limitations, the R implementation has *k2* (cutone) specified as the log of the difference from *k1* (cutzero). 
    We can convert the Julia results by doing: `log(cutone - cutzero)`.


The parameters for *mu* μ are very similar, and that of *phi* ϕ is different (but that is expected as a different parametrization is used). 
The values for the cut points *k1* and *k2* (after the transformation specified above) are also very similar.

```@example ordbeta2
# Make predictions
pred = predict(model_ordbeta([missing for _ in 1:length(data.y)], data.x), posteriors)
pred = Array(pred)

n_bins = 30
bin_zero = [-1/n_bins, eps()] # eps() is the smallest positive number to encompass zero
bin_one = [1, 1+1/n_bins] 
bins = vcat(bin_zero, range(eps(), 1, n_bins-1), bin_one)

fig = hist(data.y, color=:forestgreen, normalization=:pdf, bins=bins)
for i in 1:size(pred, 1) # Iterate over each draw
    # density!(pred[i, :], color=(:black, 0), strokecolor=(:crimson, 0.05), strokewidth=1)
    hist!(pred[i, :], color=(:crimson, 0.01), normalization=:pdf, bins=bins)
end
fig
```

!!! tip "Plotting"
    The sharp number of zeros and ones makes it hard for typical plotting approaches to accurately reflect the distribution. 
    Density plots will tend to be very distorted at the edges (due to the Gaussian kernel used), and histograms will be dependent on the binning. One option is to specify the bin edges in a convenient way to capture the zeros and ones.


**Conclusion**: it seems like the Julia version is working as expected as compared to the original R implementation.