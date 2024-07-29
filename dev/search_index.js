var documenterSearchIndex = {"docs":
[{"location":"api/#Other-Functions","page":"Other Functions","title":"Other Functions","text":"","category":"section"},{"location":"api/#data_rescale()","page":"Other Functions","title":"data_rescale()","text":"","category":"section"},{"location":"api/","page":"Other Functions","title":"Other Functions","text":"data_rescale(x::Vector{T}; old_range::Vector{T}=[minimum(x), maximum(x)], new_range::Vector{T}=[0, 1]) where {T<:Number}","category":"page"},{"location":"api/#SubjectiveScalesModels.data_rescale-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Number","page":"Other Functions","title":"SubjectiveScalesModels.data_rescale","text":"data_rescale(x; old_range=[minimum(x), maximum(x)], new_range=[0, 1])\n\nRescale a variable to a new range. Can be used to normalize a variable between 0 and 1.\n\ndanger: Danger\nThis function is currently used internally and might be moved to another package. Avoid using it directly.\n\nArguments\n\nx: Vector to rescale.\nold_range: Old range of the vector to rescale (will be taken by default from the minimum and maximum value of x).\nnew_range: Range to rescale x to. By default, [0-1].\n\nExamples\n\njulia> data_rescale([1, 2, 3])\n3-element Vector{Float64}:\n 0.0\n 0.5\n 1.0\n\njulia> data_rescale([1, 2, 3]; old_range=[1, 6], new_range=[1, 0])\n3-element Vector{Float64}:\n 1.0\n 0.8\n 0.6\n\n\n\n\n\n","category":"method"},{"location":"api/","page":"Other Functions","title":"Other Functions","text":"SubjectiveScalesModels","category":"page"},{"location":"api/#SubjectiveScalesModels","page":"Other Functions","title":"SubjectiveScalesModels","text":"SubjectiveScalesModels\n\n\n\n\n\n","category":"module"},{"location":"OrderedBeta/#Ordered-Beta-Regressions","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"","category":"section"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"Data from subjective scales often exhibit clustered responses at the extremes (e.g., 0 and ones).  This makes it challenging to model with regular Beta regressions. The Ordered Beta distribution allows for the presence of zeros and ones in an convenient way.","category":"page"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"(Image: )","category":"page"},{"location":"OrderedBeta/#Function","page":"Ordered Beta Regressions","title":"Function","text":"","category":"section"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"OrderedBeta","category":"page"},{"location":"OrderedBeta/#SubjectiveScalesModels.OrderedBeta","page":"Ordered Beta Regressions","title":"SubjectiveScalesModels.OrderedBeta","text":"OrderedBeta(μ, ϕ, k1, k2)\n\nThe distribution is defined on the interval [0, 1] with additional point masses at 0 and 1. The Beta distributions are defined using the BetaPhi2 parametrization.\n\nArguments\n\nμ: location parameter on the scale 0-1\nϕ: precision parameter (must be positive). Note that this parameter is based on the BetaPhi2 reparametrization of the Beta distribution,   which corresponds to half the precision of the traditional Beta distribution as implemented in for example the ordbetareg package.\nk1: first cutpoint (cutzero) on the logit scale (should be negative).\nk2: second cutpoint (cutone) on the logit scale (should be positive). Must be greater than k1.\n\nDetails\n\n(Image: )\n\nThe figure above shows the parameter space for k1 and k2, showing the regions that produce a large proportion of zeros and ones (in red). Understanding this is important to set appropriate priors on these parameters.\n\nExamples\n\njulia> OrderedBeta(0.5, 1, -6, 4)\nOrderedBeta{Float64}(μ=0.5, ϕ=1.0, k1=-6.0, k2=4.0)\n\n\n\n\n\n","category":"type"},{"location":"OrderedBeta/#Usage","page":"Ordered Beta Regressions","title":"Usage","text":"","category":"section"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"(Image: )","category":"page"},{"location":"OrderedBeta/#Validation-against-R-Implementation","page":"Ordered Beta Regressions","title":"Validation against R Implementation","text":"","category":"section"},{"location":"OrderedBeta/#R-Output","page":"Ordered Beta Regressions","title":"R Output","text":"","category":"section"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"Let's start by making some toy data (the data itself doesn't matter, as we are only interested in getting the same results) and fit an Ordered Beta model using the ordbetareg package.","category":"page"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"library(ordbetareg)\n\ndata <- iris \ndata$y  <- data$Petal.Width - min(data$Petal.Width)\ndata$y <- data$y / max(data$y)\ndata$x <- data$Petal.Length / max(data$Petal.Length)\n# Inflate zeros and ones\ndata <- rbind(data, data[(data$y==0) | (data$y == 1), ])\ndata <- rbind(data, data[(data$y==0) | (data$y == 1), ])\n\nordbetareg(formula=y ~ x, data=data)","category":"page"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":" Family: ord_beta_reg \n  Links: mu = identity; phi = identity; cutzero = identity; cutone = identity \nFormula: y ~ x \n   Data: data (Number of observations: 174) \n  Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;\n         total post-warmup draws = 4000\n\nRegression Coefficients:\n          Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS\nIntercept    -3.82      0.14    -4.09    -3.55 1.00     2701     2494\nx             6.42      0.21     6.00     6.83 1.00     2999     2808\n\nFurther Distributional Parameters:\n        Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS Tail_ESS\nphi        22.33      2.56    17.61    27.69 1.00     4220     3026\ncutzero    -3.41      0.27    -3.95    -2.88 1.00     3202     3248\ncutone      1.87      0.06     1.75     2.00 1.00     3323     3298\n\nDraws were sampled using sampling(NUTS). For each parameter, Bulk_ESS\nand Tail_ESS are effective sample size measures, and Rhat is the potential\nscale reduction factor on split chains (at convergence, Rhat = 1).","category":"page"},{"location":"OrderedBeta/#Julia-Implementation","page":"Ordered Beta Regressions","title":"Julia Implementation","text":"","category":"section"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"Let us now do the same thing in Julia and Turing.","category":"page"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"using RDatasets\nusing CairoMakie\nusing Turing\nusing StatsFuns: logistic\nusing SubjectiveScalesModels\n\n\ndata = dataset(\"datasets\", \"iris\")\ndata.y = data.PetalWidth .- minimum(data.PetalWidth)\ndata.y = data.y ./ maximum(data.y)\ndata.x = data.PetalLength ./ maximum(data.PetalLength)\n\n# Inflate zeros and ones\ndata = vcat(data, data[(data.y .== 0) .| (data.y .== 1), :])\ndata = vcat(data, data[(data.y .== 0) .| (data.y .== 1), :])\n\nprintln(\"N-zero: \", sum(data.y .== 0) ,  \", N-one: \", sum(data.y .== 1))","category":"page"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"@model function model_ordbeta(y, x)\n    μ_intercept ~ Normal(0, 3)\n    μ_x ~ Normal(0, 3)\n\n    ϕ ~ Normal(0, 3)\n    cutzero ~ Normal(0, 3)\n    cutone ~ Normal(0, 3)\n\n    for i in 1:length(y)\n        μ = μ_intercept + μ_x * x[i]\n        y[i] ~ OrderedBeta(logistic(μ), exp(ϕ), cutzero, cutone)\n    end\nend\n\n\nfit = model_ordbeta(data.y, data.x)\nposteriors = sample(fit, NUTS(), 1000)\n\n# Mean posterior\nmean(posteriors)","category":"page"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"danger: Important\nNote that due to Stan limitations, the R implementation has k2 (cutone) specified as the log of the difference from k1 (cutzero).  We can convert the Julia results by doing: log(cutone - cutzero).","category":"page"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"The parameters for mu μ are very similar, and that of phi ϕ is different (but that is expected as a different parametrization is used).  The values for the cut points k1 and k2 (after the transformation specified above) are also very similar.","category":"page"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"# Make predictions\npred = predict(model_ordbeta([missing for _ in 1:length(data.y)], data.x), posteriors)\npred = Array(pred)\n\nn_bins = 30\nbin_zero = [-1/n_bins, eps()] # eps() is the smallest positive number to encompass zero\nbin_one = [1, 1+1/n_bins] \nbins = vcat(bin_zero, range(eps(), 1, n_bins-1), bin_one)\n\nfig = hist(data.y, color=:forestgreen, normalization=:pdf, bins=bins)\nfor i in 1:size(pred, 1) # Iterate over each draw\n    # density!(pred[i, :], color=(:black, 0), strokecolor=(:crimson, 0.05), strokewidth=1)\n    hist!(pred[i, :], color=(:crimson, 0.01), normalization=:pdf, bins=bins)\nend\nfig","category":"page"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"tip: Plotting\nThe sharp number of zeros and ones makes it hard for typical plotting approaches to accurately reflect the distribution.  Density plots will tend to be very distorted at the edges (due to the Gaussian kernel used), and histograms will be dependent on the binning. One option is to specify the bin edges in a convenient way to capture the zeros and ones.","category":"page"},{"location":"OrderedBeta/","page":"Ordered Beta Regressions","title":"Ordered Beta Regressions","text":"Conclusion: it seems like the Julia version is working as expected as compared to the original R implementation.","category":"page"},{"location":"BetaPhi2/#BetaPhi2()-for-Beta-Regressions","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"","category":"section"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"Regression models with a Beta distribution can be useful to predict scores from bounded variables (that has an upper and lower limit), such as that of scales.","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"The SubjectiveScalesModels.jl package defines the the BetaPhi2() function that can be used to generate or model this type of data.","category":"page"},{"location":"BetaPhi2/#Function","page":"BetaPhi2() for Beta Regressions","title":"Function","text":"","category":"section"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"BetaPhi2","category":"page"},{"location":"BetaPhi2/#SubjectiveScalesModels.BetaPhi2","page":"BetaPhi2() for Beta Regressions","title":"SubjectiveScalesModels.BetaPhi2","text":"BetaPhi2(μ, ϕ)\n\nConstruct a Beta distribution with parameters mean μ and precision ϕ. It is defined as Beta(μ * 2ϕ, (1 - μ) * 2ϕ).\n\nArguments\n\nμ: Location parameter (range: 0 1).\nϕ: Precision parameter (must be  0).\n\nDetails\n\nBeta Phi2 is a variant of the traditional Mu-Phi location-precision parametrization.  The modification - scaling ϕ by a factor of 1/2 - creates in a Beta distribution in which, when μ is at its center (i.e., 0.5), a parameter ϕ equal to 1 results in a flat prior (i.e., Beta(1 1)). It is useful to set priors for ϕ on the log scale in regression models, so that a prior of Normal(0 1) assigns the most probability on a flat distribution (ϕ=1).\n\n(Image: )\n\nThe red area shows the region where the distribution assigns the highest probability to extreme values (towards 0 and/or 1). The blue area shows the region where the distribution is \"convex\" and peaks within the 0 1 interval.\n\nExamples\n\njulia> BetaPhi2(0.5, 1)\nBetaPhi2{Float64}(μ=0.5, ϕ=1.0)\n\n\n\n\n\n","category":"type"},{"location":"BetaPhi2/#Usage","page":"BetaPhi2() for Beta Regressions","title":"Usage","text":"","category":"section"},{"location":"BetaPhi2/#Simulate-Data","page":"BetaPhi2() for Beta Regressions","title":"Simulate Data","text":"","category":"section"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"tip: Summary\nYou can use rand(dist, n) to generate n observations from a BetaPhi2() distribution with pre-specified parameters.","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"Let's generate some data from a BetaPhi2() distribution with known parameters that we are going to try to recover using Bayesian modelling.","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"using DataFrames\nusing Random\nusing Turing\nusing CairoMakie\nusing StatsFuns: logistic\nusing SubjectiveScalesModels","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"Random.seed!(123)\n\ny = rand(BetaPhi2(μ=0.7, ϕ=3.0), 1000)\n\nhist(y, bins=100, color=:dodgerblue, normalization=:pdf)","category":"page"},{"location":"BetaPhi2/#Prior-Specification","page":"BetaPhi2() for Beta Regressions","title":"Prior Specification","text":"","category":"section"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"tip: Summary\nExpressing μ on the logit scale and ϕ on the log scale is recommended, with default priors as Normal(0 1).","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"Expressing parameters on the logit scale for μ and the log scale for ϕ can be useful to define priors that are more interpretable and easier to specify (and to avoid computational issues caused by the bounded nature of the parameters).","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"<details><summary>See code</summary>","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"μ = Normal(0, 1.0)\nϕ = Normal(0, 1.0)\n\nfig =  Figure(size = (850, 600))\n\nax1 = Axis(fig[1, 1], \n    xlabel=\"Prior on the logit scale\",\n    ylabel=\"Prior on μ\",\n    yticksvisible=false,\n    xticksvisible=false,\n    yticklabelsvisible=false)\n\nxaxis1 = range(-10, 10, 1000)\n\nlines!(ax1, xaxis1, pdf.(μ, xaxis1), color=:purple, linewidth=2, label=\"μ ~ Normal(0, 1)\")\naxislegend(ax1; position=:rt)\n\nax2 = Axis(fig[1, 2], \n    xlabel=\"Prior after logistic transformation\",\n    yticksvisible=false,\n    xticksvisible=false,\n    yticklabelsvisible=false)\nlines!(ax2, logistic.(xaxis1), pdf.(μ, xaxis1), color=:purple, linewidth=2, label=\"μ\")\n\nax3 = Axis(fig[2, 1], \n    xlabel=\"Prior on the log scale\",\n    ylabel=\"Prior on ϕ\",\n    yticksvisible=false,\n    xticksvisible=false,\n    yticklabelsvisible=false)\nlines!(ax3, xaxis1, pdf.(ϕ, xaxis1), color=:green, linewidth=2, label=\"ϕ ~ Normal(0, 1)\")\naxislegend(ax3; position=:rt)\n\nax4 = Axis(fig[2, 2], \n    xlabel=\"Prior after exponential transformation\",\n    yticksvisible=false,\n    xticksvisible=false,\n    yticklabelsvisible=false)\nvlines!(ax4, [1], color=:black, linestyle=:dash, linewidth=1)\nlines!(ax4, exp.(xaxis1), pdf.(ϕ, xaxis1), color=:green, linewidth=2, label=\"ϕ\")\nxlims!(ax4, -0.5, 15)\n\nfig[0, :] = Label(fig, \"Priors for Beta Regressions\", fontsize=20, color=:black, font=:bold)\nfig;","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"</details>","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"fig  # hide","category":"page"},{"location":"BetaPhi2/#Bayesian-Model-with-Turing","page":"BetaPhi2() for Beta Regressions","title":"Bayesian Model with Turing","text":"","category":"section"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"We can easily use this distribution to fit a Beta regression model using the Turing package.","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"@model function model_beta(y)\n    # Priors\n    μ ~ Normal(0, 1)\n    ϕ ~ Normal(0, 1)\n\n    # Inference\n    for i in 1:length(y)\n        y[i] ~ BetaPhi2(logistic(μ), exp(ϕ))\n    end\nend\n\nfit = model_beta(y)\nposteriors = sample(fit, NUTS(), 500)\n\n# 95% CI\nhpd(posteriors)","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"Let us do a Posterior Predictive Check which involves the generation of predictions from the model to compare the predicted distribution against the actual observed data.","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"# Make predictions\npred = predict(model_beta([missing for _ in 1:length(y)]), posteriors)\npred = Array(pred)\n\nfig = hist(y, bins=100, color=:dodgerblue, normalization=:pdf)\nfor i in 1:size(pred, 1) # Iterate over each draw\n    density!(pred[i, :], color=(:black, 0), strokecolor=(:crimson, 0.05), strokewidth=1)\nend\nxlims!(0, 1)\nfig","category":"page"},{"location":"BetaPhi2/#Recover-Parameters","page":"BetaPhi2() for Beta Regressions","title":"Recover Parameters","text":"","category":"section"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"tip: Summary\nUse the logistic() (in the StatsFuns package) and exp() functions to transform the model parameters back to the original scale.","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"Let us compare the parameters estimated by the model (the mean of the posteriors) with the true values used to generate the data (μ=0.7, ϕ=3.0).","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"means = DataFrame(mean(posteriors))\n\ntable = DataFrame(\n    Parameter = means.parameters,\n    PosteriorMean = means.mean,\n    Estimate = [logistic(means.mean[1]), exp(means.mean[2])],\n    TrueValue = [0.7, 3.0]\n)","category":"page"},{"location":"BetaPhi2/","page":"BetaPhi2() for Beta Regressions","title":"BetaPhi2() for Beta Regressions","text":"Mission accomplished! ","category":"page"},{"location":"Choco/#Choice-Confidence-(Choco)-Model","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"","category":"section"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"A Choice-Confidence scale is a subjective scale in which the left and right halves can be conceptualized as two different choices (e.g., True/False, Agree/Disagree, etc.), and the magnitude of the response (how much the cursor is set towards he extremes) as the confidence in the corresponding choice.","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"This type of data can be modeled using a \"Choice-Confidence\" model consisting of a mixture of two scaled Beta distributions expressing the confidence for each choice, each choice occurring with a certain probability (p0 and p1).  This model assumes that participant's behaviour when faced with a scale with a psychologically distinct left and right halves can be conceptualized as a decision between two discrete categories associated to a degree confidence in that choice (rather than a continuous degree of one category - e.g., \"Agreement\" - as assumed with regular Beta models).","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"(Image: )","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"The SubjectiveScalesModels.jl package defines the the Choco() function that can be used to generate or model data from choice-confidence scales.","category":"page"},{"location":"Choco/#Function","page":"Choice-Confidence (Choco) Model","title":"Function","text":"","category":"section"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"Choco","category":"page"},{"location":"Choco/#SubjectiveScalesModels.Choco","page":"Choice-Confidence (Choco) Model","title":"SubjectiveScalesModels.Choco","text":"Choco(p1, μ0, ϕ0, μ1, ϕ1; p_mid, ϕ_mid)\n\nConstruct a Choice-Confidence (Choco) model distribution.  It is defined as a mixture of two Beta distributions, one for each side of the scale, and a third (optional) non-scaled Beta distribution for the middle of the scale (allowing for scores clustered in the center of the scale). The Beta distributions are defined using the BetaPhi2 parametrization.\n\nArguments\n\np1: Overall probability of the answers being on the right half (i.e., answers between 0.5 and 1) relative to the left half (i.e., answers between 0 and 0.5). Default is 0.5, which means that both sides (i.e., \"choices\") are equally probable.\nμ0, μ1: Mean of the Beta distributions for the left and right halves, respectively.\nϕ0, ϕ1: Precision of the Beta distributions for the left and right halves, respectively.\np_mid: Probability of the answers being in the middle of the scale (i.e., answers around 0.5).  Default is 0, which means that the model is a simple mixture of the two other Beta distributions.\nϕ_mid: Precision of the Beta distribution for the middle of the scale (relevant if p_mid > 0).  Default to 100. This parameter should probably never be as low as 1, as it would be a flat distribution,  rendering the distribution unidentifiable (since the same pattern could be observed with another combination of parameters).\n\nSee BetaPhi2 for more details about the parameters.\n\nDetails\n\nBeta Phi2 is a variant of the traditional Mu-Phi location-precision parametrization.  The modification - scaling ϕ by a factor of 1/2 - creates in a Beta distribution in which, when μ is at its center (i.e., 0.5), a parameter ϕ equal to 1 results in a flat prior (i.e., Beta(1 1)). It is useful to set priors for ϕ on the log scale in regression models, so that a prior of Normal(0 1) assigns the most probability on a flat distribution (ϕ=1).\n\n(Image: )\n\nIn the case of responses clustered in the middle of the scale (at 0.5), in this possible to add a third (non-scaled) Beta distribution centered around 0.5.\n\n(Image: )\n\nExamples\n\njulia> Choco(p1=0.5, μ0=0.7, ϕ0=2, μ1=0.7, ϕ1=2)\nChoco{Float64}(p1=0.5, μ0=0.7, ϕ0=2.0, μ1=0.7, ϕ1=2.0, p_mid=0.0, ϕ_mid=100.0)\n\n\n\n\n\n","category":"type"},{"location":"Choco/#Usage","page":"Choice-Confidence (Choco) Model","title":"Usage","text":"","category":"section"},{"location":"Choco/#Simulate-Data","page":"Choice-Confidence (Choco) Model","title":"Simulate Data","text":"","category":"section"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"tip: Summary\nYou can use rand(dist, n) to generate n observations from a Choco() distribution with pre-specified parameters.","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"Let's generate some data from a Choco() distribution with known parameters that we are going to try to recover using Bayesian modelling.","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"using DataFrames\nusing Random\nusing Turing\nusing CairoMakie\nusing StatsFuns: logistic\nusing SubjectiveScalesModels","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"Random.seed!(123)\n\ny = rand(Choco(p1=0.3, μ0=0.7, ϕ0=3, μ1=0.4, ϕ1=2), 10000)\n\nhist(y, bins=100,  normalization=:pdf, color=:darkorange)","category":"page"},{"location":"Choco/#Prior-Specification","page":"Choice-Confidence (Choco) Model","title":"Prior Specification","text":"","category":"section"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"Deciding on priors requires a good understanding of the meaning of the parameters of the BetaPhi2 distribution on which the Choco model is based. Make sure you first read the documentation page about priors of the BetaPhi2() distribution.","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"The parameters of the Choco() distribution have the following requirements:","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"p0, μ0 and μ1: Must be in the interval 0-1.\nϕ0 and ϕ1: Must be positive (with a special value at 1 where the distribution is flat when μ is at 0.5).","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"Because of these specificities, it this convenient to express priors on a different scale (the logit scale for p0, μ0 and μ1, and the log scale for ϕ0 and ϕ1) and then transform them using a logistic or exponential link functions.","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"<details><summary>See code</summary>","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"p1 =  Normal(0, 2.0)\nμ0 = Normal(0, 1.0)\nμ1 = Normal(0, 0.8)\nϕ0 = Normal(0, 1.0)\nϕ1 = Normal(0, 0.5)\n\nfig =  Figure(size = (850, 600))\n\nax1 = Axis(fig[1, 1], \n    xlabel=\"Prior on the logit scale\",\n    ylabel=\"Distribution\",\n    yticksvisible=false,\n    xticksvisible=false,\n    yticklabelsvisible=false)\n\nxaxis1 = range(-10, 10, 1000)\n\nlines!(ax1, xaxis1, pdf.(p1, xaxis1), color=:purple, linewidth=2, label=\"p1 ~ Normal(0, 2)\")\naxislegend(ax1; position=:rt)\n\nax2 = Axis(fig[1, 2], \n    xlabel=\"Prior after logistic transformation\",\n    yticksvisible=false,\n    xticksvisible=false,\n    yticklabelsvisible=false)\nlines!(ax2, logistic.(xaxis1), pdf.(p1, xaxis1), color=:purple, linewidth=2, label=\"p1\")\n\nax3 = Axis(fig[2, 1], \n    xlabel=\"Prior on the logit scale\",\n    ylabel=\"Distribution\",\n    yticksvisible=false,\n    xticksvisible=false,\n    yticklabelsvisible=false)\nlines!(ax3, xaxis1, pdf.(μ0, xaxis1), color=:blue, linewidth=2, label=\"μ0 ~ Normal(0, 1)\")\nlines!(ax3, xaxis1, pdf.(μ1, xaxis1), color=:red, linewidth=2, label=\"μ1 ~ Normal(0, 0.8)\")\naxislegend(ax3; position=:rt)\n\nax4 = Axis(fig[2, 2], \n    xlabel=\"Prior after logistic transformation\",\n    yticksvisible=false,\n    xticksvisible=false,\n    yticklabelsvisible=false)\nlines!(ax4, logistic.(xaxis1), pdf.(μ0, xaxis1), color=:blue, linewidth=2, label=\"μ0\")\nlines!(ax4, logistic.(xaxis1), pdf.(μ1, xaxis1), color=:red, linewidth=2, label=\"μ1\")\n\nax5 = Axis(fig[3, 1], \n    xlabel=\"Prior on the log scale\",\n    ylabel=\"Distribution\",\n    yticksvisible=false,\n    xticksvisible=false,\n    yticklabelsvisible=false)\nlines!(ax5, xaxis1, pdf.(ϕ0, xaxis1), color=:green, linewidth=2, label=\"ϕ0 ~ Normal(0, 1)\")\nlines!(ax5, xaxis1, pdf.(ϕ1, xaxis1), color=:orange, linewidth=2, label=\"ϕ1 ~ Normal(0, 0.5)\")\naxislegend(ax5; position=:rt)\n\nax6 = Axis(fig[3, 2], \n    xlabel=\"Prior after exponential transformation\",\n    yticksvisible=false,\n    xticksvisible=false,\n    yticklabelsvisible=false)\nvlines!(ax6, [1], color=:black, linestyle=:dash, linewidth=1)\nlines!(ax6, exp.(xaxis1), pdf.(ϕ0, xaxis1), color=:green, linewidth=2, label=\"ϕ0\")\nlines!(ax6, exp.(xaxis1), pdf.(ϕ1, xaxis1), color=:orange, linewidth=2, label=\"ϕ1\")\nxlims!(ax6, 0, 10)\n\nfig[0, :] = Label(fig, \"Priors for Choco Models\", fontsize=20, color=:black, font=:bold)\nfig;","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"</details>","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"fig  # hide","category":"page"},{"location":"Choco/#Bayesian-Model-with-Turing","page":"Choice-Confidence (Choco) Model","title":"Bayesian Model with Turing","text":"","category":"section"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"@model function model_choco(y)\n    p1 ~ Normal(0, 2)\n    μ0 ~ Normal(0, 1)\n    μ1 ~ Normal(0, 0.8)\n    ϕ0 ~ Normal(0, 1)\n    ϕ1 ~ Normal(0, 0.5)\n\n    for i in 1:length(y)\n        y[i] ~ Choco(logistic(p1), logistic(μ0), exp(ϕ0), logistic(μ1), exp(ϕ1))\n    end\nend\n\nfit = model_choco(y)\nposteriors = sample(fit, NUTS(), 500)\n\n# 95% CI\nhpd(posteriors)","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"Let us do a Posterior Predictive Check which involves the generation of predictions from the model to compare the predicted distribution against the actual observed data.","category":"page"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"# Make predictions\npred = predict(model_choco([missing for _ in 1:length(y)]), posteriors)\npred = Array(pred)\n\nfig = hist(y, bins=100, color=:darkorange, normalization=:pdf)\nfor i in 1:size(pred, 1) # Iterate over each draw\n    density!(pred[i, :], color=(:black, 0), strokecolor=(:dodgerblue, 0.05), strokewidth=1)\nend\nxlims!(0, 1)\nfig","category":"page"},{"location":"Choco/#Recover-Parameters","page":"Choice-Confidence (Choco) Model","title":"Recover Parameters","text":"","category":"section"},{"location":"Choco/","page":"Choice-Confidence (Choco) Model","title":"Choice-Confidence (Choco) Model","text":"posterior_mean = DataFrame(mean(posteriors))\n\n# Format\nresults = DataFrame(\n    Parameter = posterior_mean.parameters,\n    Posterior_Mean = round.(posterior_mean.mean; digits=2),\n    Estimate = round.([\n        logistic(posterior_mean.mean[1]), \n        logistic(posterior_mean.mean[2]),\n        logistic(posterior_mean.mean[3]),\n        exp(posterior_mean.mean[4]),\n        exp(posterior_mean.mean[5])\n        ]; digits=2),\n    Truth = [0.3, 0.7, 1, 0.3, 3]\n)\n\nresults","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SubjectiveScalesModels","category":"page"},{"location":"#SubjectiveScalesModels","page":"Home","title":"SubjectiveScalesModels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Docs) (Image: Build Status) (Image: Coverage)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Toolbox for modelling scores from subjective scales (Likert scales, analog scales, ...).  This package's functions are demonstrated in the Cognitive Models book.","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"using Pkg\n\nPkg.add(\"SubjectiveScalesModels\")","category":"page"},{"location":"#Table-of-Content","page":"Home","title":"Table of Content","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
