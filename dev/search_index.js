var documenterSearchIndex = {"docs":
[{"location":"api/#Functions","page":"Functions","title":"Functions","text":"","category":"section"},{"location":"api/#BetaPhi2()","page":"Functions","title":"BetaPhi2()","text":"","category":"section"},{"location":"api/","page":"Functions","title":"Functions","text":"BetaPhi2(μ::Real, ϕ::Real)","category":"page"},{"location":"api/#SubjectiveScalesModels.BetaPhi2-Tuple{Real, Real}","page":"Functions","title":"SubjectiveScalesModels.BetaPhi2","text":"BetaPhi2(μ, ϕ)\n\nConstruct a Beta distribution with parameters mean μ and precision ϕ. It is defined as Beta(μ * 2ϕ, (1 - μ) * 2ϕ).\n\nArguments\n\nμ: Location parameter (range: 0 1).\nϕ: Precision parameter (must be  0).\n\nDetails\n\nBeta Phi2 is a variant of the traditional Mu-Phi parametrization defined as Beta(μ * ϕ (1 - μ) * ϕ) in which, when μ is at its center (i.e., 0.5), a ϕ equal to 1 results in a flat prior (i.e., Beta(1 1)). It is useful to set priors for ϕ on the log scale in regression models, so that a prior of Normal(0 1) assigns the most probability on a flat distribution.\n\n(Image: )\n\nThe red area shows the region where the distribution assigns the highest probability to extreme values (towards 0 and/or 1). The blue area shows the region where the distribution is \"convex\" and peaks within the 0 1 interval.\n\nExamples\n\njulia> BetaPhi2(0.5, 1)\nDistributions.Beta{Float64}(α=1.0, β=1.0)\n\n\n\n\n\n","category":"method"},{"location":"api/#OrderedBeta()","page":"Functions","title":"OrderedBeta()","text":"","category":"section"},{"location":"api/","page":"Functions","title":"Functions","text":"OrderedBeta","category":"page"},{"location":"api/#SubjectiveScalesModels.OrderedBeta","page":"Functions","title":"SubjectiveScalesModels.OrderedBeta","text":"OrderedBeta(μ, ϕ, k1, k2)\n\nThe distribution is defined on the interval [0, 1] with additional point masses at 0 and 1. It is defined as a mixture of a Beta Phi2 distribution and two point masses at 0 and 1.\n\nParameters\n\nμ: location parameter ]0, 1[\nϕ: precision parameter (must be positive)\nk1: first cutpoint\nk2: Difference between the second and first cutpoints\n\nExamples\n\njulia> OrderedBeta(0.5, 1)\nOrderedBeta{Float64}(\nμ: 0.5\nϕ: 1.0\nk1: -6.0\nk2: 12.0\nbeta_dist: Distributions.Beta{Float64}(α=1.0, β=1.0)\n)\n\n\n\n\n\n","category":"type"},{"location":"api/#Other","page":"Functions","title":"Other","text":"","category":"section"},{"location":"api/#data_rescale()","page":"Functions","title":"data_rescale()","text":"","category":"section"},{"location":"api/","page":"Functions","title":"Functions","text":"data_rescale(x::Vector{T}; old_range::Vector{T}=[minimum(x), maximum(x)], new_range::Vector{T}=[0, 1]) where {T<:Number}","category":"page"},{"location":"api/#SubjectiveScalesModels.data_rescale-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Number","page":"Functions","title":"SubjectiveScalesModels.data_rescale","text":"data_rescale(x; old_range=[minimum(x), maximum(x)], new_range=[0, 1])\n\nRescale a variable to a new range. Can be used to normalize a variable between 0 and 1.\n\ndanger: Danger\nThis function is currently used internally and might be moved to another package. Avoid using it directly.\n\nArguments\n\nx: Vector to rescale.\nold_range: Old range of the vector to rescale (will be taken by default from the minimum and maximum value of x).\nnew_range: Range to rescale x to. By default, [0-1].\n\nExamples\n\njulia> data_rescale([1, 2, 3])\n3-element Vector{Float64}:\n 0.0\n 0.5\n 1.0\n\njulia> data_rescale([1, 2, 3]; old_range=[1, 6], new_range=[1, 0])\n3-element Vector{Float64}:\n 1.0\n 0.8\n 0.6\n\n\n\n\n\n","category":"method"},{"location":"api/","page":"Functions","title":"Functions","text":"SubjectiveScalesModels","category":"page"},{"location":"api/#SubjectiveScalesModels","page":"Functions","title":"SubjectiveScalesModels","text":"SubjectiveScalesModels\n\n\n\n\n\n","category":"module"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SubjectiveScalesModels","category":"page"},{"location":"#SubjectiveScalesModels","page":"Home","title":"SubjectiveScalesModels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SubjectiveScalesModels.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
