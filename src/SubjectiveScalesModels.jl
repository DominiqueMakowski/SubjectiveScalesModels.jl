"""
    SubjectiveScalesModels

"""
module SubjectiveScalesModels


export data_rescale
export BetaPhi2
export OrderedBeta
export Choco
export beta_bins


include("data_rescale.jl")
include("BetaPhi2.jl")
include("OrderedBeta.jl")
include("Choco.jl")
include("beta_bins.jl")
end
