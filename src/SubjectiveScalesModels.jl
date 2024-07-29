"""
    SubjectiveScalesModels

"""
module SubjectiveScalesModels


export data_rescale
export BetaPhi2
export Choco
export OrderedBeta


include("data_rescale.jl")
include("BetaPhi2.jl")
include("Choco.jl")
include("OrderedBeta.jl")

end
