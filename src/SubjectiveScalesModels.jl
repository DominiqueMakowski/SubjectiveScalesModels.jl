"""
    SubjectiveScalesModels

"""
module SubjectiveScalesModels


export data_rescale
export BetaPhi2
export OrderedBeta
export ExtremeBeta
export Choco

include("data_rescale.jl")
include("BetaPhi2.jl")
include("OrderedBeta.jl")
include("ExtremeBeta.jl")
include("Choco.jl")

end
