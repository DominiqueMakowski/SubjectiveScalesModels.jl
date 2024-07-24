"""
    SubjectiveScalesModels

"""
module SubjectiveScalesModels


export data_rescale
export BetaPhi2
export Choco
export OrderedBeta
export ExtremeBeta


include("data_rescale.jl")
include("BetaPhi2.jl")
include("Choco.jl")
include("OrderedBeta.jl")
include("ExtremeBeta.jl")

end
