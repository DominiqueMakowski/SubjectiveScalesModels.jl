using Revise
using Pkg

Pkg.activate("C:/Users/domma/Dropbox/Software/SubjectiveScalesModels.jl/")

using Distributions
using SubjectiveScalesModels
using GLMakie

d = BetaMuPhi(0.5, 2.0)

xaxis = range(0, 1, length=100)
f = lines(xaxis, pdf.(Beta(0.5, 0.5), xaxis))
lines!(xaxis, pdf.(d, xaxis))
xlims!(0, 1)
f