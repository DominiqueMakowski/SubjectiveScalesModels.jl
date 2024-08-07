# SubjectiveScalesModels.jl

[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://DominiqueMakowski.github.io/SubjectiveScalesModels.jl/)
[![Build Status](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/DominiqueMakowski/SubjectiveScalesModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/DominiqueMakowski/SubjectiveScalesModels.jl)


Toolbox for modelling scores from subjective scales (for example Likert or analog scales), common in psychology and cognitive neuroscience.

## Installation

```julia
using Pkg

Pkg.add("SubjectiveScalesModels")
```

## Features

### BetaPhi2() Distribution for Beta Regressions

```julia
BetaPhi2(0.5, 1)
```

- [x] [**Documentation**](https://dominiquemakowski.github.io/SubjectiveScalesModels.jl/dev/BetaPhi2/)

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/animation_BetaPhi2.gif?raw=true)


### Ordered Beta Model

- [x] [**Documentation**](https://dominiquemakowski.github.io/SubjectiveScalesModels.jl/dev/OrderedBeta/)

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/animation_OrderedBeta.gif?raw=true)

### Choice-Confidence (Choco) Model

```julia
Choco(p1=0.3, μ0=0.7, ϕ0=3, μ1=0.4, ϕ1=2)
```

- [x] [**Documentation**](https://dominiquemakowski.github.io/SubjectiveScalesModels.jl/dev/Choco/)

![](https://github.com/DominiqueMakowski/SubjectiveScalesModels.jl/blob/main/docs/img/animation_Choco1.gif?raw=true)



### Zero-One-Inflated Beta Model (ZOIB)

- [ ] To do.

