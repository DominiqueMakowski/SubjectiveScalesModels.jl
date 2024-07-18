using SubjectiveScaleModels
using Documenter

DocMeta.setdocmeta!(SubjectiveScaleModels, :DocTestSetup, :(using SubjectiveScaleModels); recursive=true)

makedocs(;
    modules=[SubjectiveScaleModels],
    authors="Dominique Makowski",
    sitename="SubjectiveScaleModels.jl",
    format=Documenter.HTML(;
        canonical="https://DominiqueMakowski.github.io/SubjectiveScaleModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DominiqueMakowski/SubjectiveScaleModels.jl",
    devbranch="main",
)
