using SubjectiveScalesModels
using Documenter

DocMeta.setdocmeta!(SubjectiveScalesModels, :DocTestSetup, :(using SubjectiveScalesModels); recursive=true)

makedocs(;
    modules=[SubjectiveScalesModels],
    authors="Dominique Makowski",
    sitename="SubjectiveScalesModels.jl",
    format=Documenter.HTML(;
        canonical="https://DominiqueMakowski.github.io/SubjectiveScalesModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DominiqueMakowski/SubjectiveScalesModels.jl",
    devbranch="main",
)
