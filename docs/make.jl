using CannotWaitForTheseOptimisers
using Documenter

DocMeta.setdocmeta!(CannotWaitForTheseOptimisers, :DocTestSetup, :(using CannotWaitForTheseOptimisers); recursive=true)

makedocs(;
    modules=[CannotWaitForTheseOptimisers],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="CannotWaitForTheseOptimisers.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/CannotWaitForTheseOptimisers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/CannotWaitForTheseOptimisers.jl",
    devbranch="main",
)
