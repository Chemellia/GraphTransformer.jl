using GraphTransformer
using Documenter

DocMeta.setdocmeta!(GraphTransformer, :DocTestSetup, :(using GraphTransformer); recursive=true)

makedocs(;
    modules=[GraphTransformer],
    authors="Dwaraknath",
    repo="https://github.com/DwaraknathT/GraphTransformer.jl/blob/{commit}{path}#{line}",
    sitename="GraphTransformer.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://DwaraknathT.github.io/GraphTransformer.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DwaraknathT/GraphTransformer.jl",
)
