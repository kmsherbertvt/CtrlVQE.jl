using CtrlVQE
using Documenter

DocMeta.setdocmeta!(CtrlVQE, :DocTestSetup, :(using CtrlVQE); recursive=true)

makedocs(;
    modules=[CtrlVQE],
    authors="Kyle Sherbert <kyle.sherbert@vt.edu> and contributors",
    repo="https://github.com/kmsherbertvt/CtrlVQE.jl/blob/{commit}{path}#{line}",
    sitename="CtrlVQE.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://kmsherbertvt.github.io/CtrlVQE.jl",
        edit_link="main",
        assets=String[],
    ),
    # pages=[
    #     "Home" => "index.md",
    # ],
)

deploydocs(;
    repo="github.com/kmsherbertvt/CtrlVQE.jl",
    devbranch="main",
)
