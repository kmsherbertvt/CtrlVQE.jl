module Tutorials
    function run(script)
        include("$(@__DIR__)/../$script.jl")
    end
end