"""
    Evolutions

Algorithms to run time evolution, and related constructs like gradient signals.

"""
module Evolutions
    include("Evolutions/__abstractinterface.jl")
        export EvolutionType
        export evolve!, workbasis, gradientsignals
    include("Evolutions/__concreteinterface.jl")
        export evolve
        # Also implements certain method signatures from the abstract interface.
    include("Evolutions/__validation.jl")
end