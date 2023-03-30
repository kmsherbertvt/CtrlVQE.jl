module TestLinearAlgebraTools
    include("LinearAlgebraTools.jl")
end

module TestSignals
    include("Signals.jl")
end

module TestDevices
    include("Devices.jl")
end

# # TODO
# module TestEvolutions
#     include("Evolutions.jl")
#   # This should compare against single-qubit solutions for m=2, 3 from AnalyticSquarePulse
#   # It should also compare device gradient to a finite difference.
# end