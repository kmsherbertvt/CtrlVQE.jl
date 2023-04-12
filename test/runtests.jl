module TestLinearAlgebraTools
    include("LinearAlgebraTools.jl")
end

module TestSignals
    include("Signals.jl")
end

module TestDevices
    include("Devices.jl")
end

module TestEvolutions
    include("Evolutions.jl")
end

#= TODO (mid): A script to thoroughly check for type stability.
    (looks a lot like this one but @code_warntype instead of @time)
=#