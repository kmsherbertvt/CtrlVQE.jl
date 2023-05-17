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

The script "device_codecheck" gives a nice complete validate(::Device) method.
We just need to decide how to fold it into the testing script,
    and make similar `validate` methods for signals and costfunctions,
    and work out a clever way to change the `@time`s to `@code_warntype`s.

It seems like the JET package ought to help, but it finds more problems than there are...

=#