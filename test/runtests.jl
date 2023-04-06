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

#= This script tests for correctness,
    but the `test` directory may also include the benchmarking scripts, I think.

In particular:

TODO (hi): Time/allocations as we scale up n, for increasing r.
TODO (hi): Time/allocations as we scale up r, for increasing n.

TODO (hi): Trotter infidelity plot for single qubit.
TODO (hi): Trotter infidelity plots for increasing n
    (use sufficiently high r for reference such that plot levels out).
TODO (hi): Trotter rms-gradient-error plots for, um, reasonable r.

TODO (mid): A script to thoroughly check for type stability.
    (looks a lot like this one but @code_warntype instead of @time)

=#