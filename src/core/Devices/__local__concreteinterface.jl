import .Devices: LocallyDrivenDevice
import .Devices: drivequbit, ndrives, nlevels, nqubits
import .Devices: localalgebra, driveoperator

import ..CtrlVQE: LAT

import TemporaryArrays: @temparray

"""
    localdriveoperators(device, t; kwargs...)

A matrix list `v̄`, where `v̄[:,:,q]` represents
    a sum of all drives acting on qubit `q` in the bare basis.

# Arguments
- `device::LocallyDrivenDevice`: which device is being described.
- `t::Real`: the time each drive operator is evaluated at.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    Omitting `result` will return an array with type `Complex{eltype(device)}`.

"""
function localdriveoperators(
    device::LocallyDrivenDevice,
    t::Real;
    result=nothing,
)
    ā = localalgebra(device)

    m = nlevels(device)
    n = nqubits(device)
    isnothing(result) && (result = Array{Complex{eltype(device)},3}(undef, m, m, n))
    u = @temparray(Complex{eltype(device)}, (m, m), :localdriveoperators)

    result .= 0
    for i in 1:ndrives(device)
        q = drivequbit(device, i)
        u = driveoperator(device, ā, i, t; result=u)
        result[:,:,q] .+= u
    end
    return result
end

"""
    localdrivepropagators(device, τ, t; kwargs...)

A matrix list `ū`, where `ū[:,:,q]` is the propagator for a local drive term.

# Arguments
- `device::LocallyDrivenDevice`: which device is being described.
- `τ::Real`: the amount to move forward in time by.
        Note that the propagation is only approximate for time-dependent operators.
        The smaller `τ` is, the more accurate the approximation.
- `t::Real`: the time each drive operator is evaluated at.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    Omitting `result` will return an array with type `Complex{eltype(device)}`.

"""
function localdrivepropagators(
    device::LocallyDrivenDevice,
    τ::Real,
    t::Real;
    result=nothing,
)
    m = nlevels(device)
    n = nqubits(device)
    isnothing(result) && (result = Array{Complex{eltype(device)},3}(undef, m, m, n))
    result = localdriveoperators(device, t; result=result)
    for q in 1:n
        LAT.cis!(@view(result[:,:,q]), -τ)
    end
    return result
end