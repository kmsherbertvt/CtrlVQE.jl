import ..Parameters, ..Devices
export TransmonDevice, FixedFrequencyTransmonDevice

import ..LinearAlgebraTools
import ..Integrations, ..Signals

import ..Signals: SignalType
import ..LinearAlgebraTools: MatrixList
import ..Quples: Quple

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using Memoization: @memoize
using LinearAlgebra: I, mul!
using NLSolversBase: OnceDifferentiable

#= TODO: Change the name.

At present this is workshopping a new possibly-standardizable strategy
    for designing concrete devices.
It involves separation of the static Hamiltonian from the drive Hamiltonians,
    which we had decided to abandon a long time ago.
However I think we can have our cake and eat it too -
    the `Devices` interface makes no separation but the implementations are divorced.

In any case, my current ideas will at the very least unify most of the device types
    we've been meaning to create...

=#


#= TODO: CHANGES FOR THE DEVICES INTERFACE

Not very many, I think!
- `FΩ` is a wholly redundant type parameter in Devices.
    I think it is a relic of when we were strongly typing signals.
    (TransmonDevices will retain it, though.)
- `drivesignal` is not sustainable. Axe it.
- `drivefrequency` may have been a mistake, but we can just use t=0 for backward compatibility if you judge it appropriate. Oh of course just give it a time parameter, defaulting to 0!

=#

include("Algebras.jl")
include("StaticHamiltonians.jl")
include("Channels.jl")

struct MyDevice{
    F, FΩ,   # TODO: Get rid of FΩ
    A <: AlgebraType{F},
    H <: StaticHamiltonianType{F,A},
    C <: LocalChannel{F,A},
} <: Devices.LocallyDrivenDevice{F,FΩ}
    algebra::A
    static::H
    channels::Vector{C}
    x::Vector{F}                    # Device parameters.
    f::Vector{OnceDifferentiable}   # Each channel parameter as a fn of the device's.
end


"""
    update_channels(device::MyDevice)

Update all parameters in each channel to match the current device parameters.

"""
function update_channels!(device::MyDevice)
    offset = 0
    for channel in device.channels
        L = Parameters.count(channel)
        Parameters.bind(channel, map(f->f(device.x), device.f[offset+1:offset+L]))
        offset += L
    end
end












##########################################################################################
#= `Parameters` interface =#

Parameters.count(::MyDevice) = length(device.x)
Parameters.values(device::MyDevice) = copy(device.x)
Parameters.names(device::MyDevice) = ["x$k" for k in eachindex(device.x)]
    #= TODO: This is a fine default but how do we change it for each recipe?
        Seems like MyDevice should have another field, for the list of names! =#

function Parameters.bind(device::MyDevice, x̄::Vector{F})
    device.x .= x̄
    update_channels!(device)
end

##########################################################################################
# Counting methods
Devices.nqubits(device::MyDevice) = Devices.nqubits(device.static)
Devices.nlevels(device::MyDevice) = Devices.nlevels(device.algebra)
Devices.ndrives(device::MyDevice) = length(Devices.channels)
Devices.ngrades(device::MyDevice) = Devices.ndrives(devices) * 2

##########################################################################################
# Algebra methods

function Devices.localloweringoperator(device::MyDevice; result=nothing)
    return Devices.localloweringoperator(device.algebra; result=result)
end

# TODO: This method should actually just replace the default in Devices.
@memoize function Devices.algebra(
    device::MyDevice,
    basis::Bases.BasisType=Bases.OCCUPATION,
)
    F = eltype_algebra(device, basis)
    U = basisrotation(basis, Bases.OCCUPATION, device)

    ā_local = Devices.localalgebra(device)
    n = size(ā_local,3)
    N = nstates(device)
    ā = Array{F,3}(undef, N, N, n)
    for q in 1:n
        ā[:,:,q] .= globalize(device, @view(ā_local[:,:,q]), q)
        LinearAlgebraTools.rotate!(U, @view(ā[:,:,q]))
    end
    return ā
end

@memoize Dict function Devices.localalgebra(
    device::DeviceType,
    basis::Bases.LocalBasis=Bases.OCCUPATION,
)
    ā = Devices.localalgebra(device.algebra)
    basis == Bases.OCCUPATION && return ā

    for q in 1:Devices.nqubits(device)
        u = basisrotation(basis, Bases.OCCUPATION, device, q)
        LinearAlgebraTools.rotate!(u, @view(ā[:,:,q]))
    end
    return ā
end

##########################################################################################
# Operator methods
function Devices.qubithamiltonian(
    device::MyDevice,
    ā::MatrixList,
    q::Int;
    result=nothing,
)
    return Devices.qubithamiltonian(device.static, ā, q; result=result)
end

function Devices.staticcoupling(
    device::MyDevice,
    ā::MatrixList;
    result=nothing,
)
    return Devices.staticcoupling(device.static, ā; result=result)
end

function Devices.driveoperator(
    device::MyDevice,
    ā::MatrixList,
    i::Int,
    t::Real;
    result=nothing,
)
    return Devices.driveoperator(device.channel[i], ā, t; result=result)
end

function Devices.gradeoperator(
    device::MyDevice,
    ā::MatrixList,
    j::Int,
    t::Real;
    result=nothing,
)
    i = ((j-1) >> 1) + 1
    return Devices.gradeoperator(device.channel[i], ā, t; result=result)
end

##########################################################################################
# Type methods

function Devices.eltype_localloweringoperator(device::MyDevice)
    return Devices.eltype_localloweringoperator(device.algebra)
end

function Devices.eltype_qubithamiltonian(device::MyDevice)
    return Devices.eltype_qubithamiltonian(eltype(device.static))
end

function Devices.eltype_staticcoupling(device::MyDevice)
    return Devices.eltype_staticcoupling(eltype(device.static))
end

function Devices.eltype_driveoperator(device::MyDevice)
    return Devices.eltype_driveoperator(eltype(device.channels))
end

function Devices.eltype_gradeoperator(device::MyDevice)
    return Devices.eltype_gradeoperator(eltype(device.channels))
end

##########################################################################################
# Frequency methods

function Devices.resonancefrequency(device::MyDevice, q::Int)
    return Devices.resonancefrequency(device, q)
end

function Devices.drivefrequency(device::MyDevice, i::Int)
    #= TODO: I think probably this method should be redesigned into a signal. =#
end

##########################################################################################
# Signal methods
Devices.__get__drivesignals(::D) = nothing
    #= TODO: Unh-uh. We can't expose drive signals directly, sorry.
        We've got to axe this.

    Adaptive scripts that need to do so just will not be agnostic of the device type.

    I think we probably do need to expose signals for Ω and ν,
        for the sake of global bounds, but I have no ambitions whatsoever
        to allow standardized modification.
    =#

##########################################################################################
# Gradient methods
function Devices.gradient(
    device::MyDevice,
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix;
    result=nothing,
)
    # TODO: the hard part
end

##########################################################################################
# `LocallyDrivenDevice` interface

function Devices.drivequbit(device::MyDevice, i::Int)
    return Devices.drivequbit(device.channel[i])
end

function Devices.gradequbit(device::MyDevice, j::Int)
    i = ((j-1) >> 1) + 1
    return Devices.drivequbit(device.channel[i])
end