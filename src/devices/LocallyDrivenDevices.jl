import LinearAlgebra: I
import ..Bases, ..Operators, ...LinearAlgebraTools, ...Devices

import ...TempArrays: array
const LABEL = Symbol(@__MODULE__)

"""
    LocallyDrivenDevice

Super-type for device objects whose drive channels act locally on individual qubits.

Inherit from this type if your `driveoperator` and `gradeoperator` methods
    depend only on a single annihilation operator `ā[:,:,q]`.
This enables more efficient propagation methods which exploit a tensor product structure.

# Implementation

Any concrete sub-type `D` must implement *everything* required in the `Device` interface,
    so consult the documentation for `Device` carefully.

In addition, the following methods must be implemented:
- `drivequbit(::D, i::Int)`: index of the qubit on which channel `i` is applied.
- `gradequbit(::D, j::Int)`: index of the qubit associated with the jth gradient operator.

It's usually trivial to infer the channel index i associated with each gradient operator,
    in which case `gradequbit(device, j) = drivequbit(device, i)`,
    but this is left as an implementation detail.

"""
abstract type LocallyDrivenDevice <: Devices.Device end

"""
    drivequbit(device, i::Int)

Index of the qubit on which channel `i` is applied.

"""
function drivequbit(::LocallyDrivenDevice, i::Int)
    error("Not Implemented")
    return 0
end

"""
    gradequbit(device, j::Int)

Index of the qubit associated with the jth gradient operator.

"""
function gradequbit(::LocallyDrivenDevice, j::Int)
    error("Not Implemented")
    return 0
end

"""
    localdriveoperators(device[, basis], t; kwargs...)

A matrix list `v̄`, where `v̄[:,:,q]` represents a sum of all drives acting on qubit `q`.

# Arguments
- `device::Device`: which device is being described.
- `basis::Bases.BasisType`: which basis the operators will be represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `t::Real`: the time each drive operator is evaluated at.

"""
function localdriveoperators(device::LocallyDrivenDevice, t::Real; kwargs...)
    return localdriveoperators(device, Bases.OCCUPATION, t; kwargs...)
end

function localdriveoperators(
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    t::Real;
    result=nothing,
)
    F = Devices.eltype(Operators.Drive(t), device, basis)
    ā = Devices.localalgebra(device, basis)

    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    isnothing(result) && (result = Array{F,3}(undef, m, m, n))
    result .= 0
    for i in 1:n
        q = drivequbit(device, i)
        result[:,:,q] .+= Devices.driveoperator(device, ā, i, t)
    end
    return result
end

"""
    localdrivepropagators(device[, basis], τ, t; kwargs...)

A matrix list `ū`, where `ū[:,:,q]` is the propagator for a local drive term.

# Arguments
- `device::Device`: which device is being described.
- `basis::Bases.BasisType`: which basis the operators will be represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `τ::Real`: the amount to move forward in time by.
        Note that the propagation is only approximate for time-dependent operators.
        The smaller `τ` is, the more accurate the approximation.
- `t::Real`: the time each drive operator is evaluated at.

"""
function localdrivepropagators(device::LocallyDrivenDevice, τ::Real, t::Real; kwargs...)
    return localdrivepropagators(device, Bases.OCCUPATION, τ, t; kwargs...)
end

function localdrivepropagators(
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    τ::Real,
    t::Real;
    result=nothing,
)
    F = Devices.eltype(Operators.Drive(t), device, basis)

    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    isnothing(result) && (result = Array{F,3}(undef, m, m, n))
    result = localdriveoperators(device, basis, t; result=result)
    for q in 1:n
        LinearAlgebraTools.cis!(@view(result[:,:,q]), -τ)
    end
    return result
end

#=

The remainder of this module overrides `Device` methods
    to exploit tensor structure in the drives.

=#

function Devices.propagator(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    τ::Real;
    result=nothing,
)
    F = LinearAlgebraTools.cis_type(Devices.eltype(op, device, basis))
    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    ū = array(F, (m,m,n), LABEL)
    ū = localdrivepropagators(device, basis, τ, op.t; result=ū)
    return LinearAlgebraTools.kron(ū; result=result)
end

function Devices.propagator(
    op::Operators.Channel,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    τ::Real;
    result=nothing,
)
    F = LinearAlgebraTools.cis_type(Devices.eltype(op, device, basis))
    ā = Devices.localalgebra(device, basis)
    q = drivequbit(device, op.i)

    m = Devices.nlevels(device)
    u = array(F, (m, m), LABEL)
    u .= Devices.driveoperator(device, ā, op.i, op.t)
    u = LinearAlgebraTools.cis!(u, -τ)
    return Devices.globalize(device, u, q; result=result)
end

function Devices.propagate!(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    τ::Real,
    ψ::Devices.Evolvable,
)
    F = LinearAlgebraTools.cis_type(Devices.eltype(op, device, basis))
    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    ū = array(F, (m,m,n), LABEL)
    ū = localdrivepropagators(device, basis, τ, op.t; result=ū)
    return LinearAlgebraTools.rotate!(ū, ψ)
end

function Devices.propagate!(
    op::Operators.Channel,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    τ::Real,
    ψ::Devices.Evolvable,
)
    F = LinearAlgebraTools.cis_type(Devices.eltype(op, device, basis))
    ā = Devices.localalgebra(device, basis)
    q = drivequbit(device, op.i)

    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    ops = array(F, (m,m,n), LABEL)
    for p in 1:n
        if p == q
            Devices.driveoperator(device, ā, op.i, op.t; result=@view(ops[:,:,p]))
            LinearAlgebraTools.cis!(@view(ops[:,:,p]), -τ)
        else
            ops[:,:,p] .= Matrix(I, m, m)
        end
    end
    return LinearAlgebraTools.rotate!(ops, ψ)
end

function Devices.braket(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    return sum(
        Devices.braket(
            Operators.Channel(i, op.t),
            device, basis, ψ1, ψ2
        ) for i in 1:Devices.ndrives(device)
    )
end

function Devices.braket(
    op::Operators.Channel,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    F = Devices.eltype(op, device, basis)
    ā = Devices.localalgebra(device, basis)
    q = drivequbit(device, op.i)

    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    ops = array(F, (m,m,n), LABEL)
    for p in 1:n
        if p == q
            Devices.driveoperator(device, ā, op.i, op.t; result=@view(ops[:,:,p]))
        else
            ops[:,:,p] .= Matrix(I, m, m)
        end
    end
    return LinearAlgebraTools.braket(ψ1, ops, ψ2)
end

function Devices.braket(
    op::Operators.Gradient,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    F = Devices.eltype(op, device, basis)
    ā = Devices.localalgebra(device, basis)
    q = gradequbit(device, op.j)

    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    ops = array(F, (m,m,n), LABEL)
    for p in 1:n
        if p == q
            Devices.gradeoperator(device, ā, op.j, op.t; result=@view(ops[:,:,p]))
        else
            ops[:,:,p] .= Matrix(I, m, m)
        end
    end
    return LinearAlgebraTools.braket(ψ1, ops, ψ2)
end