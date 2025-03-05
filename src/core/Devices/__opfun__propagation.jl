using .Devices: DeviceType, Evolvable
using .Devices: nlevels, nqubits, nstates
using .Devices: globalize, operator, localqubitoperators

import ..CtrlVQE: LAT
import ..CtrlVQE: Bases, Operators

import TemporaryArrays: @temparray

import Memoization: @memoize

"""
    propagator(op, device[, basis], τ; kwargs...)

A unitary propagator describing evolution under a Hermitian operator for a small time τ.

# Arguments
- `op::Operators.OperatorType`: which operator to evolve under (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operator will be represented in.
        Defaults to `Bases.BARE` when omitted.
- `τ::Real`: the amount to move forward in time by.
        Note that the propagation is only approximate for time-dependent operators.
        The smaller `τ` is, the more accurate the approximation.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    Omitting `result` returns an array of type `Complex{eltype(device)}`.
    For static operators only, omitting `result` will return a cached result.

"""
function propagator(op::Operators.OperatorType, device::DeviceType, τ::Real; kwargs...)
    return propagator(op, device, Bases.BARE, τ; kwargs...)
end

@memoize Dict function _propagator(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real,
)
    N = nstates(device)
    result = Matrix{Complex{eltype(device)}}(undef, N, N)
    return propagator(op, device, basis, τ; result=result)
end

function propagator(
    op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real;
    result=nothing,
)
    N = nstates(device)
    H = @temparray(Complex{eltype(device)}, (N,N), :propagator)
    H = operator(op, device, basis; result=H)

    isnothing(result) && (result=similar(H))
    result .= H
    return LAT.cis!(result, -τ)
end

function propagator(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real;
    result=nothing,
)
    isnothing(result) && return _propagator(op, device, basis, τ)
    result .= _operator(op, device, basis)
    return LAT.cis!(result, -τ)
end

function propagator(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real;
    result=nothing,
)
    isnothing(result) && return _propagator(op, device, basis, τ)
    result .= _operator(op, device, basis)
    result .*= exp(-im*τ)   # Include global phase.
    return result
end

function propagator(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.Bare,
    τ::Real;
    result=nothing,
)
    isnothing(result) && return _propagator(op, device, basis, τ)
    ū = localqubitpropagators(device, τ)
    return LAT.kron(ū; result=result)
end

function propagator(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.Bare,
    τ::Real;
    result=nothing,
)
    isnothing(result) && return _propagator(op, device, basis, τ)
    ū = localqubitpropagators(device, τ)
    return globalize(device, @view(ū[:,:,op.q]), op.q; result=result)
end

"""
    localqubitpropagators(device, τ; kwargs...)

A matrix list `ū`, where each `ū[:,:,q]` is a propagator
    for a local qubit hamiltonian in the bare basis.

# Arguments
- `device::DeviceType`: which device is being described.
- `τ::Real`: the amount to move forward in time by.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    Omitting `result` will return a cached result, with type `Complex{eltype(device)}`.

"""
function localqubitpropagators(
    device::DeviceType,
    τ::Real;
    result=nothing,
)
    isnothing(result) && return _localqubitpropagators(device, τ)

    result = localqubitoperators(device; result=result)
    for q in 1:nqubits(device)
        LAT.cis!(@view(result[:,:,q]), -τ)
    end
    return result
end

@memoize Dict function _localqubitpropagators(
    device::DeviceType,
    τ::Real,
)
    m = nlevels(device)
    n = nqubits(device)
    result = Array{Complex{eltype(device)},3}(undef, m, m, n)
    return localqubitpropagators(device, τ, result=result)
end

"""
    propagate!(op, device[, basis], τ, ψ)

Propagate a state `ψ` by a small time `τ` under the Hermitian `op` describing a `device`.

# Arguments
- `op::Operators.OperatorType`: which operator to evolve under (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the state `ψ` is represented in.
        Defaults to `Bases.BARE` when omitted.
- `τ::Real`: the amount to move forward in time by.
        Note that the propagation is only approximate for time-dependent operators.
        The smaller `τ` is, the more accurate the approximation.
- `ψ`: Either a vector or a matrix, defined over the full Hilbert space of the device.

"""
function propagate!(
    op::Operators.OperatorType, device::DeviceType, τ::Real, ψ::Evolvable;
    kwargs...
)
    return propagate!(op, device, Bases.BARE, τ, ψ; kwargs...)
end

function propagate!(
    op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real,
    ψ::Evolvable,
)
    N = nstates(device)
    U = @temparray(Complex{eltype(device)}, (N,N), :propagate)
    U = propagator(op, device, basis, τ; result=U)
    return LAT.rotate!(U, ψ)
end

function propagate!(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real,
    ψ::Evolvable,
)
    U = _propagator(op, device, basis, τ)
    return LAT.rotate!(U, ψ)
end

function propagate!(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real,
    ψ::Evolvable,
)
    isa(ψ, AbstractMatrix) && return ψ  # CONJUGATION CANCELS ACCUMULATED PHASE

    ψ .*= exp(-im*τ)   # Include global phase.
    return ψ
end

function propagate!(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.Bare,
    τ::Real,
    ψ::Evolvable,
)
    ū = localqubitpropagators(device, τ)
    return LAT.rotate!(ū, ψ)
end

function propagate!(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.Bare,
    τ::Real,
    ψ::Evolvable,
)
    m = nlevels(device)
    n = nqubits(device)
    ops = @temparray(Complex{eltype(device)}, (m,m,n), :qubitpropagate)
    ops .= localqubitpropagators(device, τ)
    for p in 1:n
        p == op.q && continue
        LAT.basisvectors(m; result=@view(ops[:,:,p]))
    end
    return LAT.rotate!(ops, ψ)
end