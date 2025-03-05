using .Devices: DeviceType
using .Devices: ndrives, nlevels, nqubits, nstates
using .Devices: localalgebra, globalalgebra, dress
using .Devices: qubithamiltonian, staticcoupling, driveoperator, gradeoperator

import ..CtrlVQE: LAT
import ..CtrlVQE: Bases, Operators

import TemporaryArrays: @temparray

import Memoization: @memoize

import LinearAlgebra: Diagonal

"""
    operator(op, device[, basis]; kwargs...)

A Hermitian operator describing a `device`, represented in the given `basis`.

For example, to construct the static Hamiltonian of a device in the dressed basis,
    call `operator(Operators.STATIC, device, Bases.DRESSED)`.

# Arguments
- `op::Operators.OperatorType`: which operator to construct (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operator will be represented in.
        Defaults to `Bases.BARE` when omitted.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    Omitting `result` returns an array of type `Complex{eltype(device)}`.
    For static operators only, omitting `result` will return a cached result.

"""
function operator(op::Operators.OperatorType, device::DeviceType; kwargs...)
    return operator(op, device, Bases.BARE; kwargs...)
end

@memoize Dict function _operator(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
)
    N = nstates(device)
    result = Matrix{Complex{eltype(device)}}(undef, N, N)
    return operator(op, device, basis; result=result)
end

function operator(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return _operator(op, device, basis)
    LAT.basisvectors(size(result,1); result=result)
    return result
end

function operator(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return _operator(op, device, basis)
    ā = globalalgebra(device, basis)
    return qubithamiltonian(device, ā, op.q; result=result)
end

function operator(
    op::Operators.Coupling,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return _operator(op, device, basis)
    ā = globalalgebra(device, basis)
    return staticcoupling(device, ā; result=result)
end

function operator(
    op::Operators.Channel,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    ā = globalalgebra(device, basis)
    return driveoperator(device, ā, op.i, op.t; result=result)
end

function operator(
    op::Operators.Gradient,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    ā = globalalgebra(device, basis)
    return gradeoperator(device, ā, op.j, op.t; result=result)
end

function operator(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return _operator(op, device, basis)
    result .= 0
    for q in 1:nqubits(device)
        result .+= operator(Operators.Qubit(q), device, basis)
    end
    return result
end

function operator(
    op::Operators.Static,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return _operator(op, device, basis)
    result .= 0
    result .+= operator(Operators.UNCOUPLED, device, basis)
    result .+= operator(Operators.COUPLING, device, basis)
    return result
end

function operator(
    op::Operators.Static,
    device::DeviceType,
    ::Bases.Dressed;
    result=nothing,
)
    isnothing(result) && return _operator(op, device, Bases.DRESSED)
    ΛU = dress(device)
    result .= Diagonal(ΛU.values)
    return result
end

function operator(
    op::Operators.Drive,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    if isnothing(result)
        N = nstates(device)
        result = Matrix{Complex{eltype(device)}}(undef, N, N)
    end
    result .= 0
    intermediate = @temparray(eltype(result), size(result), :driveoperator)

    for i in 1:ndrives(device)
        intermediate = operator(
            Operators.Channel(i, op.t), device, basis;
            result=intermediate,
        )
        result .+= intermediate
    end
    return result
end

function operator(
    op::Operators.Hamiltonian,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    if isnothing(result)
        N = nstates(device)
        result = Matrix{Complex{eltype(device)}}(undef, N, N)
    end
    result = operator(Operators.Drive(op.t), device, basis; result=result)
    result .+= operator(Operators.STATIC, device, basis)
    return result
end

"""
    localqubitoperators(device[, basis]; kwargs...)

A matrix list `h̄`, where each `h̄[:,:,q]` represents
    a local qubit hamiltonian in the bare basis.

# Arguments
- `device::DeviceType`: which device is being described.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    Omitting `result` will return a cached result, with type `Complex{eltype(device)}`.

"""
function localqubitoperators(
    device::DeviceType;
    result=nothing,
)
    isnothing(result) && return _localqubitoperators(device)

    ā = localalgebra(device)
    for q in 1:nqubits(device)
        result[:,:,q] .= qubithamiltonian(device, ā, q)
    end
    return result
end

@memoize Dict function _localqubitoperators(
    device::DeviceType,
)
    m = nlevels(device)
    n = nqubits(device)
    result = Array{Complex{eltype(device)},3}(undef, m, m, n)
    return localqubitoperators(device; result=result)
end