using .Devices: DeviceType
using .Devices: nlevels, nqubits, nstates
using .Devices: operator, localqubitoperators

import ..CtrlVQE: LAT
import ..CtrlVQE: Bases, Operators

import TemporaryArrays: @temparray

"""
    expectation(op, device[, basis], ψ)

The expectation value of an operator describing a `device` with respect to the state `ψ`.

If ``A`` is the operator specified by `op`, this method calculates ``⟨ψ|A|ψ⟩``.

# Arguments
- `op::Operators.OperatorType`: which operator to estimate (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the state `ψ` is represented in.
        Defaults to `Bases.BARE` when omitted.
- `ψ`: A statevector defined over the full Hilbert space of the device.

"""
function expectation(op::Operators.OperatorType, device::DeviceType, ψ::AbstractVector)
    return expectation(op, device, Bases.BARE, ψ)
end

function expectation(
    op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    ψ::AbstractVector,
)
    return braket(op, device, basis, ψ, ψ)
end




"""
    braket(op, device[, basis], ψ1, ψ2)

The braket of an operator describing a `device` with respect to states `ψ1` and `ψ2`.

If ``A`` is the operator specified by `op`, this method calculates ``⟨ψ1|A|ψ2⟩``.

# Arguments
- `op::Operators.OperatorType`: which operator to estimate (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the state `ψ` is represented in.
        Defaults to `Bases.BARE` when omitted.
- `ψ1`, `ψ2`: Statevectors defined over the full Hilbert space of the device.

"""
function braket(
    op::Operators.OperatorType,
    device::DeviceType,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    return braket(op, device, Bases.BARE, ψ1, ψ2)
end

function braket(
    op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    N = nstates(device)
    H = @temparray(eltype(op, device, basis), (N,N), :braket)
    H = operator(op, device, basis; result=H)
    return LAT.braket(ψ1, H, ψ2)
end

function braket(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    H = _operator(op, device, basis)
    return LAT.braket(ψ1, H, ψ2)
end

function braket(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.Bare,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    return sum(
        braket(Operators.Qubit(q), device, basis, ψ1, ψ2) for q in 1:nqubits(device)
    )
end

function braket(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.Bare,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    m = nlevels(device)
    n = nqubits(device)
    ops = @temparray(Complex{eltype(device)}, (m,m,n), :qubitbraket)
    ops .= localqubitoperators(device)
    for p in 1:n
        p == op.q && continue
        LAT.basisvectors!(@view(ops[:,:,p]))
    end
    return LAT.braket(ψ1, ops, ψ2)
end


