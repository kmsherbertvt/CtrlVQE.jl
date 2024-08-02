#= Contains methods for `evolver`, `localqubitevolvers`, and `evolve!`. =#

"""
    evolver(op, device[, basis], t; kwargs...)

A unitary propagator describing evolution under a Hermitian operator for a time t.

This function is identical to `propagator`,
    except that the argument `t` is considered an absolute time so it is never cached,
    and that it is undefined for time-dependent operators.
It exists solely to perform rotating-frame rotations at every time-step
    without worrying about over-caching.

# Arguments
- `op::Operators.OperatorType`: which operator to evolve under (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operator will be represented in.
        Defaults to `Bases.BARE` when omitted.
- `t::Real`: the amount to move forward in time by.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    Omitting `result` returns an array of type `Complex{eltype(device)}`.

"""
function evolver(op::Operators.OperatorType, device::DeviceType, t::Real; kwargs...)
    return evolver(op, device, Bases.BARE, t; kwargs...)
end

function evolver(
    op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    t::Real;
    result=nothing
)
    error("Not implemented for non-static operator.")
end

function evolver(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    t::Real;
    result=nothing
)
    H = _operator(op, device, basis)
    isnothing(result) && (result=similar(H))
    result .= H
    return LinearAlgebraTools.cis!(result, -t)
end

function evolver(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType,
    t::Real;
    result=nothing,
)
    identity = _operator(op, device, basis)
    isnothing(result) && (result=similar(identity))
    result .= identity
    result .*= exp(-im*t)   # Include global phase.
    return result
end

function evolver(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.Bare,
    t::Real;
    result=nothing
)
    m = nlevels(device)
    n = nqubits(device)
    ū = array(Complex{eltype(device)}, (m,m,n), LABEL)
    ū = localqubitevolvers(device, t; result=ū)
    return LinearAlgebraTools.kron(ū; result=result)
end

function evolver(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.Bare,
    t::Real;
    result=nothing
)
    h̄ = localqubitoperators(device)
    u = array(Complex{eltype(device)}, (m,m), LABEL)
    u .= @view(h̄[:,:,op.q])
    u = LinearAlgebraTools.cis!(u, -t)
    return globalize(device, u, op.q; result=result)
end

"""
    localqubitevolvers(device, τ; kwargs...)

A matrix list `ū`, where each `ū[:,:,q]` is a propagator for a local qubit hamiltonian.

This function is identical to `localqubitevolvers`,
    except that the argument `t` is considered an absolute time so it is never cached.

# Arguments
- `device::DeviceType`: which device is being described.
- `τ::Real`: the amount to move forward in time by.
        Note that the propagation is only approximate for time-dependent operators.
        The smaller `τ` is, the more accurate the approximation.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    Omitting `result` will return an array of type `Complex{eltype(device)}`.

"""
function localqubitevolvers(
    device::DeviceType,
    t::Real;
    result=nothing
)
    m = nlevels(device)
    n = nqubits(device)
    isnothing(result) && (result = Array{Complex{eltype(device)},3}(undef, m, m, n))
    result = localqubitoperators(device; result=result)
    for q in 1:n
        LinearAlgebraTools.cis!(@view(result[:,:,q]), -t)
    end
    return result
end

"""
    evolve!(op, device[, basis], t, ψ)

Propagate a state `ψ` by a time `t` under the Hermitian `op` describing a `device`.

This function is identical to `propagate!`,
    except that the cache is not used for intermediate propagator matrices,
    and that it is undefined for time-dependent operators.
Look to the `Evolutions` module for algorithms compatible with time-dependence!

# Arguments
- `op::Operators.OperatorType`: which operator to evolve under (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the state `ψ` is represented in.
        Defaults to `Bases.BARE` when omitted.
- `t::Real`: the amount to move forward in time by.
- `ψ`: Either a vector or a matrix, defined over the full Hilbert space of the device.

"""
function evolve!(op::Operators.OperatorType, device::DeviceType, t::Real, ψ::Evolvable)
    return evolve!(op, device, Bases.BARE, t, ψ)
end

function evolve!(op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    t::Real,
    ψ::Evolvable,
)
    error("Not implemented for non-static operator.")
end

function evolve!(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    t::Real,
    ψ::Evolvable,
)
    N = nstates(device)
    U = array(Complex{eltype(device)}, (N,N), LABEL)
    U = evolver(op, device, basis, t; result=U)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function evolve!(
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

function evolve!(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.Bare,
    t::Real,
    ψ::Evolvable,
)
    m = nlevels(device)
    n = nqubits(device)
    ū = array(Complex{eltype(device)}, (m,m,n), LABEL)
    ū = localqubitevolvers(device, t; result=ū)
    return LinearAlgebraTools.rotate!(ū, ψ)
end

function evolve!(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.Bare,
    t::Real,
    ψ::Evolvable,
)
    ā = localalgebra(device)
    m = nlevels(device)
    n = nqubits(device)
    identity = Matrix(I, m, m)
    ops = array(Complex{eltype(device)}, (m,m,n), LABEL)
    for p in 1:n
        if p == op.q
            qubithamiltonian(device, ā, op.q; result=@view(ops[:,:,p]))
            LinearAlgebraTools.cis!(@view(ops[:,:,p]), -t)
        else
            ops[:,:,p] .= identity
        end
    end
    return LinearAlgebraTools.rotate!(ops, ψ)
end