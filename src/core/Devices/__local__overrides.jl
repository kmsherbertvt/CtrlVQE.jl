using .Devices: LocallyDrivenDevice, Evolvable
using .Devices: drivequbit, gradequbit, ndrives, nlevels, nqubits
using .Devices: localdrivepropagators, localalgebra, driveoperator, gradeoperator

import ..CtrlVQE: LAT
import ..CtrlVQE: Bases, Operators
import ..CtrlVQE: Devices

import TemporaryArrays: @temparray

function Devices.propagator(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.Bare,
    τ::Real;
    result=nothing,
)
    m = nlevels(device)
    n = nqubits(device)
    ū = @temparray(Complex{eltype(device)}, (m,m,n), :drivepropagator)
    ū = localdrivepropagators(device, τ, op.t; result=ū)
    return LAT.kron(ū; result=result)
end

function Devices.propagator(
    op::Operators.Channel,
    device::LocallyDrivenDevice,
    basis::Bases.Bare,
    τ::Real;
    result=nothing,
)
    ā = localalgebra(device)
    q = drivequbit(device, op.i)

    m = nlevels(device)
    n = nqubits(device)
    u = @temparray(Complex{eltype(device)}, (m, m), :channelpropagator)
    u = driveoperator(device, ā, op.i, op.t; result=u)
    u = LAT.cis!(u, -τ)
    return LAT.globalize(u, n, q; result=result)
end

function Devices.propagate!(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.Bare,
    τ::Real,
    ψ::Evolvable,
)
    m = nlevels(device)
    n = nqubits(device)
    ū = @temparray(Complex{eltype(device)}, (m,m,n), :drivepropagate)
    ū = localdrivepropagators(device, τ, op.t; result=ū)
    return LAT.rotate!(ū, ψ)
end

function Devices.propagate!(
    op::Operators.Channel,
    device::LocallyDrivenDevice,
    basis::Bases.Bare,
    τ::Real,
    ψ::Evolvable,
)
    ā = localalgebra(device)
    q = drivequbit(device, op.i)

    m = nlevels(device)
    n = nqubits(device)
    ops = @temparray(Complex{eltype(device)}, (m,m,n), :channelpropagate)
    for p in 1:n
        if p == q
            driveoperator(device, ā, op.i, op.t; result=@view(ops[:,:,p]))
            LAT.cis!(@view(ops[:,:,p]), -τ)
        else
            LAT.basisvectors(m; result=@view(ops[:,:,p]))
        end
    end
    return LAT.rotate!(ops, ψ)
end

function Devices.braket(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.Bare,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    return sum(
        Devices.braket(
            Operators.Channel(i, op.t),
            device, basis, ψ1, ψ2
        ) for i in 1:ndrives(device)
    )
end

function Devices.braket(
    op::Operators.Channel,
    device::LocallyDrivenDevice,
    basis::Bases.Bare,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    ā = localalgebra(device)
    q = drivequbit(device, op.i)

    m = nlevels(device)
    n = nqubits(device)
    ops = @temparray(Complex{eltype(device)}, (m,m,n), :channelbraket)
    for p in 1:n
        if p == q
            driveoperator(device, ā, op.i, op.t; result=@view(ops[:,:,p]))
        else
            LAT.basisvectors(m; result=@view(ops[:,:,p]))
        end
    end
    return LAT.braket(ψ1, ops, ψ2)
end

function Devices.braket(
    op::Operators.Gradient,
    device::LocallyDrivenDevice,
    basis::Bases.Bare,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    ā = localalgebra(device)
    q = gradequbit(device, op.j)

    m = nlevels(device)
    n = nqubits(device)
    ops = @temparray(Complex{eltype(device)}, (m,m,n), :gradientbraket)
    for p in 1:n
        if p == q
            gradeoperator(device, ā, op.j, op.t; result=@view(ops[:,:,p]))
        else
            LAT.basisvectors(m; result=@view(ops[:,:,p]))
        end
    end
    return LAT.braket(ψ1, ops, ψ2)
end