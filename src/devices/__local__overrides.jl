function Devices.propagator(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.Bare,
    τ::Real;
    result=nothing,
)
    m = nlevels(device)
    n = nqubits(device)
    ū = array(Complex{eltype(device)}, (m,m,n), LABEL)
    ū = localdrivepropagators(device, τ, op.t; result=ū)
    return LinearAlgebraTools.kron(ū; result=result)
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
    u = array(Complex{eltype(device)}, (m, m), LABEL)
    u = driveoperator(device, ā, op.i, op.t; result=u)
    u = LinearAlgebraTools.cis!(u, -τ)
    return globalize(device, u, q; result=result)
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
    ū = array(Complex{eltype(device)}, (m,m,n), LABEL)
    ū = localdrivepropagators(device, τ, op.t; result=ū)
    return LinearAlgebraTools.rotate!(ū, ψ)
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
    identity = Matrix(I, m, m)
    ops = array(Complex{eltype(device)}, (m,m,n), LABEL)
    for p in 1:n
        if p == q
            driveoperator(device, ā, op.i, op.t; result=@view(ops[:,:,p]))
            LinearAlgebraTools.cis!(@view(ops[:,:,p]), -τ)
        else
            ops[:,:,p] .= identity
        end
    end
    return LinearAlgebraTools.rotate!(ops, ψ)
end

function Devices.braket(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.Bare,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    return sum(
        braket(
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
    identity = Matrix(I, m, m)
    ops = array(Complex{eltype(device)}, (m,m,n), LABEL)
    for p in 1:n
        if p == q
            driveoperator(device, ā, op.i, op.t; result=@view(ops[:,:,p]))
        else
            ops[:,:,p] .= identity
        end
    end
    return LinearAlgebraTools.braket(ψ1, ops, ψ2)
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
    identity = Matrix(I, m, m)
    ops = array(Complex{eltype(device)}, (m,m,n), LABEL)
    for p in 1:n
        if p == q
            gradeoperator(device, ā, op.j, op.t; result=@view(ops[:,:,p]))
        else
            ops[:,:,p] .= identity
        end
    end
    return LinearAlgebraTools.braket(ψ1, ops, ψ2)
end