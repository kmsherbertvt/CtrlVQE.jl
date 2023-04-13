import LinearAlgebra: I, diag
import ..Operators, ..LinearAlgebraTools, ..Devices

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

function qubitprojector(device::Devices.Device)
    return LinearAlgebraTools.kron(localqubitprojectors(device))
end

function qubitisometry(device::Devices.Device)
    # NOTE: Acts on qubit space, projects up to device space.
    return LinearAlgebraTools.kron(localqubitisometries(device))
end

function localqubitprojectors(device::Devices.Device)
    return Matrix{Bool}[ϕ*ϕ' for ϕ in localqubitisometries(device)]
end

function localqubitisometries(device::Devices.Device)
    # NOTE: Acts on qubit space, projects up to device space.
    ϕ(q) = Matrix(I,Devices.nstates(device,q), 2)
    return Matrix{Bool}[ϕ(q) for q in 1:Devices.nqubits(device)]
end




nqubits(H::AbstractMatrix) = round(Int, log2(size(H,1)))

reference(H::AbstractMatrix) = LinearAlgebraTools.basisvector(size(H,1), argmin(diag(H)))










function bare(H::AbstractMatrix, device::Devices.Device)
    # TODO (hi): I think just move the project methods here. And just assume m=2!!!
    return Devices.project(device, H)
end

function interactionframe(H::AbstractMatrix, T::Real, device::Devices.Device)
    O = bare(H, device)
    O = Devices.evolve!(Operators.STATIC, device, -T, O)
    return O
end

function partialinteractionframe(H::AbstractMatrix, T::Real, device::Devices.Device)
    # TODO (lo): We have to implement tensored rotate! before this is useful.
    O = bare(H, device)
    O = Devices.evolve!(Operators.UNCOUPLED, device, -T, O)
    return O
end