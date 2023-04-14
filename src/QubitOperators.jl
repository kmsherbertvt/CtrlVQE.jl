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




nqubits(H::AbstractVecOrMat) = round(Int, log2(size(H,1)))

function reference(H::AbstractMatrix)
    ix = argmin(diag(real.(H)))
    LinearAlgebraTools.basisvector(size(H,1), ix)
end










function project(ψ::AbstractVector{F}, device::Devices.Device) where {F}
    N0 = length(ψ)
    m̄0 = fill(2, Devices.nqubits(device))

    N = Devices.nstates(device)
    m̄ = [Devices.nstates(device,q) for q in 1:Devices.nqubits(device)]
    ix_map = Dict(i0 => _ix_from_cd(_cd_from_ix(i0,m̄0),m̄) for i0 in 1:N0)
    result = zeros(F, N)
    for i in 1:N0
        result[ix_map[i]] = ψ[i]
    end
    return result
end

function project(H::AbstractMatrix{F}, device::Devices.Device) where {F}
    N0 = size(H, 1)
    m̄0 = fill(2, Devices.nqubits(device))

    N = Devices.nstates(device)
    m̄ = [Devices.nstates(device,q) for q in 1:Devices.nqubits(device)]
    ix_map = Dict(i0 => _ix_from_cd(_cd_from_ix(i0,m̄0),m̄) for i0 in 1:N0)
    result = zeros(F, N, N)
    for i in 1:N0
        for j in 1:N0
            result[ix_map[i],ix_map[j]] = H[i,j]
        end
    end
    return result
end


function driveframe(H::AbstractMatrix, device::Devices.Device, T::Real)
    O = project(H, device)
    O = Devices.evolve!(Operators.STATIC, device, T, O)
    return O
end

function interactionframe(H::AbstractMatrix, device::Devices.Device, T::Real)
    O = project(H, device)
    O = Devices.evolve!(Operators.UNCOUPLED, device, T, O)
    return O
end







function _cd_from_ix(i::Int, m̄::AbstractVector{<:Integer})
    i = i - 1       # SWITCH TO INDEXING FROM 0
    ī = Vector{Int}(undef, length(m̄))
    for q in eachindex(m̄)
        i, ī[q] = divrem(i, m̄[q])
    end
    return ī
end

function _ix_from_cd(ī::AbstractVector{<:Integer}, m̄::AbstractVector{<:Integer})
    i = 0
    offset = 1
    for q in eachindex(m̄)
        i += offset * ī[q]
        offset *= m̄[q]
    end
    return i + 1    # SWITCH TO INDEXING FROM 1
end