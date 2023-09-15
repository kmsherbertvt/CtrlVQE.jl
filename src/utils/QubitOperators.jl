import ..LinearAlgebraTools
import ..Devices

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using LinearAlgebra: I, diag

"""
    qubitprojector(device::Devices.DeviceType)

A projector from the physical Hilbert space of a device onto a logical two-level space.

The projector `Π` does not reduce the *size* of its operand;
    it only removes *support* on the non-logical states.
If you want to change the size, use `qubitisometry` instead.

"""
function qubitprojector(device::Devices.DeviceType)
    return LinearAlgebraTools.kron(localqubitprojectors(device))
end

"""
    qubitisometry(device::Devices.DeviceType)

An isometry from the physical Hilbert space of a device onto a logical two-level space.

If `Φ` is the isometry, and `|ψ⟩` is a statevector living in the full Hilbert space,
    the vector `Φ|ψ⟩` is a smaller vector living in the two-level space.

"""
function qubitisometry(device::Devices.DeviceType)
    # NOTE: Acts on qubit space, projects up to device space.
    return LinearAlgebraTools.kron(localqubitisometries(device))
end


"""
    localqubitprojectors(device::Devices.DeviceType)

A matrix list of local qubit projectors for each individual qubit in the device.

"""
function localqubitprojectors(device::Devices.DeviceType)
    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    π = Matrix(I, m, m)
    for l in 3:m
        π[l,l] = 0
    end
    π̄ = Array{Bool}(undef, m, m, n)
    for q in 1:n
        π̄[:,:,q] .= π
    end
    return π̄
end

"""
    localqubitprojectors(device::Devices.DeviceType)

A matrix list of local qubit isometries for each individual qubit in the device.

"""
function localqubitisometries(device::Devices.DeviceType)
    # NOTE: Acts on qubit space, projects up to device space.
    m = Devices.nlevels(device)
    n = Devices.nqubits(device)
    ϕ = Matrix(I, m, 2)
    ϕ̄ = Array{Bool}(undef, m, 2, n)
    for q in 1:n
        ϕ̄[:,:,q] .= ϕ
    end
    return ϕ̄
end

"""
    nqubits(H)

Infer the number of qubits of a statevector or matrix living in a logical qubit space.

In other words, calculates `n` assuming the dimension of `H` is `2^n`.

"""
nqubits(H::AbstractVecOrMat) = round(Int, log2(size(H,1)))

"""
    reference(H)

The basis state which minimizes the matrix `H`.

"""
function reference(H::AbstractMatrix)
    ix = argmin(diag(real.(H)))
    LinearAlgebraTools.basisvector(size(H,1), ix)
end









"""
    project(A, device::Devices.DeviceType)

Extend a statevector or matrix living in a two-level space onto a physical Hilbert space.

"""
function project(ψ::AbstractVector{F}, device::Devices.DeviceType) where {F}
    N0 = length(ψ)
    N = Devices.nstates(device)
    m = Devices.nlevels(device)

    m̄0 = fill(2, Devices.nqubits(device))
    m̄  = fill(m, Devices.nqubits(device))

    ix_map = Dict(i0 => _ix_from_cd(_cd_from_ix(i0,m̄0),m̄) for i0 in 1:N0)
    result = zeros(F, N)
    for i in 1:N0
        result[ix_map[i]] = ψ[i]
    end
    return result
end

function project(H::AbstractMatrix{F}, device::Devices.DeviceType) where {F}
    N0 = size(H, 1)
    m̄ = fill(Devices.nlevels(device), Devices.nqubits(device))
    m̄0 = fill(2, Devices.nqubits(device))

    N = Devices.nstates(device)
    ix_map = Dict(i0 => _ix_from_cd(_cd_from_ix(i0,m̄0),m̄) for i0 in 1:N0)
    result = zeros(F, N, N)
    for i in 1:N0
        for j in 1:N0
            result[ix_map[i],ix_map[j]] = H[i,j]
        end
    end
    return result
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