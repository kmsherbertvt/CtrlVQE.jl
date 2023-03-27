using StaticArrays: SVector, SMatrix, MVector, MMatrix
using Memoization: @memoize
import ..Parameter, ..Bases, ..Operators, ..LinearAlgebraTools, ..Signals, ..Devices


struct TransmonDevice{nQ,nS,nD,F<:AbstractFloat} <: Devices.Device
    ω::SVector{nQ,F}
    δ::SVector{nQ,F}
    G::SMatrix{Devices.Quple,F}

    q::SVector{nD,Int}
    Ω::SVector{nD,Signals.AbstractSignal}   # NOTE: Each Ω[q] is mutable.
    ν::MVector{nD,F}                        # NOTE: ν is mutable.

    function TransmonDevice(
        m::I,
        ω::SVector{nQ,F},
        δ::SVector{nQ,F},
        G::SMatrix{Devices.Quple,F},
        q::SVector{nD,Int},
        Ω::SVector{nD,Signals.AbstractSignal},
        ν::MVector{nD,F},
    ) where {I<:Integer,nQ,nD,F<:AbstractFloat}
        return new{nQ,m,nD,F}(ω,δ,G,q,Ω,ν)
    end
end

#= Other types:

FixedFrequencyTransmonDevice: ν is SVector and not included in parameters.
LinearTransmonDevice: G is replaced by SVector g, efficient static propagate.
a mix of both I guess...
PhasedTransmonDevice: each channel has a real Ω and imaginary Ω, different drives
TransmonDeviceSansRWA: implicitly one channel per qubit, different drive
a linear version I guess


=#

#= Constructors:

Obv. accept AbstractVector in lieu of SVector for all parameters.
Accept Dict{Quple,F} for G.
If q is omitted, defaults to [1..n]. Only works if Ω/ν lengths are n.
If ν is omitted, initializes to ω[q]
If m is omitted, defaults to 2.

=#


#= IMPLEMENT BOOK-KEEPING =#

nqubits(::TransmonDevice{nQ,nS,nD,F}) where {nQ,nS,nD,F} = nQ
nstates(::TransmonDevice{nQ,nS,nD,F}) where {nQ,nS,nD,F} = nS
ndrives(::TransmonDevice{nQ,nS,nD,F}) where {nQ,nS,nD,F} = nD
ngrades(::TransmonDevice{nQ,nS,nD,F}) where {nQ,nS,nD,F} = 2*nD

#= IMPLEMENT OPERATORS =#

@memoize function Devices.localloweringoperator(
    ::TransmonDevice{nQ,nS,nD,F},
    ::Int,
) where {nQ,nS,nD,F}
    a = zeros(F, nS, nS)
    for i ∈ 1:m-1
        a[i,i+1] = √i
    end
    return SMatrix{nS,nS}(a)
end

function qubithamiltonian(
    device::TransmonDevice,
    ā::AbstractVector{AbstractMatrix},
    q::Int,
)
    hq = zero(ā[q])
    hq .+=   device.ω[q]    .* (       ā[q]'* ā[q]       )
    hq .+= (-device.δ[q]/2) .* (ā[q]'* ā[q]'* ā[q] * ā[q])
    return hq
end

function staticcoupling(
    device::TransmonDevice{nQ,nS,nD,F},
    ā::AbstractVector{AbstractMatrix},
) where {nQ,nS,nD,F}
    G = zero(ā[1])  # NOTE: Raises error if device has no qubits.
    for p in 1:nQ; for q in 1:nQ
        G .+= device.G[p,q] .* (ā[p]'* ā[q])
    end; end
    return G
end

function driveoperator(
    device::TransmonDevice{nQ,nS,nD,F},
    ā::AbstractVector{AbstractMatrix},
    i::Int,
    t::Real,
) where {nQ,nS,nD,F}
    V = device.Ω[i] * exp(im*device.ν[i]*t) .* ā[device.q[i]]
    V .+= V'
    return V
end

function gradeoperator(
    device::TransmonDevice{nQ,nS,nD,F},
    ā::AbstractVector{AbstractMatrix},
    j::Int,
    t::Real,
) where {nQ,nS,nD,F}
    q = ((j-1) >> 1) + 1        # If Julia indexed from 0, this could just be j÷2...
    phase = im ^ ((j-1) & 1)    # OPERATOR FOR REAL OR IMAGINARY SIGNAL?
    V = (phase * exp(im*device.ν[q]*t)) .* ā[device.q[q]]
    V .+= V'
    return V
end

function gradient(
    device::TransmonDevice,
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractMatrix,
)::AbstractVector
    grad = [zero(type) for type in Parameter.types(device)]

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    offset = 0
    for (i, Ωi) in enumerate(device.Ω)
        j = 2*(i-1) + 1         # If Julia indexed from 0, this could just be 2i...
        L = Parameter.count(Ωi)
        for k in 1:L
            ∂̄ = Signals.partial(k, Ωi, t̄)
            grad[offset + k] .+= (τ̄ .* real.(∂̄) .* ϕ̄[j,:])
            grad[offset + k] .+= (τ̄ .* imag.(∂̄) .* ϕ̄[j+1,:])
        end
        offset += L
    end

    # CALCULATE GRADIENT FOR FREQUENCY PARAMETERS
    for (i, Ωi) in enumerate(device.Ω)
        j = 2*(i-1) + 1         # If Julia indexed from 0, this could just be 2i...
        Ω̄ = Ωi(t̄)
        grad[offset + i] .+= (τ̄ .* t̄ .* real.(Ω̄) .* ϕ̄[j+1,:])
        grad[offset + i] .-= (τ̄ .* t̄ .* imag.(Ω̄) .* ϕ̄[j,:])

    end

    return grad
end

#= IMPLEMENT PARAMETER INTERFACE =#

function Parameter.count(device::TransmonDevice)
    return sum(Parameter.count(Ωi) for Ωi in device.Ω) + length(device.ν)
end

function Parameter.names(device::TransmonDevice)
    names = []

    # STRING TOGETHER PARAMETER NAMES FOR EACH SIGNAL Ω[i]
    annotate(name,i) = "Ω$i(q$(device.q[i])):$name"
    for i in eachindex(device.Ω)
        append!(names, (annotate(name,i) for name in Parameter.names(device.Ω[i])))
    end

    # TACK ON PARAMETER NAMES FOR EACH ν[i]
    append!(names, ("ν$i" for i in eachindex(device.ν)))
    return names
end

function Parameter.types(device::TransmonDevice)
    types = []

    # STRING TOGETHER PARAMETER TYPES FOR EACH SIGNAL Ω[i]
    for i in eachindex(device.Ω)
        append!(types, (type for type in Parameter.types(device.Ω[i])))
    end

    # TACK ON PARAMETER NAMES FOR EACH ν[i]
    append!(types, (eltype(ν) for i in eachindex(device.ν)))
    return types
end

function Parameter.bind(device::TransmonDevice, x̄::AbstractVector)
    offset = 0

    # BIND PARAMETERS FOR EACH SIGNAL Ω[i]
    for Ωi in device.Ω
        L = Parameter.count(Ωi)
        Parameter.bind(Ωi, x̄[offset+1:offset+L])
        offset += L
    end

    # BIND PARAMETERS FOR EACH ν[i]
    for i in eachindex(device.ν)
        device.ν[i] = x̄[offset+i]
    end
    offset += length(device.ν)
end




#= ADDITIONAL OVERRRIDES

All drive operators are local, so we can simplify calculations for local bases.
Same goes for gradient operators.

=#

# LOCALIZING DRIVE OPERATORS

function localdriveoperators(device::TransmonDevice, t::Real)
    return localdriveoperators(device, Basis.Occupation, t)
end

function localdriveoperators(
    device::TransmonDevice{nQ,nS,nD,F},
    basis::Type{<:Bases.LocalBasis},
    t::Real,
) where {nQ,nS,nD,F}
    ā = Devices.localalgebra(device, basis)
    v̄ = [zero(ā[q]) for q in 1:nQ]
    for i in 1:nD
        v̄[device.q[i]] .+= Devices.driveoperator(device, ā, i, t)
    end
    return v̄
end

function localdrivepropagators(device::TransmonDevice, τ::Real, t::Real)
    return localdrivepropagators(device, Basis.Occupation, τ, t)
end

function localdrivepropagators(
    device::TransmonDevice{nQ,nS,nD,F},
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    t::Real,
) where {nQ,nS,nD,F}
    v̄ = localdriveoperators(device, basis, t)
    return [LinearAlgebraTools.propagator(v̄[q], τ) for q in 1:nQ]
end

function Devices.propagator(::Type{Operators.Drive},
    device::TransmonDevice,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    t::Real,
)
    ū = localdrivepropagators(device, basis, τ, t)
    return LinearAlgebraTools.kron(ū)
end

function Devices.propagator(::Type{Operators.Channel},
    device::TransmonDevice,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    i::Int,
    t::Real,
)
    ā = Devices.localalgebra(device, basis)
    vi = Devices.driveoperator(device, ā, i, t)
    u = LinearAlgebraTools.propagator(vi, τ)
    return globalize(device, u, device.q[i])
end

function Devices.propagate!(::Type{Operators.Drive},
    device::TransmonDevice,
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
    t::Real,
)
    ū = localdrivepropagators(device, basis, τ, t)
    return LinearAlgebraTools.rotate!(ū, ψ)
end

function Devices.propagate!(::Type{Operators.Channel},
    device::TransmonDevice{nQ,nS,nD,F},
    basis::Type{<:Bases.LocalBasis},
    τ::Real,
    ψ::AbstractVecOrMat{<:Complex{<:AbstractFloat}},
    i::Int,
    t::Real,
) where {nQ,nS,nD,F}
    ā = Devices.localalgebra(device, basis)
    vi = Devices.driveoperator(device, ā, i, t)
    u = LinearAlgebraTools.propagator(vi, τ)
    ops = [p == device.q[i] ? u : one(u) for p in 1:nQ]
    return LinearAlgebraTools.rotate!(ops, ψ)
end

# LOCALIZING GRADIENT OPERATORS

function Devices.braket(::Type{Operators.Gradient},
    device::TransmonDevice{nQ,nS,nD,F},
    basis::Type{<:Bases.LocalBasis},
    ψ1::AbstractVector,
    ψ2::AbstractVector,
    j::Int,
    t::Real,
) where {nQ,nS,nD,F}
    ā = Devices.localalgebra(device, basis)
    A = Devices.gradeoperator(device, ā, j, t)
    q = ((j-1) >> 1) + 1
    ops = [p == q ? A : one(A) for p in 1:nQ]
    return LinearAlgebraTools.braket(ψ1, ops, ψ2)
end