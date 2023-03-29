using Memoization: @memoize
using ReadOnlyArrays: ReadOnlyArray
import ..Parameter, ..Bases, ..Operators, ..LinearAlgebraTools, ..Signals, ..Devices


struct TransmonDevice{F<:AbstractFloat} <: Devices.LocallyDrivenDevice
    m::Int,

    ω::Tuple{Vararg{F}}     # length is number of qubits
    δ::Tuple{Vararg{F}}     # length is number of qubits

    quples::Tuple{Vararg{Devices.Quple}}
    g::Tuple{Vararg{F}}     # length is number of quples

    q::Tuple{Vararg{Int}}   # length is number of drives
    Ω::Tuple{Vararg{Signals.AbstractSignal}}
    ν::Vector{F}

    #= TODO: Inner constructor validates ω, δ have consistent n,
        and all qubits in quples and q are consistent with that length.

        And length of g matches length of quples.

        And length of q matches length of Ω matches length of ν.
    =#
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
Accept Dict{Quple,F} or Matrix{F} for G.
If q is omitted, defaults to [1..n]. Only works if Ω/ν lengths are n.
If ν is omitted, initializes to ω[q]
If m is omitted, defaults to 2.

=#


#= IMPLEMENT BOOK-KEEPING =#

nqubits(device::TransmonDevice) = length(device.ω)
nstates(device::TransmonDevice, q::Int) = device.m
ndrives(device::TransmonDevice) = length(device.q)
ngrades(device::TransmonDevice) = 2*ndrives(device)

drivequbit(device::TransmonDevice, i::Int) = device.q[i]
gradequbit(device::TransmonDevice, j::Int) = device.q[((j-1) >> 1) + 1]





#= IMPLEMENT OPERATORS =#

@memoize function Devices.localloweringoperator(
    device::TransmonDevice{F},
    q::Int,
) where {F}
    a = zeros(F, nstates(device,q), nstates(device,q))
    for i ∈ 1:m-1
        a[i,i+1] = √i
    end
    return ReadOnlyArray(a)
end

function Devices.qubithamiltonian(
    device::TransmonDevice,
    ā::AbstractVector{AbstractMatrix},
    q::Int,
)
    a = ā[q]
    h = zero(a)
    h .+=   device.ω[q]    .* (    a'*a     )
    h .+= (-device.δ[q]/2) .* (a'* a'* a * a)
    return h
end

function Devices.staticcoupling(
    device::TransmonDevice{nQ,nS,nD,F},
    ā::AbstractVector{AbstractMatrix},
) where {nQ,nS,nD,F}
    G = zero(ā[1])  # NOTE: Raises error if device has no qubits.
    for (pq, (p,q)) in enumerate(device.quples)
        G .+= device.g[pq] .* (ā[p]'* ā[q])
    end
    return G
end

function Devices.driveoperator(
    device::TransmonDevice{nQ,nS,nD,F},
    ā::AbstractVector{AbstractMatrix},
    i::Int,
    t::Real,
) where {nQ,nS,nD,F}
    a = ā[drivequbit(device, i)]
    e = exp(im * device.ν[i] * t)
    Ω = device.Ω[i](t)

    V   = (real(Ω) * e ) .* a
    V .+= (real(Ω) * e') .* a'

    if Ω isa Complex
        V .+= (imag(Ω) * im*e ) .* a
        V .-= (imag(Ω) * im*e') .* a'
    end

    return V
end

function Devices.gradeoperator(
    device::TransmonDevice{nQ,nS,nD,F},
    ā::AbstractVector{AbstractMatrix},
    j::Int,
    t::Real,
) where {nQ,nS,nD,F}
    a = ā[drivequbit(device, i)]
    e = exp(im * device.ν[i] * t)

    phase = Bool(j & 1) ? 1 : im    # Odd j -> "real" gradient operator; even j  -> "imag"
    A = (phase * e) .* a
    A .+= A'
    return A
end

function Devices.gradient(
    device::TransmonDevice,
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractMatrix,
)::AbstractVector
    grad = [zero(type) for type in Parameter.types(device)]

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    offset = 0
    for (i, Ω) in enumerate(device.Ω)
        j = 2*(i-1) + 1         # If Julia indexed from 0, this could just be 2i...
        L = Parameter.count(Ω)
        for k in 1:L
            ∂̄ = Signals.partial(k, Ω, t̄)
            grad[offset + k] += sum(τ̄ .* real.(∂̄) .* ϕ̄[j,:])
            grad[offset + k] += sum(τ̄ .* imag.(∂̄) .* ϕ̄[j+1,:])
        end
        offset += L
    end

    # CALCULATE GRADIENT FOR FREQUENCY PARAMETERS
    for (i, Ω) in enumerate(device.Ω)
        j = 2*(i-1) + 1         # If Julia indexed from 0, this could just be 2i...
        Ω̄ = Ω(t̄)
        grad[offset + i] += sum(τ̄ .* t̄ .* real.(Ω̄) .* ϕ̄[j+1,:])
        grad[offset + i] -= sum(τ̄ .* t̄ .* imag.(Ω̄) .* ϕ̄[j,:])

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
