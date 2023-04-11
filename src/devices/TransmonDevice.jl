using Memoization: @memoize
using ...LinearAlgebraTools: List
import ...Parameters, ...LinearAlgebraTools, ...Signals, ...Devices

@memoize Dict function bosonic_annihilator(::Type{F}, m::Int) where {F<:AbstractFloat}
    a = zeros(F, m, m)
    for i ∈ 1:m-1
        a[i,i+1] = √i
    end
    return a
end



abstract type AbstractTransmonDevice{F} <: Devices.LocallyDrivenDevice end

# THE INTERFACE TO IMPLEMENT

# Devices.nstates
# Devices.nqubits
resonancefrequency(::AbstractTransmonDevice, q::Int)::Real = error("Not Implemented")
anharmonicity(::AbstractTransmonDevice, q::Int)::Real = error("Not Implemented")

ncouplings(::AbstractTransmonDevice)::Int = error("Not Implemented")
couplingpair(::AbstractTransmonDevice, k::Int)::Devices.Quple = error("Not Implemented")
couplingstrength(::AbstractTransmonDevice, k::Int)::Real = error("Not Implemented")

# Devices.ndrives
# Devices.drivequbit
drivefrequency(::AbstractTransmonDevice, i::Int)::Real = error("Not Implemented")
drivesignal(::AbstractTransmonDevice, i::Int)::AbstractSignal = error("Not Implemented")

bindfrequencies(::AbstractTransmonDevice, ν̄::AbstractVector) = error("Not Implemented")


# THE INTERFACE ALREADY IMPLEMENTED

function Devices.ngrades(device::AbstractTransmonDevice)
    return 2 * Devices.ndrives(device)
end

function Devices.gradequbit(device::AbstractTransmonDevice, j::Int)
    return Devices.drivequbit(device, ((j-1) >> 1) + 1)
end

function Devices.localloweringoperator(
    device::AbstractTransmonDevice{F},
    q::Int
) where {F}
    return bosonic_annihilator(F, Devices.nstates(device, q))
end

function Devices.qubithamiltonian(
    device::AbstractTransmonDevice,
    ā::List{<:AbstractMatrix},
    q::Int,
)
    a = ā[q]
    I = one(a)
    h = zero(a)
    h .-= (anharmonicity(device,q)/2)  .* I     #       - δ/2    I
    h = LinearAlgebraTools.rotate!(a', h)       #       - δ/2   a'a
    h .+= resonancefrequency(device,q) .* I     # ω     - δ/2   a'a
    h = LinearAlgebraTools.rotate!(a', h)       # ω a'a - δ/2 a'a'aa
    return h
end

function Devices.staticcoupling(
    device::AbstractTransmonDevice,
    ā::List{<:AbstractMatrix},
)
    G = zero(ā[1])  # NOTE: Raises error if device has no qubits.
    for pq in 1:ncouplings(device)
        g = couplingstrength(device, pq)
        p, q = couplingpair(device, pq)
        G .+= g .* (ā[p]'* ā[q])
        G .+= g .* (ā[q]'* ā[p])
    end
    return G
end

function Devices.driveoperator(
    device::AbstractTransmonDevice,
    ā::List{<:AbstractMatrix},
    i::Int,
    t::Real,
)
    a = ā[Devices.drivequbit(device, i)]
    e = exp(im * drivefrequency(device, i) * t)
    Ω = drivesignal(device, i)(t)

    V   = (real(Ω) * e ) .* a
    V .+= (real(Ω) * e') .* a'

    if Ω isa Complex
        V .+= (imag(Ω) * im *e ) .* a
        V .+= (imag(Ω) * im'*e') .* a'
    end

    return V
end

function Devices.gradeoperator(
    device::AbstractTransmonDevice,
    ā::List{<:AbstractMatrix},
    j::Int,
    t::Real,
)
    i = ((j-1) >> 1) + 1
    a = ā[Devices.drivequbit(device, i)]
    e = exp(im * drivefrequency(device, i) * t)

    phase = Bool(j & 1) ? 1 : im    # Odd j -> "real" gradient operator; even j  -> "imag"
    A   = (phase * e ) .* a
    A .+= (phase'* e') .* a'
    return A
end

function Devices.gradient(
    device::AbstractTransmonDevice,
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractMatrix,
)::AbstractVector
    grad = zero.(Parameters.values(device))

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    offset = 0
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        j = 2*(i-1) + 1             # If Julia indexed from 0, this could just be 2i...
        L = Parameters.count(Ω)
        for k in 1:L
            ∂̄ = Signals.partial(k, Ω, t̄)
            grad[offset + k] += sum(τ̄ .* real.(∂̄) .* ϕ̄[:,j])
            grad[offset + k] += sum(τ̄ .* imag.(∂̄) .* ϕ̄[:,j+1])
        end
        offset += L
    end

    # CALCULATE GRADIENT FOR FREQUENCY PARAMETERS
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        j = 2*(i-1) + 1             # If Julia indexed from 0, this could just be 2i...
        Ω̄ = Ω(t̄)
        grad[offset + i] += sum(τ̄ .* t̄ .* real.(Ω̄) .* ϕ̄[:,j+1])
        grad[offset + i] -= sum(τ̄ .* t̄ .* imag.(Ω̄) .* ϕ̄[:,j])

    end

    return grad
end

function Parameters.count(device::AbstractTransmonDevice)
    cnt = Devices.ndrives(device)           # NOTE: There are `ndrives` frequencies.
    for i in 1:Devices.ndrives(device)
        cnt += Parameters.count(drivesignal(device, i))
    end
    return cnt
end

function Parameters.names(device::AbstractTransmonDevice)
    names = []

    # STRING TOGETHER PARAMETER NAMES FOR EACH SIGNAL Ω̄[i]
    annotate(name,i) = "Ω$i(q$(device.q̄[i])):$name"
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        append!(names, (annotate(name,i) for name in Parameters.names(Ω)))
    end

    # TACK ON PARAMETER NAMES FOR EACH ν̄[i]
    append!(names, ("ν$i" for i in 1:Devices.ndrives(device)))
    return names
end

function Parameters.values(device::AbstractTransmonDevice)
    values = []

    # STRING TOGETHER PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        append!(values, Parameters.values(Ω))
    end

    # TACK ON PARAMETERS FOR EACH ν̄[i]
    append!(values, (drivefrequency(device, i) for i in 1:Devices.ndrives(device)))
    return values
end

function Parameters.bind(device::AbstractTransmonDevice, x̄::AbstractVector)
    offset = 0

    # BIND PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in 1:Devices.ndrives(device)
        Ω = drivesignal(device, i)
        L = Parameters.count(Ω)
        Parameters.bind(Ω, x̄[offset+1:offset+L])
        offset += L
    end

    # BIND PARAMETERS FOR EACH ν̄[i]
    bindfrequencies(device, x̄[offset+1:end])
end







struct TransmonDevice{
    F<:AbstractFloat,
    S<:Signals.AbstractSignal,
} <: AbstractTransmonDevice{F}
    # QUBIT LISTS
    ω̄::Vector{F}
    δ̄::Vector{F}
    # COUPLING LISTS
    ḡ::Vector{F}
    quples::Vector{Devices.Quple}
    # DRIVE LISTS
    q̄::Vector{Int}
    ν̄::Vector{F}
    Ω̄::Vector{S}
    # OTHER PARAMETERS
    m::Int

    function TransmonDevice(
        ω̄::AbstractVector{<:Real},
        δ̄::AbstractVector{<:Real},
        ḡ::AbstractVector{<:Real},
        quples::AbstractVector{Devices.Quple},
        q̄::AbstractVector{Int},
        ν̄::AbstractVector{<:AbstractFloat},
        Ω̄::AbstractVector{S},
        m::Int,
    ) where {S<:Signals.AbstractSignal}
        # VALIDATE PARALLEL LISTS ARE CONSISTENT SIZE
        @assert length(ω̄) == length(δ̄) ≥ 1              # NUMBER OF QUBITS
        @assert length(ḡ) == length(quples)             # NUMBER OF COUPLINGS
        @assert length(q̄) == length(ν̄) == length(Ω̄)     # NUMBER OF DRIVES

        # VALIDATE QUBIT INDICES
        for (p,q) in quples
            @assert 1 <= p <= length(ω̄)
            @assert 1 <= q <= length(ω̄)
        end
        for q in q̄
            @assert 1 <= q <= length(ω̄)
        end

        # VALIDATE THAT THE HILBERT SPACE HAS SOME VOLUME...
        @assert m ≥ 2

        # STANDARDIZE TYPING AND CONVERT ALL LISTS TO IMMUTABLE TUPLE (except ν)
        F = promote_type(eltype(ω̄), eltype(δ̄), eltype(ḡ), eltype(ν̄))
        return new{F,S}(
            convert(Vector{F}, ω̄),
            convert(Vector{F}, δ̄),
            convert(Vector{F}, ḡ),
            quples,
            q̄,
            convert(Vector{F}, ν̄),
            Ω̄,
            m,
        )
    end
end

Devices.nstates(device::TransmonDevice, q::Int) = device.m

Devices.nqubits(device::TransmonDevice) = length(device.ω̄)
resonancefrequency(device::TransmonDevice, q::Int) = device.ω̄[q]
anharmonicity(device::TransmonDevice, q::Int) = device.δ̄[q]

ncouplings(device::TransmonDevice) = length(device.quples)
couplingpair(device::TransmonDevice, k::Int) = device.quples[k]
couplingstrength(device::TransmonDevice, k::Int) = device.ḡ[k]

Devices.ndrives(device::TransmonDevice) = length(device.q̄)
Devices.drivequbit(device::TransmonDevice, i::Int) = device.q̄[i]
drivefrequency(device::TransmonDevice, i::Int) = device.ν̄[i]
drivesignal(device::TransmonDevice, i::Int) = device.Ω̄[i]

bindfrequencies(device::TransmonDevice, ν̄::AbstractVector) = (device.ν̄ .= ν̄)
#= TODO (hi): a reliable way to bind new signals.

The problem is the typing.
We don't want any struct to have to have abstract fields,
    but we don't want to have to specify the type of the signal.

 =#


#= TODO (low): Other types

FixedFrequencyTransmonDevice: ν is tuple and not included in parameters.
LinearTransmonDevice: quples and ḡ replaced by n-length tuple ḡ, efficient static propagate.
TransmonDeviceSansRWA: implicitly one channel per qubit, different drive
a mix of the three I guess...

=#