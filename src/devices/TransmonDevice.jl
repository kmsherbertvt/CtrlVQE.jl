using Memoization: @memoize
using ..LinearAlgebraTools: List
import ..Parameters, ..LinearAlgebraTools, ..Signals, ..Devices


struct TransmonDevice{
    F<:AbstractFloat,
    S<:Signals.AbstractSignal,
} <: Devices.LocallyDrivenDevice
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
        @assert length(ω̄) == length(δ̄)                  # NUMBER OF QUBITS
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



#= Other types:

FixedFrequencyTransmonDevice: ν is tuple and not included in parameters.
LinearTransmonDevice: quples and ḡ replaced by n-length tuple ḡ, efficient static propagate.
TransmonDeviceSansRWA: implicitly one channel per qubit, different drive
a mix of the three I guess...

=#




#= Constructors:

Obv. accept AbstractVector in lieu of SVector for all parameters.
Accept Dict{Quple,F} or Matrix{F} for G.
If q is omitted, defaults to [1..n]. Only works if Ω/ν lengths are n.
If ν is omitted, initializes to ω̄[q]
If m is omitted, defaults to 2.

=#







#= IMPLEMENT BOOK-KEEPING =#

Devices.nqubits(device::TransmonDevice) = length(device.ω̄)
Devices.nstates(device::TransmonDevice, q::Int) = device.m
Devices.ndrives(device::TransmonDevice) = length(device.q̄)
Devices.ngrades(device::TransmonDevice) = 2*Devices.ndrives(device)

Devices.drivequbit(device::TransmonDevice, i::Int) = device.q̄[i]
Devices.gradequbit(device::TransmonDevice, j::Int) = device.q̄[((j-1) >> 1) + 1]





#= IMPLEMENT OPERATORS =#

@memoize Dict function localloweringoperator(
    device::TransmonDevice{F,S},
    q::Int,
) where {F,S}
    m = Devices.nstates(device,q)
    a = zeros(F, m, m)
    for i ∈ 1:m-1
        a[i,i+1] = √i
    end
    return a
end

function Devices.localloweringoperator(
    device::TransmonDevice,
    q::Int,
)
    return localloweringoperator(device,q)
end

function Devices.qubithamiltonian(
    device::TransmonDevice,
    ā::List{<:AbstractMatrix},
    q::Int,
)
    a = ā[q]
    I = one(a)
    h = zero(a)
    h .+= (-device.δ̄[q] / 2) .* I               #       - δ/2    I
    h = LinearAlgebraTools.rotate!(a', h)       #       - δ/2   a'a
    h .+= ( device.ω̄[q]    ) .* I               # ω     - δ/2   a'a
    h = LinearAlgebraTools.rotate!(a', h)       # ω a'a - δ/2 a'a'aa
    return h
end

function Devices.staticcoupling(
    device::TransmonDevice,
    ā::List{<:AbstractMatrix},
)
    G = zero(ā[1])  # NOTE: Raises error if device has no qubits.
    for (pq, (p,q)) in enumerate(device.quples)
        G .+= device.ḡ[pq] .* (ā[p]'* ā[q])
        G .+= device.ḡ[pq] .* (ā[q]'* ā[p])
    end
    return G
end

function Devices.driveoperator(
    device::TransmonDevice,
    ā::List{<:AbstractMatrix},
    i::Int,
    t::Real,
)
    a = ā[Devices.drivequbit(device, i)]
    e = exp(im * device.ν̄[i] * t)
    Ω = device.Ω̄[i](t)

    V   = (real(Ω) * e ) .* a
    V .+= (real(Ω) * e') .* a'

    if Ω isa Complex
        V .+= (imag(Ω) * im *e ) .* a
        V .+= (imag(Ω) * im'*e') .* a'
    end

    return V
end

function Devices.gradeoperator(
    device::TransmonDevice,
    ā::List{<:AbstractMatrix},
    j::Int,
    t::Real,
)
    i = ((j-1) >> 1) + 1
    a = ā[Devices.drivequbit(device, i)]
    e = exp(im * device.ν̄[i] * t)

    phase = Bool(j & 1) ? 1 : im    # Odd j -> "real" gradient operator; even j  -> "imag"
    A   = (phase * e ) .* a
    A .+= (phase'* e') .* a'
    return A
end

function Devices.gradient(
    device::TransmonDevice,
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractMatrix,
)::AbstractVector
    grad = [zero(value) for value in Parameters.values(device)]

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    offset = 0
    for (i, Ω) in enumerate(device.Ω̄)
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
    for (i, Ω) in enumerate(device.Ω̄)
        j = 2*(i-1) + 1             # If Julia indexed from 0, this could just be 2i...
        Ω̄ = Ω(t̄)
        grad[offset + i] += sum(τ̄ .* t̄ .* real.(Ω̄) .* ϕ̄[:,j+1])
        grad[offset + i] -= sum(τ̄ .* t̄ .* imag.(Ω̄) .* ϕ̄[:,j])

    end

    return grad
end





#= IMPLEMENT PARAMETER INTERFACE =#

function Parameters.count(device::TransmonDevice)
    return sum(Parameters.count(Ω) for Ω in device.Ω̄) + length(device.ν̄)
end

function Parameters.names(device::TransmonDevice)
    names = []

    # STRING TOGETHER PARAMETER NAMES FOR EACH SIGNAL Ω̄[i]
    annotate(name,i) = "Ω$i(q$(device.q̄[i])):$name"
    for i in eachindex(device.Ω̄)
        append!(names, (annotate(name,i) for name in Parameters.names(device.Ω̄[i])))
    end

    # TACK ON PARAMETER NAMES FOR EACH ν̄[i]
    append!(names, ("ν$i" for i in eachindex(device.ν̄)))
    return names
end

function Parameters.values(device::TransmonDevice)
    values = []

    # STRING TOGETHER PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for i in eachindex(device.Ω̄)
        append!(values, (value for value in Parameters.values(device.Ω̄[i])))
    end

    # TACK ON PARAMETERS FOR EACH ν̄[i]
    append!(values, device.ν̄)
    return values
end

function Parameters.bind(device::TransmonDevice, x̄::AbstractVector)
    offset = 0

    # BIND PARAMETERS FOR EACH SIGNAL Ω̄[i]
    for Ω in device.Ω̄
        L = Parameters.count(Ω)
        Parameters.bind(Ω, x̄[offset+1:offset+L])
        offset += L
    end

    # BIND PARAMETERS FOR EACH ν̄[i]
    for i in eachindex(device.ν̄)
        device.ν̄[i] = x̄[offset+i]
    end
    offset += length(device.ν̄)
end
