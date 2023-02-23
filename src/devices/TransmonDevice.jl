using LinearAlgebra: I

import ..Parameter, ..Signals, ..Devices
import ..Devices

struct Quple
    q1::Int
    q2::Int
    # INNER CONSTRUCTOR: Constrain order so that `Qu...ple(q1,q2) == Qu...ple(q2,q1)`.
    Quple(q1, q2) = q1 > q2 ? new(q2, q1) : new(q1, q2)
end

##########################################################################################
#=                  TRANSMON DEVICE (with rotating wave approximation)

TODO: TransmonDevice may be abstract; just assume subtypes have n, ω, δ, G.
        Maybe go ahead and assume m and channels also and just let violating subtypes override the relevant methods. Or, make those methods less dependent on m.
=#

struct TransmonDevice{F<:AbstractFloat} <: Devices.Device
    n::Int,
    m::Int,
    ω::Tuple{Vararg{F}}
    δ::Tuple{Vararg{F}}
    G::Dict{Quple,F}
    channels::Tuple{Vararg{Pulses.AbstractChannel}}

    function TransmonDevice(
        n::Int,
        m::Int,
        ω::Tuple{Vararg{F}},
        δ::Tuple{Vararg{F}},
        G::Dict{Quple,F},
        channels::Tuple{Vararg{Pulses.AbstractChannel}},
    ) where {F}
        length(ω) ≠ n && error("length(ω) does not match n")
        length(δ) ≠ n && error("length(δ) does not match n")
        for (quple, g) in G
            !(1 <= quple.q1 <= quple.q2 <= n) && error("qubits in g do not match n")
        end
        for channel in channels
            !(1 <= channel.q <= n) && error("qubits in channels do not match n")
        end
        return new{F}(n,m,ω,δ,G,channels)
    end
end

#= IMPLEMENT PARAMETER INTERFACE =#

function Parameter.count(device::TransmonDevice)
    return sum(Parameter.count(channel) for channel in device.channels)
end

function Parameter.names(device::TransmonDevice)
    names(i) = (
        channel = device.channels[i];
        ["c$i(q$(channel.q)):$name" for name in Parameter.names(channel)]
    )
    return vcat((names(i) for i in eachindex(device.channels))...)
end

function Parameter.types(device::TransmonDevice)
    return vcat((Parameter.types(channel) for channel in device.channels)...)
end

function Parameter.bind(device::TransmonDevice, x̄::AbstractVector)
    offset = 0
    for channel in device.channels
        L = Parameter.count(channel)
        Parameter.bind(channel, x̄[offset+1:offset+L])
        offset += l
    end
end

#= MANDATORY DEVICE METHODS =#

Devices.nqubits(device::TransmonDevice) = device.n
Devices.nstates(device::TransmonDevice, q::Int) = device.m

function Devices.localidentityoperator(
    device::TransmonDevice{F},
    q::Int,
) where {F}
    return Matrix{F}(I, device.m, device.m)
end

function Devices.localloweringoperator(
    device::TransmonDevice{F},
    q::Int,
) where {F}
    a = zeros(F, device.m, device.m)
    for i ∈ 1:device.m-1
        a[i,i+1] = √i
    end
    return a
end

function Devices.localstatichamiltonian(
    device::TransmonDevice,
    ā::AbstractVector{AbstractMatrix},
    q::Int,
)
    return Hermitian(sum((
         device.ω[q]   *        ā[q]'* ā[q],
        -device.δ[q]/2 * ā[q]'* ā[q]'* ā[q] * ā[q],
    )))
end

function Devices.localdrivenhamiltonian(
    device::TransmonDevice{F},
    ā::AbstractVector{AbstractMatrix},
    q::Int,
    t::Real,
)
    V = zeros(Complex{F}, size(ā[q]))
    for channel in device.channels
        if channel.q != q; continue; end
        z = channel.Ω(t) * exp(- im * channel.ν * t)
        V .+= (z .* ā[q]) .+ (z' .* ā[q]')
    end
    return Hermitian(V)
end

function Devices.mixedstatichamiltonian(
    device::TransmonDevice{F},
    ā::AbstractVector{AbstractMatrix},
) where {F}
    G = zero(ā[1])
    for (quple, g) in device.G
        p, q = quple.q1, quple.q2
        G .+= g .* (ā[p]' * ā[q] .+ ā[q]' * ā[p])
    end
    return Hermitian(G)
end

function Devices.mixeddrivenhamiltonian(
    device::TransmonDevice{F},
    ā::AbstractVector{AbstractMatrix},
    t::Real,
) where {F}
    return zero(ā[1])
end

#= ADDITIONAL OVERRRIDES =#

# No need to chain so many function calls together when all m are constant.

Devices.nstates(device::TransmonDevice) = device.n ^ device.m

# Because J(t)=0, we can override Driven methods to use purely Local functionality.

function Devices.hamiltonian(::Type{Temporality.Driven},
    device::TransmonDevice,
    t::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return Devices.hamiltonian(Locality.Local, Temporality.Driven, device, t, basis)
end

function Devices.propagator(::Type{Temporality.Driven},
    device::TransmonDevice,
    t::Real,
    τ::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return Devices.propagator(Locality.Local, Temporality.Driven, device, t, τ, basis)
end

function Devices.propagate!(::Type{Temporality.Driven},
    device::TransmonDevice,
    t::Real,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return Devices.propagate!(Locality.Local, Temporality.Driven, device, t, τ, ψ, basis)
end


