#= NOTE: Last updated 2/23/23. Code practically copied into TransmonDevice (WITH RWA),
    so changes made to that module will likely be wanted here also.

Ultimately it would be good to hack in some `_super` functionality,
    so changes are automatic. But, spare no time on that now.
=#

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
#=                  TRANSMON DEVICE (no rotating wave approximation)
=#

struct TransmonDeviceSansRWA{F<:AbstractFloat} <: Devices.Device
    n::Int,
    m::Int,
    ω::Tuple{Vararg{F}}
    δ::Tuple{Vararg{F}}
    G::Dict{Quple,F}
    f::Tuple{Vararg{Pulses.AbstractSignal}}

    function TransmonDeviceSansRWA(
        n::Int,
        m::Int,
        ω::Tuple{Vararg{F}},
        δ::Tuple{Vararg{F}},
        g::Dict{Quple,F},
        f::Tuple{Vararg{Pulses.AbstractSignal}},
    ) where {F}
        length(ω) ≠ n && error("length(ω) does not match n")
        length(δ) ≠ n && error("length(δ) does not match n")
        length(f) ≠ n && error("length(f) does not match n")
        for (quple, g) in G
            !(1 <= quple.q1 <= quple.q2 <= n) && error("qubits in g do not match n")
        end
        return new{F}(n,m,ω,δ,G,f)
    end
end

#= IMPLEMENT PARAMETER INTERFACE =#

function Parameter.count(device::TransmonDeviceSansRWA)
    sum(Parameter.count(signal) for signal in device.f)
end

function Parameter.names(device::TransmonDeviceSansRWA)
    names(q) = ["q$q:$name" for name in Parameter.names(device.f[q])]
    return vcat((names(q) for q in eachindex(device.f))...)
end

function Parameter.types(device::TransmonDeviceSansRWA)
    return vcat((Parameter.types(signal) for signal in device.f)...)
end

function Parameter.bind(device::TransmonDeviceSansRWA, x̄::AbstractVector)
    offset = 0
    for signal in device.f
        L = Parameter.count(signal)
        Parameter.bind(signal, x̄[offset+1:offset+L])
        offset += L
    end
end

#= MANDATORY DEVICE METHODS =#

Devices.nqubits(device::TransmonDeviceSansRWA) = device.n
Devices.nstates(device::TransmonDeviceSansRWA, q::Int) = device.m

function Devices.localidentityoperator(
    device::TransmonDeviceSansRWA{F},
    q::Int,
) where {F}
    return Matrix{F}(I, device.m, device.m)
end

function Devices.localloweringoperator(
    device::TransmonDeviceSansRWA{F},
    q::Int,
) where {F}
    a = zeros(F, device.m, device.m)
    for i ∈ 1:device.m-1
        a[i,i+1] = √i
    end
    return a
end

function Devices.localstatichamiltonian(
    device::TransmonDeviceSansRWA,
    ā::AbstractVector{AbstractMatrix},
    q::Int,
)
    return Hermitian(sum((
         device.ω[q]   *        ā[q]'* ā[q],
        -device.δ[q]/2 * ā[q]'* ā[q]'* ā[q] * ā[q],
    )))
end

function Devices.localdrivenhamiltonian(
    device::TransmonDeviceSansRWA,
    ā::AbstractVector{AbstractMatrix},
    q::Int,
    t::Real,
)
    return Hermitian(device.f[q](t) * (ā[q] + ā[q]'))
end

function Devices.mixedstatichamiltonian(
    device::TransmonDeviceSansRWA{F},
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
    device::TransmonDeviceSansRWA{F},
    ā::AbstractVector{AbstractMatrix},
    t::Real,
) where {F}
    return zero(ā[1])
end

#= ADDITIONAL OVERRRIDES =#

# No need to chain so many function calls together when all m are constant.

Devices.nstates(device::TransmonDeviceSansRWA) = device.n ^ device.m

# Because J(t)=0, we can override Driven methods to use purely Local functionality.

function Devices.hamiltonian(::Type{Temporality.Driven},
    device::TransmonDeviceSansRWA,
    t::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return Devices.hamiltonian(Locality.Local, Temporality.Driven, device, t, basis)
end

function Devices.propagator(::Type{Temporality.Driven},
    device::TransmonDeviceSansRWA,
    t::Real,
    τ::Real,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return Devices.propagator(Locality.Local, Temporality.Driven, device, t, τ, basis)
end

function Devices.propagate!(::Type{Temporality.Driven},
    device::TransmonDeviceSansRWA,
    t::Real,
    τ::Real,
    ψ::AbstractVector,
    basis::Type{<:AbstractBasis}=Basis.Occupation,
)
    return Devices.propagate!(Locality.Local, Temporality.Driven, device, t, τ, ψ, basis)
end


