import ..Signals
export Gaussian

import ..ParametricSignals: ParametricSignal, parameters

"""
    Gaussian(A::F, σ::F, s::F) where {F<:AbstractFloat}

A Gaussian real signal ``Ω(t) = A \\exp( -((t-s)/σ)^2 / 2)``.

"""
mutable struct Gaussian{F} <: ParametricSignal{F,F}
    A::F    # MAXIMUM PEAK
    σ::F    # EFFECTIVE WIDTH
    s::F    # CENTRAL TIME
end

function Signals.valueat(signal::Gaussian{F}, t::Real) where {F}
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return A * exp( -χ^2 / 2 )
end

function Signals.partial(i::Int, signal::Gaussian, t::Real)
    field = parameters(signal)[i]
    return (field == :A ?   partial_A(signal, t)
        :   field == :σ ?   partial_σ(signal, t)
        :   field == :s ?   partial_s(signal, t)
        :                   error("Not Implemented"))
end

function partial_A(signal::Gaussian, t::Real)
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return exp( -χ^2 / 2 )
end

function partial_σ(signal::Gaussian, t::Real)
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return A * exp( -χ^2 / 2) * χ^2 / σ
end

function partial_s(signal::Gaussian, t::Real)
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return A * exp( -χ^2 / 2) * χ / σ
end

function Base.string(::Gaussian, names::AbstractVector{String})
    A, σ, s = names
    return "$A exp(-(t-$s)²/2$(σ)²)"
end
