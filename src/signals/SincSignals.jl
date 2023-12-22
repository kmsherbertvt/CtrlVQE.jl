import ..Signals
export Sinc

import ..ParametricSignals: ParametricSignal, parameters

"""
    Sinc(A::F, σ::F, s::F) where {F<:AbstractFloat}

A real signal ``Ω(t) = A \\sin(χ)/χ`` where ``χ≡(t-s)/σ``.
Note that Julia's built-in `sinc` function implicitly sets A=1, s=0, and σ=1/π.

"""
mutable struct Sinc{F} <: ParametricSignal{F,F}
    A::F    # MAXIMUM PEAK
    σ::F    # EFFECTIVE WIDTH
    s::F    # CENTRAL TIME
end

function Signals.valueat(signal::Sinc{F}, t::Real) where {F}
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return A * sinc(χ/π)
end

function Signals.partial(i::Int, signal::Sinc, t::Real)
    field = parameters(signal)[i]
    return (field == :A ?   partial_A(signal, t)
        :   field == :σ ?   partial_σ(signal, t)
        :   field == :s ?   partial_s(signal, t)
        :                   error("Not Implemented"))
end

function partial_A(signal::Sinc, t::Real)
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return sinc(χ/π)
end

function partial_σ(signal::Sinc, t::Real)
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return -A*χ/(π*σ) * cosc(χ/π)
end

function partial_s(signal::Sinc, t::Real)
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return -A/(π*σ) * cosc(χ/π)
end

function Base.string(::Sinc, names::AbstractVector{String})
    A, σ, s = names
    return "$A sinc((t-$s)/σ)"
end
