import ..Signals
export Tanh

import ..ParametricSignals: ParametricSignal, parameters

"""
    Tanh(A::F, σ::F, s::F) where {F<:AbstractFloat}

A real signal ``Ω(t) = A \\sin(χ)/χ`` where ``χ≡(t-s)/σ``.
Note that Julia's built-in `sinc` function implicitly sets A=1, s=0, and σ=1/π.

"""
mutable struct Tanh{F} <: ParametricSignal{F,F}
    A::F    # MAXIMUM PEAK
    σ::F    # EFFECTIVE WIDTH
    s::F    # CENTRAL TIME
end

function Signals.valueat(signal::Tanh{F}, t::Real) where {F}
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return A * tanh(χ)
end

function Signals.partial(i::Int, signal::Tanh, t::Real)
    field = parameters(signal)[i]
    return (field == :A ?   partial_A(signal, t)
        :   field == :σ ?   partial_σ(signal, t)
        :   field == :s ?   partial_s(signal, t)
        :                   error("Not Implemented"))
end

function partial_A(signal::Tanh, t::Real)
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return tanh(χ)
end

function partial_σ(signal::Tanh, t::Real)
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return -A*χ/σ * sech(χ)^2
end

function partial_s(signal::Tanh, t::Real)
    A = signal.A; σ = signal.σ; s = signal.s
    χ = (t - s) / σ
    return -A/σ * sech(χ)^2
end

function Base.string(::Tanh, names::AbstractVector{String})
    A, σ, s = names
    return "$A tanh((t-$s)/σ)"
end
