import ..Signals
export Sine

import ..ParametricSignals: ParametricSignal, parameters

"""
    Sine(A::F, ν::F, ϕ::F) where {F<:AbstractFloat}

A real signal ``Ω(t) = A \\sin(νt+ϕ)``.

"""
mutable struct Sine{F} <: ParametricSignal{F,F}
    A::F    # MAXIMUM PEAK
    ν::F    # EFFECTIVE WIDTH
    ϕ::F    # CENTRAL TIME
end

function Signals.valueat(signal::Sine{F}, t::Real) where {F}
    A = signal.A; ν = signal.ν; ϕ = signal.ϕ
    return A * sin(ν*t + ϕ)
end

function Signals.partial(i::Int, signal::Sine, t::Real)
    field = parameters(signal)[i]
    return (field == :A ?   partial_A(signal, t)
        :   field == :ν ?   partial_ν(signal, t)
        :   field == :ϕ ?   partial_ϕ(signal, t)
        :                   error("Not Implemented"))
end

function partial_A(signal::Sine, t::Real)
    A = signal.A; ν = signal.ν; ϕ = signal.ϕ
    return sin(ν*t + ϕ)
end

function partial_ν(signal::Sine, t::Real)
    A = signal.A; ν = signal.ν; ϕ = signal.ϕ
    return A * cos(ν*t+ϕ) * t
end

function partial_ϕ(signal::Sine, t::Real)
    A = signal.A; ν = signal.ν; ϕ = signal.ϕ
    return A * cos(ν*t+ϕ)
end

function Base.string(::Sine, names::AbstractVector{String})
    A, ν, ϕ = names
    return "$A sin($ν⋅t + $ϕ)"
end
