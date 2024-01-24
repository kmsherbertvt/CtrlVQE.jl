import ..Parameters, ..Signals
export AnalyticSignal

import ..Signals: SignalType

"""
    AnalyticSignal(signal::SignalType{P,R}, phase::Complex{R})

Convert a real-valued signal into a complex one.

The name is a play on analytic continuation,
    which is a method to convert a function with a real domain
    into one with complex-valued parameters.
But, strictly speaking, it is the range that is being extended, not the domain.

Note that the scale (typically either just 1 or i) is fixed.
If there is a need to vary it,
    the signal type should probably have its own complex implementation.

"""
struct AnalyticSignal{
    P,R,C<:Complex{R},
    S<:SignalType{P,R},
} <: SignalType{P,C}
    continued::S
    scale::C
end

function AnalyticSignal(continued::SignalType{P,R}, scale) where {P,R}
    AnalyticSignal(continued, Complex{R}(scale))
end

#= `Parameters` INTERFACE =#

Parameters.count(signal::AnalyticSignal) = Parameters.count(signal.continued)
Parameters.names(signal::AnalyticSignal) = Parameters.names(signal.continued)
Parameters.values(signal::AnalyticSignal) = Parameters.values(signal.continued)

function Parameters.bind(signal::AnalyticSignal{P,C}, x̄::AbstractVector{P}) where {P,C}
    Parameters.bind(signal.continued, x̄)
end

#= `Signals` INTERFACE =#

function Signals.valueat(signal::AnalyticSignal, t::Real)
    return Signals.valueat(signal.continued, t) * signal.scale
end

function Signals.partial(i::Int, signal::AnalyticSignal, t::Real)
    return Signals.partial(i, signal.continued, t) * signal.scale
end

function Base.string(signal::AnalyticSignal, names::AbstractVector{String})
    text = string(signal.continued, names)::String
    return "($(signal.scale)) ($text)"
end

