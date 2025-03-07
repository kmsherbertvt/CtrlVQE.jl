import .Signals: SignalType

import ..CtrlVQE: Integrations, Signals

"""
    parametertype(signal)

Returns the number type for parameters in `signal`.

"""
parametertype(::SignalType{P,R}) where {P,R} = P

"""
    returntype(signal)

Returns the number type for function values of `signal`.

"""
returntype(::SignalType{P,R}) where {P,R} = R

"""
    (signal::SignalType{P,R})(t)

Syntactic sugar: if Ω is a `SignalType`, then `Ω(t)` gives `valueat(Ω,t)`.

The time `t` may be a scalar time, (abstract) vector of times, or an `IntegrationType`.

"""
(signal::SignalType)(t; kwargs...) = Signals.valueat(signal, t; kwargs...)

"""
    Base.string(Ω::SignalType)

A human-readable string description of the signal.

"""
function Base.string(signal::SignalType)
    return string(signal, Parameters.names(signal))
end

#= Vectorized versions of `valueat` and `partial`. =#

"""
    valueat(signal, grid; result=nothing)

Evaluate the signal at each point in the time lattice defined by an integration.

# Parameters
- signal: the `SignalType`.
- grid: an `IntegrationType`.

# Keyword Arguments
- result: if provided, a pre-allocated array to store the returned vector

"""
function Signals.valueat(
    signal::SignalType{P,R},
    grid::Integrations.IntegrationType;
    result=nothing,
) where {P,R}
    isnothing(result) && (result=Vector{R}(undef, length(grid)))
    for i in eachindex(grid)
        t = Integrations.timeat(grid,i)
        result[1+i] = Signals.valueat(signal, t)
    end
    return result
end

"""
    partial(i, signal, t̄; result=nothing)

Evaluate the partial at each point in the time lattice defined by an integration.

# Parameters
- i: indexes which parameter to take the partial derivative with respect to
- signal: the `SignalType`.
- grid: an `IntegrationType`.

# Keyword Arguments
- result: if provided, a pre-allocated array to store the returned vector

"""
function Signals.partial(
    k::Int,
    signal::SignalType{P,R},
    grid::Integrations.IntegrationType;
    result=nothing,
) where {P,R}
    isnothing(result) && (result=Vector{R}(undef, length(grid)))
    for i in eachindex(grid)
        t = Integrations.timeat(grid,i)
        result[1+i] = Signals.partial(k, signal, t)
    end
    return result
end