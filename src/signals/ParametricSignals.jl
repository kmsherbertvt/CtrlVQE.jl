import Memoization: @memoize
import ...Parameters, ...Signals


"""
    abstract type ParametricSignal{P,R} ... end

# Implementation
- Subtypes S are mutable structs.
- All differentiable parameters of S are of type P.
- Subtypes implement the following methods:
    (::S)(t::Real)::R
    partial(i::Int, ::S, t::Real)::R

By default, all fields of type P are treated as differentiable parameters.
You can narrow this selection by implementing:
    parameters(::S)::Vector{Symbol}
But before you do that, consider using a `ConstrainedSignal` instead.

"""
abstract type ParametricSignal{P,R} <: Signals.AbstractSignal{P,R} end

@memoize Dict function parameters(::Type{S}) where {P,R,S<:ParametricSignal{P,R}}
    return [field for (i, field) in enumerate(fieldnames(S)) if S.types[i] == P]
end

function parameters(signal::S) where {P,R,S<:ParametricSignal{P,R}}
    return parameters(S)
end

Parameters.count(signal::S) where {S<:ParametricSignal} = length(parameters(S))

function Parameters.names(signal::S) where {S<:ParametricSignal}
    return [string(field) for field in parameters(S)]
end

function Parameters.values(signal::S) where {P,R,S<:ParametricSignal{P,R}}
    return P[getfield(signal, field) for field in parameters(S)]
end

function Parameters.bind(
    signal::S,
    x̄::AbstractVector{P}
) where {P,R,S<:ParametricSignal{P,R}}
    for (i, field) in enumerate(parameters(signal))
        setfield!(signal, field, x̄[i])
    end
end

#= TO BE IMPLEMENTED BY SUB-CLASSES:
    (::S)(t::Real)::R
    Signals.partial(i::Int, ::S, t::Real)::R
    Base.string(::S, ::AbstractVector{String})::String
=#








##########################################################################################
#=                              SPECIAL SIGNAL TYPES
=#


#= CONSTRAINED SIGNAL =#

struct ConstrainedSignal{P,R,S<:ParametricSignal{P,R}} <: Signals.AbstractSignal{P,R}
    constrained::S
    constraints::Vector{Symbol}
    _map::Vector{Int}

    function ConstrainedSignal(
        constrained::S,
        constraints::Vector{Symbol}
    ) where {P,R,S<:ParametricSignal{P,R}}
        fields = parameters(constrained)
        _map = [j for (j, field) in enumerate(fields) if field ∉ constraints]
        return new{P,R,S}(constrained, constraints, _map)
    end
end

function ConstrainedSignal(constrained::ParametricSignal, constraints::Symbol...)
    return ConstrainedSignal(constrained, collect(constraints))
end

function Parameters.count(signal::ConstrainedSignal)
    return Parameters.count(signal.constrained) - length(signal.constraints)
end

function Parameters.names(signal::ConstrainedSignal)
    return Parameters.names(signal.constrained)[collect(signal._map)]
end

function Parameters.values(signal::ConstrainedSignal)
    return Parameters.values(signal.constrained)[collect(signal._map)]
end

function Parameters.bind(signal::ConstrainedSignal{P,R,S}, x̄::AbstractVector{P}) where {P,R,S}
    fields = parameters(signal.constrained)
    for i in eachindex(x̄)
        setfield!(signal.constrained, fields[signal._map[i]], x̄[i])
    end
end

(signal::ConstrainedSignal)(t::Real) = signal.constrained(t)
function Signals.partial(i::Int, signal::ConstrainedSignal, t::Real)
    return Signals.partial(signal._map[i], signal.constrained, t)
end

function Base.string(signal::ConstrainedSignal, names::AbstractVector{String})
    newnames = string.(Parameters.values(signal.constrained))
    for i in eachindex(names)
        newnames[signal._map[i]] = names[i]
    end
    return Base.string(signal.constrained, newnames)
end