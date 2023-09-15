import ..Parameters, ..Signals
export ParametricSignal, ConstrainedSignal
export parameters

using Memoization: @memoize


"""
    ParametricSignal{P,R} ... end

Super-type for user-defined signal objects ``Ω(t)``.

# Type Parameters
- `P` denotes the type of all variational parameters. Must be a real float.
- `R` denotes the type of ``Ω(t)`` itself. May be any number type.

# Implementation

Any concrete sub-type `S` must be a mutable struct,
    and all of its variational parameters are of type P.
By default, *all* fields of type P are treated as variational parameters.
You may optionally change this by implementing `parameters(::S)`,
    which should return a vector of each variational parameter of `S`.
But before you do that,
    consider whether a `ConstrainedSignal` provides the desired behavior.

The following methods must be implemented:

- `Signals.valueat(Ω::S, t::Real)`:
        the actual function ``Ω(t)``. Must return a number of type `R`.

- `Signals.partial(i::Int, Ω::S, t::Real)`:
        the partial derivative ``∂Ω/∂x_i`` evaluated at time `t`,
        where ``x_i`` is Ω's i-th variational parameter (ie. `Parameters.names(Ω)[i]`).
        Must return a number of type `R`.

- `Base.string(Ω::S, names::AbstractVector{String})`:
        a human-readable description of the signal,
        inserting each element of `names` in the place of the corresponding parameter.
    For example, a complex constant signal might return a description like "\$A + i \$B",
        where `A` and `B` are the "names" given by the `names` argument.

"""
abstract type ParametricSignal{P,R} <: Signals.SignalType{P,R} end

#= TO BE IMPLEMENTED BY SUB-CLASSES:
    Signals.valueat(::S, t::Real)::R
    Signals.partial(i::Int, ::S, t::Real)::R
    Base.string(::S, ::AbstractVector{String})::String
=#

"""
    parameters(::Type{S<:ParametricSignal{P,R}})

A vector of all the variational parameters in a signal of type `S`.

"""
@memoize Dict function parameters(::Type{S}) where {P,R,S<:ParametricSignal{P,R}}
    return [field for (i, field) in enumerate(fieldnames(S)) if S.types[i] == P]
end

"""
    parameters(::S<:ParametricSignal{P,R})

A vector of all the variational parameters in a signal of type `S`.

"""
function parameters(signal::S) where {P,R,S<:ParametricSignal{P,R}}
    return parameters(S)
end

#= `Parameters` INTERFACE =#

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







"""
    ConstrainedSignal(constrained::<:ParametricSignal, constraints::Vector{Symbol})

The parametric signal `constrained`, freezing all fields in `constraints`.

Frozen parameters are omitted from the `Parameters` interface.
In other words, they do not appear in `Parameters.names` or `Parameters.values`,
    and they are not mutated by `Parameters.bind`.

"""
struct ConstrainedSignal{P,R,S<:ParametricSignal{P,R}} <: Signals.SignalType{P,R}
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

"""
    ConstrainedSignal(constrained::ParametricSignal, constraints::Symbol...)

Alternate constructor, letting each field be passed as its own argument.

"""
function ConstrainedSignal(constrained::ParametricSignal, constraints::Symbol...)
    return ConstrainedSignal(constrained, collect(constraints))
end

#= `Parameters` INTERFACE =#

function Parameters.count(signal::ConstrainedSignal)
    return Parameters.count(signal.constrained) - length(signal.constraints)
end

function Parameters.names(signal::ConstrainedSignal)
    return Parameters.names(signal.constrained)[collect(signal._map)]
end

function Parameters.values(signal::ConstrainedSignal)
    return Parameters.values(signal.constrained)[collect(signal._map)]
end

function Parameters.bind(
    signal::ConstrainedSignal{P,R,S},
    x̄::AbstractVector{P}
) where {P,R,S}
    fields = parameters(signal.constrained)
    for i in eachindex(x̄)
        setfield!(signal.constrained, fields[signal._map[i]], x̄[i])
    end
end

#= `Signals` INTERFACE =#

function Signals.valueat(signal::ConstrainedSignal, t::Real)
    return Signals.valueat(signal.constrained, t)
end

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