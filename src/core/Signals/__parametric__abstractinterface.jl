import .Signals: SignalType

"""
    ParametricSignal{P,R}

The super-type for simple user-defined SignalTypes.

# Type Parameters
- `P` denotes the type of all variational parameters. Must be a real float.
- `R` denotes the type of ``Ω(t)`` itself. May be any number type.

# Implementation

Concrete sub-types `S` must be mutable structs,
    and all of its variational parameters
    (as indicated by the `parameters` function) must have type `P`.

The following methods must be implemented:

- `Signals.parameters(Ω::S)`:
        returns a tuple of the fields in `S` treated as variational parameters.

- `Signals.valueat(Ω::S, t::Real)`:
        the actual function ``Ω(t)``. Must return a number of type `R`.

- `Signals.partial(k::Int, Ω::S, t::Real)`:
        the partial derivative ``∂Ω/∂x_k`` evaluated at time `t`,
        where ``x_k`` is Ω's k-th variational parameter
        (ie. `Parameters.names(Ω)[k]`).
        Must return a number of type `R`.

- `Base.string(Ω::S, names::AbstractVector{String})`:
        a human-readable description of the signal,
        inserting each element of `names` in the place of the corresponding parameter.
    For example, complex constant signals may return a description like "A + i B",
        where `A` and `B` are the "names" given by the `names` argument.

"""
abstract type ParametricSignal{P,R} <: SignalType{P,R} end

"""
    parameters(::S<:ParametricSignal{P,R})::Tuple{Symbol...}

Returns a tuple of the fields in `S` treated as variational parameters.

"""
function parameters end