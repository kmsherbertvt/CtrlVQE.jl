"""
    SignalType{P,R}

Encapsulates a parametric and differntiable scalar function ``Ω(t)``.

# Type Parameters
- `P` denotes the type of all variational parameters. Must be a real float.
- `R` denotes the type of ``Ω(t)`` itself. May be any number type.

# Implementation

Any concrete sub-type `S` must implement all functions in the `Parameters` module.
- In particular, `Parameters.values(Ω::S)` must return a vector of type `P`.
- If you are trying to create your own signal type,
    you *probably* want to implement a `ParametricSignal`,
    which already has an implementation for the `Parameters` interface.

In addition, the following methods must be implemented:

- `valueat(Ω::S, t::Real)`:
        the actual function ``Ω(t)``. Must return a number of type `R`.

- `partial(i::Int, Ω::S, t::Real)`:
        the partial derivative ``∂Ω/∂x_i`` evaluated at time `t`,
        where ``x_i`` is Ω's i-th variational parameter (ie. `Parameters.names(Ω)[i]`).
        Must return a number of type `R`.

- `Base.string(Ω::S, names::AbstractVector{String})`:
        a human-readable description of the signal,
        inserting each element of `names` in the place of the corresponding parameter.
    For example, a complex constant signal might return a description like "\$A + i \$B",
        where `A` and `B` are the "names" given by the `names` argument.

"""
abstract type SignalType{P<:AbstractFloat,R<:Number} end


"""
    valueat(signal::SignalType{P,R}, t::Real)::R

The signal at time `t`, ie. ``Ω(t)``.
"""
function valueat end


"""
    partial(k::Int, signal::SignalType{P,R}, t::Real)::R

The partial derivative ``∂Ω/∂x_k|_t``.

Here ``x_k`` is the signal's k-th variational parameter
    (ie. `Parameters.names(signal)[k]`).

"""
function partial end

"""
    Base.string(Ω::SignalType, names::AbstractVector{String})::String

Substitutes the default name of each variational parameter for the ones in `names`.

Note that this is not the usual signature for Base.string!

"""
function Base.string(Ω::SignalType, names::AbstractVector{String})
    error("Not Implemented")
end