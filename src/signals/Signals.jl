import ..Parameters
export SignalType
export valueat, partial, integrate_signal, integrate_partials

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

"""
    SignalType{P,R}

Super-type for all signal objects ``Ω(t)``.

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
    valueat(signal::SignalType{P,R}, t::Real)

The signal at time `t`, ie. ``Ω(t)``. Returns a number of type `R`.
"""
function valueat(signal::SignalType{P,R}, t::Real) where {P,R}
    error("Not Implemented")
    return zero(R)
end

"""
    valueat(signal::SignalType{P,R}, t̄::AbstractVector{<:Real}; result=nothing)

Vectorized version. Returns a vector of type `R`.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function valueat(
    signal::SignalType{P,R},
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    isnothing(result) && return valueat(signal, t̄; result=Vector{R}(undef, size(t̄)))
    for (i, t) in enumerate(t̄)
        result[i] = valueat(signal, t)
    end
    return result
end


"""
    (signal::SignalType{P,R})(t::Real)

Syntactic sugar: if Ω is a `SignalType`, then `Ω(t)` gives `valueat(Ω,t)`.

"""
(signal::SignalType)(t) = valueat(signal, t)


"""
    partial(i::Int, signal::SignalType{P,R}, t::Real)

The partial derivative ``∂Ω/∂x_i|_t``. Returns a number of type `R`.

Here ``x_i`` is the signal's i-th variational parameter
    (ie. `Parameters.names(signal)[i]`).

"""
function partial(i::Int, signal::SignalType{P,R}, t::Real) where {P,R}
    error("Not Implemented")
    return zero(R)
end

"""
    partial(
        i::Int, signal::SignalType{P,R}, t̄::AbstractVector{<:Real};
        result=nothing,
    )

Vectorized version. Returns a vector of type `R`.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function partial(
    i::Int,
    signal::SignalType{P,R},
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    isnothing(result) && return partial(i, signal, t̄; result=Vector{R}(undef, size(t̄)))
    result .= partial.(i, Ref(signal), t̄)
    return result
end


"""
    Base.string(Ω::SignalType)

A human-readable description of the signal. Returns type `string`.

"""
function Base.string(signal::SignalType)
    return string(signal, Parameters.names(signal))
end

"""
    Base.string(Ω::SignalType, names::AbstractVector{String})

Substitutes the default name of each variational parameter for the ones in `names`.

"""
function Base.string(Ω::SignalType, names::AbstractVector{String})
    error("Not Implemented")
    return ""
end



"""
    integrate_partials(signal::SignalType{P,R}, τ̄, t̄; ϕ̄=1, result=nothing)

Integrates each partial derivative ``∂Ω/∂x_i|_t``, modulated by a function ``ϕ(t)``.

Specifically, this method returns a vector of integrals `I`,
    where `I[i]` is the *real part* of ``∫ ϕ(t) ⋅ ∂Ω/∂x_i|_t dt``.

Taking the real part is a little ad hoc.
If Ω and ϕ are complex functions -
    let's just say ``Ω(t)=α(t)+i β(t)`` and ``ϕ(t)=ϕ_α(t) - i ϕ_β(t)`` -
    the integrals become ``∫ ϕ_α(t) ⋅ ∂α/∂x_i|_t dt + ∫ ϕ_β(t) ⋅ ∂β/∂x_i|_t dt``.
This turns out to be the relevant quantity in many gradient calculations.

# Arguments
- signal
- τ̄: a vector of time spacings, as given by `Evolutions.trapezoidaltimegrid`.
- t̄: a vector of time points, as given by `Evolutions.trapezoidaltimegrid`.

# Keyword Arguments
- ϕ̄: a vector of the modulating function ``ϕ(t)`` evaluated at each point in `t̄`.
    Alternatively, ϕ̄ may be scalar to represent a constant function.
- result: a pre-allocated array of compatible type and shape (optional)

"""
function integrate_partials(
    signal::SignalType{P,R},
    τ̄::AbstractVector,
    t̄::AbstractVector;
    ϕ̄=1,
    result=nothing,
) where {P,R}
    isnothing(result) && return integrate_partials(
        signal, τ̄, t̄;
        ϕ̄=ϕ̄,
        result=Vector{real(R)}(undef, Parameters.count(signal))
    )

    # TEMPORARY VARIABLES NEEDED IN GRADIENT INTEGRALS
    ∂̄ = array(R, size(t̄), (LABEL, :signal))
    integrand = array(P, size(t̄), (LABEL, :integrand))

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    for k in 1:Parameters.count(signal)
        ∂̄ = Signals.partial(k, signal, t̄; result=∂̄)
        integrand .= τ̄ .* real.(∂̄ .* ϕ̄)
        result[k] = sum(integrand)
    end

    return result
end

"""
    integrate_signal(signal::SignalType{P,R}, τ̄, t̄; ϕ̄=1)

Integrates a signal ``Ω(t)``, optionally modulated by a function ``ϕ(t)``.

Specifically, this method returns the *real part* of ``∫ ϕ(t) ⋅ Ω(t) dt``.

Taking the real part is a little ad hoc.
If Ω and ϕ are complex functions -
    let's just say ``Ω(t)=α(t)+i β(t)`` and ``ϕ(t)=t[ϕ_β(t) + i ϕ_α(t)]`` -
    the integral becomes ``∫ t⋅ϕ_β(t)⋅α(t) dt - ∫ t⋅ϕ_α(t)⋅β(t) dt``.
This turns out to be the relevant quantity in transmon frequency gradient calculations.

# Arguments
- signal
- τ̄: a vector of time spacings, as given by `Evolutions.trapezoidaltimegrid`.
- t̄: a vector of time points, as given by `Evolutions.trapezoidaltimegrid`.

# Keyword Arguments
- ϕ̄: a vector of the modulating function ``ϕ(t)`` evaluated at each point in `t̄`.
    Alternatively, ϕ̄ may be scalar to represent a constant function.

"""
function integrate_signal(
    signal::SignalType{P,R},
    τ̄::AbstractVector,
    t̄::AbstractVector;
    ϕ̄=1,
) where {P,R}
    # USE PRE-ALLOCATED ARRAYS TO EXPLOIT DOT NOTATION WITHOUT ASYMPTOTIC PENALTY
    Ω̄ = array(R, size(t̄), (LABEL, :signal))
    integrand = array(P, size(t̄), (LABEL, :integrand))

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    Ω̄ = valueat(signal, t̄; result=Ω̄)
    integrand .= τ̄ .* real.(Ω̄ .* ϕ̄)

    return sum(integrand)
end


