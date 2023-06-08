# TODO (hi): add exports
import ..Parameters

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

"""
    AbstractSignal{P,R}

Super-type for all signal objects ``Ω(t)``.

# Type Parameters
- `P` denotes the type of all variational parameters. Must be a real float.
- `R` denotes the type of ``Ω(t)`` itself. May be any number type.

# Implementation

Any sub-type `S` must implement all functions in the `Parameters` module.
- In particular, `Parameters.values(Ω::S)` must return a vector of type `P`.

In addition, the following methods must be implemented:

- `(Ω::S)(t::Real)`:
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
abstract type AbstractSignal{P<:AbstractFloat,R<:Number} end

"""
    (signal::AbstractSignal{P,R})(t::Real)

The signal at time `t`, ie. ``Ω(t)``. Returns a number of type `R`.

"""
function (signal::AbstractSignal{P,R})(t::Real)::Number where {P,R}
    error("Not Implemented")
    return zero(R)
end

"""
    (signal::AbstractSignal{P,R})(t̄::AbstractVector{<:Real}; result=nothing)

Vectorized version. Returns a vector of type `R`.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function (signal::AbstractSignal{P,R})(
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    isnothing(result) && return signal(t̄; result=Vector{R}(undef, size(t̄)))
    result .= signal.(t̄)
    return result
end




"""
    partial(i::Int, signal::AbstractSignal{P,R}, t::Real)

The partial derivative ``∂Ω/∂x_i|_t``. Returns a number of type `R`.

Here ``x_i`` is the signal's i-th variational parameter
    (ie. `Parameters.names(signal)[i]`).

"""
function partial(i::Int, signal::AbstractSignal{P,R}, t::Real) where {P,R}
    error("Not Implemented")
    return zero(R)
end

"""
    partial(
        i::Int, signal::AbstractSignal{P,R}, t̄::AbstractVector{<:Real};
        result=nothing,
    )

Vectorized version. Returns a vector of type `R`.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function partial(
    i::Int,
    signal::AbstractSignal{P,R},
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    isnothing(result) && return partial(i, signal, t̄; result=Vector{R}(undef, size(t̄)))
    result .= partial.(i, Ref(signal), t̄)
    return result
end


"""
    Base.string(Ω::AbstractSignal)

A human-readable description of the signal. Returns type `string`.

"""
function Base.string(signal::AbstractSignal)
    return string(signal, Parameters.names(signal))
end

"""
    Base.string(Ω::AbstractSignal, names::AbstractVector{String})

Substitutes the default name of each variational parameter for the ones in `names`.

"""
function Base.string(Ω::AbstractSignal, names::AbstractVector{String})
    error("Not Implemented")
    return ""
end



"""
    integrate_partials(signal::AbstractSignal{P,R}, τ̄, t̄, ϕ̄; result=nothing)

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
- ϕ̄: a vector of the modulating function ``ϕ(t)`` evaluated at each point in `t̄`.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function integrate_partials(
    signal::AbstractSignal{P,R},
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractVector;
    result=nothing,
) where {P,R}
    isnothing(result) && return integrate_partials(
        signal, τ̄, t̄, ϕ̄;
        result=Vector{P}(undef, Parameters.count(signal))
    )

    # TODO (hi): I think result should be type `real(R)`, no?
    # TODO (mid): ϕ̄ defaults to ϕ(t)=1

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
    integrate_signal(signal::AbstractSignal{P,R}, τ̄, t̄, ϕ̄; result=nothing)

Integrates a signal ``Ω(t)``, modulated by a function ``ϕ(t)``.

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
- ϕ̄: a vector of the modulating function ``ϕ(t)`` evaluated at each point in `t̄`.

"""
function integrate_signal(
    signal::AbstractSignal{P,R},
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractVector,
) where {P,R}
    # TODO (mid): ϕ̄ defaults to ϕ(t)=1

    # USE PRE-ALLOCATED ARRAYS TO EXPLOIT DOT NOTATION WITHOUT ASYMPTOTIC PENALTY
    Ω̄ = array(R, size(t̄), (LABEL, :signal))
    integrand = array(P, size(t̄), (LABEL, :integrand))

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    Ω̄ = signal(t̄; result=Ω̄)
    integrand .= τ̄ .* real.(Ω̄ .* ϕ̄)

    return sum(integrand)
end


