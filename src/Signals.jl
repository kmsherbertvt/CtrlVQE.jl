using Memoization: @memoize
import ..Parameters

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

##########################################################################################
#=                                  ABSTRACT INTERFACES
=#




"""
NOTE: Implements `Parameters` interface.
"""
abstract type AbstractSignal{P<:AbstractFloat,R<:Number} end

# In addition to `Parameters` interface:
(::AbstractSignal)(t::Real)::Number = error("Not Implemented")
partial(i::Int, ::AbstractSignal, t::Real)::Number = error("Not Implemented")
    # NOTE: Return type is R

Base.string(::AbstractSignal, ::AbstractVector{String})::String = error("Not Implemented")



# VECTORIZED METHODS
function (signal::AbstractSignal{P,R})(
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    isnothing(result) && return signal(t̄; result=Vector{R}(undef, size(t̄)))
    result .= signal.(t̄)
    return result
end

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

# CONVENIENCE FUNCTIONS
function Base.string(signal::AbstractSignal)
    return string(signal, Parameters.names(signal))
end

function integrate_partials(
    signal::AbstractSignal{P,R},
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractVector;
    result=nothing,
) where {P,R}
    # NOTE: Calculates ∫τ ℜ(∂k⋅ϕ) for each parameter k. Let ϕ = ϕα - 𝑖 ϕβ to get desired gradient calculation.
    isnothing(result) && return integrate_partials(
        signal, τ̄, t̄, ϕ̄;
        result=Vector{P}(undef, Parameters.count(signal))
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

function integrate_signal(
    signal::AbstractSignal{P,R},
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractVector,
) where {P,R}
    # NOTE: Calculates ∫τ ℜ(Ω⋅ϕ). Let ϕ = t⋅(ϕβ + 𝑖 ϕα) to get frequency gradient.

    # USE PRE-ALLOCATED ARRAYS TO EXPLOIT DOT NOTATION WITHOUT ASYMPTOTIC PENALTY
    Ω̄ = array(R, size(t̄), (LABEL, :signal))
    integrand = array(P, size(t̄), (LABEL, :integrand))

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    Ω̄ = signal(t̄; result=Ω̄)
    integrand .= τ̄ .* real.(Ω̄ .* ϕ̄)

    return sum(integrand)
end


