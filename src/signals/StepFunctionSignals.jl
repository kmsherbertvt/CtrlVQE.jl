import ..Signals
export StepFunction

import ..ParametricSignals: ParametricSignal, parameters

"""
    StepFunction(A::F, s::F) where {F<:AbstractFloat}

The piecewise signal ``Ω(t) = A⋅Θ(t-s)``, where ``Θ`` is the Heaviside step function.

"""
mutable struct StepFunction{F} <: ParametricSignal{F,F}
    A::F    # CONSTANT VALUE AFTER STEP
    s::F    # TIME COORDINATE OF STEP
end

function Signals.valueat(signal::StepFunction{F}, t::Real) where {F}
    return (t < signal.s ?  zero(F)
        :   t > signal.s ?  signal.A
        :                   signal.A / 2)
end

function Signals.partial(i::Int, signal::StepFunction, t::Real)
    field = parameters(signal)[i]
    return (field == :A ?   partial_A(signal, t)
        :   field == :s ?   partial_s(signal, t)
        :                   error("Not Implemented"))
end

function partial_A(signal::StepFunction{F}, t::Real) where {F}
    return (t < signal.s ?  zero(F)
        :   t > signal.s ?  one(F)
        :                   one(F) / 2)
end

function partial_s(signal::StepFunction{F}, t::Real) where {F}
    return - signal.A * (t ≈ signal.s)
end
    #= NOTE: This method is not numerically stable, so you should usually constrain `s`.

    Properly speaking, the gradient is ``-A⋅δ(t-s)``
        This is NOT well-defined in float arithmetic,
            and is especially ill-defined in time-discretization.
        The best I could do is ``δ(t) → Θ(τ/2+t)⋅Θ(τ/2-t) / τ`` where τ is time-step.
        This would require an additional parameter τ, which is *doable* but tedious,
            and empirically it doesn't really function any better
            than the lazy definition ``δ(t) → t ≈ 0``.

    =#

function Base.string(::StepFunction, names::AbstractVector{String})
    A, s = names
    return "$A Θ(t-$s)"
end

