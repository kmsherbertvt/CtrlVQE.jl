import ..Parameter

##########################################################################################
#=                                  ABSTRACT INTERFACES
=#




"""
NOTE: Implements `Parameter` interface.
"""
abstract type AbstractSignal end

# In addition to `Parameter` interface:
(::AbstractSignal)(t::Real)::Number = error("Not Implemented")
partial(i::Int, ::AbstractSignal, t::Real)::Number = error("Not Implemented")


#= TODO: I kinda want to override string(⋅), but I'd want it to accept parameter names,
            so it would be a little bit fancy.
Like, string(⋅) = string(⋅, parameters(⋅)),
    but parametric signals must implement string(⋅,⋅)?

`Constrained` inserts actual number instead of parameter name for contrained values.
`Composite` surrounds each component with parentheses and joins with " + "
`Modulated` does same but joins with " ⋅ ".

That feels doable.
=#

# TODO: Check that value and slope vectorize properly; if so we don't need the following.
#
# function value(signal::AbstractSignal, t̄::AbstractVector{<:Real})
#     return [value(signal, t) for t in t̄]
# end

# function slope(signal::AbstractSignal{F}, i::Int, t̄::AbstractVector{<:Real})
#     return [slope(signal, i, t) for t in t̄]
# end


"""
    abstract type ParametricSignal ... end

# Implementation
- Subtypes S are mutable structs.
- Subtypes implement the following methods:
    (::S)(t::Real)::Number
    partial(i::Int, ::S, t::Real)::Number

By default, all fields are treated as parameters.
You can narrow this selection by implementing the `parameters` method.
But before you do that, consider using a `Constrained` signal instead.

"""
abstract type ParametricSignal <: AbstractSignal end

function parameters(signal::S) where {S<:ParametricSignal}
    return fieldnames(S)
end

Parameter.count(signal::S) where {S<:ParametricSignal} = length(parameters(signal))

function Parameter.names(signal::S) where {S<:ParametricSignal}
    return [string(field) for field in parameters(signal)]
end

function Parameter.types(signal::S) where {S<:ParametricSignal}
    return [typeof(getfield(signal, field)) for field in parameters(signal)]
end

function Parameter.bind(signal::S, x̄::AbstractVector) where {S<:ParametricSignal}
    for (i, field) in enumerate(parameters(signal))
        setfield!(signal, field, x̄[i])
    end
end

#= TO BE IMPLEMENTED BY SUB-CLASSES:
    (::S)(t::Real)::Number
    partial(i::Int, ::S, t::Real)::Number
=#








##########################################################################################
#=                              SPECIAL SIGNAL TYPES
=#


#= CONSTRAINED SIGNAL =#

struct Constrained{S<:ParametricSignal} <: AbstractSignal
    constrained::S
    constraints::Tuple{Vararg{Symbol}}
    _map::Tuple{Vararg{Int}}

    function Constrained(constrained::S, constraints::Tuple{Vararg{Symbol}}) where {S}
        fields = parameters(constrained)
        _map = Tuple(j for (j, field) in enumerate(fields) if field ∉ constraints)
        return new{S}(constrained, constraints, _map)
    end
end

Constrained(constrained, constraints::Symbol...) = Constrained(constrained, constraints)

function Parameter.count(signal::Constrained)
    return Parameter.count(signal.constrained) - length(signal.constraints)
end

Parameter.names(signal::Constrained) = Parameter.names(signal.constrained)[signal._map]
Parameter.types(signal::Constrained) = Parameter.types(signal.constrained)[signal._map]

function Parameter.bind(signal::Constrained, x̄::AbstractVector)
    fields = parameters(signal.constrained)
    for i in eachindex(x̄)
        setfield!(signal.constrained, fields[signal._map[i]], x̄[i])
    end
end

(signal::Constrained)(t::Real) = signal.constrained(t)
partial(i::Int, signal::Constrained, t::Real) = partial(signal._map[i], signal, t)




#= COMPOSITE SIGNAL =#

struct Composite{S<:AbstractSignal} <: AbstractSignal
    components::Tuple{Vararg{S}}
end

Composite(components::S...) where {S} = Composite(components)

function Parameter.count(signal::Composite)
    return sum(Parameter.count(component) for component in signal.components)
end

function Parameter.names(signal::Composite)
    names(i) = ["$name.$i" for name in Parameter.names(signal.components[i])]
    return vcat((names(i) for i in eachindex(signal.components))...)
end

function Parameter.types(signal::Composite)
    return vcat((Parameter.types(component) for component in signal.components)...)
end

function Parameter.bind(signal::Composite, x̄::AbstractVector)
    offset = 0
    for component in signal.components
        L = Parameter.count(component)
        Parameter.bind(component, x̄[offset+1:offset+L])
        offset += l
    end
end

(signal::Composite)(t::Real) = sum(component(t) for component in signal.components)

function partial(i::Int, signal::Composite, t::Real)
    for component in signal.components
        L = Parameter.count(component)
        if i <= L
            return partial(i, component, t)
        end
        i -= L
    end
end




#= MODULATED SIGNAL =#


struct Modulated{S<:AbstractSignal} <: AbstractSignal
    components::Tuple{Vararg{S}}
end

Modulated(components::S...) where {S} = Modulated(components)

function Parameter.count(signal::Modulated)
    return sum(Parameter.count(component) for component in signal.components)
end

function Parameter.names(signal::Modulated)
    names(i) = ["$name.$i" for name in Parameter.names(signal.components[i])]
    return vcat((names(i) for i in eachindex(signal.components))...)
end

function Parameter.types(signal::Modulated)
    return vcat((Parameter.types(component) for component in signal.components)...)
end

function Parameter.bind(signal::Modulated, x̄::AbstractVector)
    offset = 0
    for component in signal.components
        L = Parameter.count(component)
        Parameter.bind(component, x̄[offset+1:offset+L])
        offset += l
    end
end

(signal::Modulated)(t::Real) = prod(component(t) for component in signal.components)

function partial(i::Int, signal::Modulated, t::Real)
    ∂f = 1
    for component in signal.components
        L = Parameter.count(component)
        if 0 <= i <= L
            ∂f *= partial(i, component, t)
        else
            ∂f *= component(t)
        end
        i -= L
    end
    return ∂f
end


##########################################################################################
#=                              COMMON PARAMETRIC SIGNALS
=#

#= CONSTANT SIGNAL =#

mutable struct Constant{F} <: ParametricSignal
    A::F    # CONSTANT VALUE
end

(signal::Constant)(t::Real) = signal.A
partial(i::Int, ::Constant{F}, t::Real) where {F} = zero(F)





#= STEP FUNCTION =#

mutable struct StepFunction{F} <: ParametricSignal
    A::F    # CONSTANT VALUE AFTER STEP
    s::F    # TIME COORDINATE OF STEP
end

function (signal::StepFunction{F})(t::Real) where {F}
    return t < signal.s ?   zero(F)
        :  t > signal.s ?   signal.A
        :                   signal.A / 2
end

function partial(i::Int, signal::StepFunction, t::Real)
    field = parameters(signal)[i]
    return field == :A ?    partial_A(signal, t)
        :  field == :s ?    partial_s(signal, t)
        :                   error("Not Implemented")

function partial_A(signal::StepFunction{F}, t::Real) where {F}
    return t < signal.s ?   zero(F)
        :  t > signal.s ?   one(F)
        :                   one(F) / 2

function partial_s(signal::StepFunction{F}, t::Real) where {F}
    return - signal.A * (t ≈ signal.s)
    #= NOTE: This method is not stable, so you should usually constrain `s`.

    Properly speaking, the gradient is ``-A⋅δ(t-s)``
        This is NOT well-defined in float arithmetic,
            and is especially ill-defined in time-discretization.
        The best I could do is ``δ(t) → Θ(τ/2+t)⋅Θ(τ/2-t) / τ`` where τ is time-step.
        This would require an additional parameter τ, which is *doable* but tedious,
            and empirically it doesn't really function any better
            than the lazy definition ``δ(t) → t ≈ 0``.

    =#


# TODO: Cosine, Gaussian








