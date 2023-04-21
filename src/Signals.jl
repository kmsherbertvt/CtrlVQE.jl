import ..Parameters

##########################################################################################
#=                                  ABSTRACT INTERFACES
=#




"""
NOTE: Implements `Parameters` interface.
"""
abstract type AbstractSignal end

# In addition to `Parameters` interface:
(::AbstractSignal)(t::Real)::Number = error("Not Implemented")
partial(i::Int, ::AbstractSignal, t::Real)::Number = error("Not Implemented")
Base.string(::AbstractSignal, ::AbstractVector{String})::String = error("Not Implemented")

# VECTORIZED METHODS
(signal::AbstractSignal)(t̄::AbstractVector{<:Real}) = [signal(t) for t in t̄]
function partial(i::Int, signal::AbstractSignal, t̄::AbstractVector{<:Real})
    return [partial(i, signal, t) for t in t̄]
end

# CONVENIENCE FUNCTIONS
function Base.string(signal::AbstractSignal)
    return string(signal, Parameters.names(signal))
end


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

Parameters.count(signal::S) where {S<:ParametricSignal} = length(parameters(signal))

function Parameters.names(signal::S) where {S<:ParametricSignal}
    return [string(field) for field in parameters(signal)]
end

function Parameters.values(signal::S) where {S<:ParametricSignal}
    return identity.([getfield(signal, field) for field in parameters(signal)])
end

function Parameters.bind(signal::S, x̄::AbstractVector) where {S<:ParametricSignal}
    for (i, field) in enumerate(parameters(signal))
        setfield!(signal, field, x̄[i])
    end
end

#= TO BE IMPLEMENTED BY SUB-CLASSES:
    (::S)(t::Real)::Number
    partial(i::Int, ::S, t::Real)::Number
    Base.string(::S, ::AbstractVector{String})::String
=#








##########################################################################################
#=                              SPECIAL SIGNAL TYPES
=#


#= CONSTRAINED SIGNAL =#

struct Constrained{S<:ParametricSignal} <: AbstractSignal
    constrained::S
    constraints::Vector{Symbol}
    _map::Vector{Int}

    function Constrained(constrained::S, constraints::Vector{Symbol}) where {S}
        fields = parameters(constrained)
        _map = [j for (j, field) in enumerate(fields) if field ∉ constraints]
        return new{S}(constrained, constraints, _map)
    end
end

function Constrained(constrained, constraints::Symbol...)
    return Constrained(constrained, collect(constraints))
end

function Parameters.count(signal::Constrained)
    return Parameters.count(signal.constrained) - length(signal.constraints)
end

function Parameters.names(signal::Constrained)
    return Parameters.names(signal.constrained)[collect(signal._map)]
end

function Parameters.values(signal::Constrained)
    return Parameters.values(signal.constrained)[collect(signal._map)]
end

function Parameters.bind(signal::Constrained, x̄::AbstractVector)
    fields = parameters(signal.constrained)
    for i in eachindex(x̄)
        setfield!(signal.constrained, fields[signal._map[i]], x̄[i])
    end
end

(signal::Constrained)(t::Real) = signal.constrained(t)
function partial(i::Int, signal::Constrained, t::Real)
    return partial(signal._map[i], signal.constrained, t)
end

function Base.string(signal::Constrained, names::AbstractVector{String})
    newnames = string.(Parameters.values(signal.constrained))
    for i in eachindex(names)
        newnames[signal._map[i]] = names[i]
    end
    return Base.string(signal.constrained, newnames)
end

#= ARBITRARY SIGNAL =#

#= NOTE: This is a hack.

Devices need to have some signal type.
But we should be able to adaptively CHANGE the signal, without having to change the device.
Thus, this wrapper.

This struct is by itself ill-designed; its field is an abstract type.
But, all its methods should nevertheless be type-stable,
    because we've explicitly declared return types in our "abstract" methods.
=#
struct ArbitrarySignal <: AbstractSignal
    S::AbstractSignal
end

Parameters.count(S::ArbitrarySignal) = Parameters.count(S.S)
Parameters.names(S::ArbitrarySignal) = Parameters.names(S.S)
Parameters.values(S::ArbitrarySignal) = Parameters.values(S.S)
Parameters.bind(S::ArbitrarySignal, x̄::AbstractVector) = Parameters.bind(S.S, x̄)

(S::ArbitrarySignal)(t::Real) = S.S(t)
partial(i::Int, S::ArbitrarySignal, t::Real) = partial(i, S.S, t)
Base.string(S::ArbitrarySignal, names::AbstractVector{String}) = Base.string(S.S, names)



#= COMPOSITE SIGNAL =#

struct Composite <: AbstractSignal
    components::Vector{ArbitrarySignal}
end

function Composite(components::AbstractSignal...)
    return Composite([ArbitrarySignal(component) for component in components])
end

function Parameters.count(signal::Composite)
    return sum(Parameters.count(component) for component in signal.components)
end

function Parameters.names(signal::Composite)
    names(i) = ["$name.$i" for name in Parameters.names(signal.components[i])]
    return vcat((names(i) for i in eachindex(signal.components))...)
end

function Parameters.values(signal::Composite)
    return vcat((Parameters.values(component) for component in signal.components)...)
end

function Parameters.bind(signal::Composite, x̄::AbstractVector)
    offset = 0
    for component in signal.components
        L = Parameters.count(component)
        Parameters.bind(component, x̄[offset+1:offset+L])
        offset += L
    end
end

(signal::Composite)(t::Real) = sum(component(t) for component in signal.components)

function partial(i::Int, signal::Composite, t::Real)
    for component in signal.components
        L = Parameters.count(component)
        if i <= L
            return partial(i, component, t)
        end
        i -= L
    end
end

function Base.string(signal::Composite, names::AbstractVector{String})
    texts = String[]
    offset = 0
    for component in signal.components
        L = Parameters.count(component)
        text = string(component, names[offset+1:offset+L])
        push!(texts, "($text)")
        offset += L
    end

    return join(texts, " + ")
end




#= MODULATED SIGNAL =#

struct Modulated <: AbstractSignal
    components::Vector{ArbitrarySignal}
end

function Modulated(components::AbstractSignal...)
    return Modulated([ArbitrarySignal(component) for component in components])
end

function Parameters.count(signal::Modulated)
    return sum(Parameters.count(component) for component in signal.components)
end

function Parameters.names(signal::Modulated)
    names(i) = ["$name.$i" for name in Parameters.names(signal.components[i])]
    return vcat((names(i) for i in eachindex(signal.components))...)
end

function Parameters.values(signal::Modulated)
    return vcat((Parameters.values(component) for component in signal.components)...)
end

function Parameters.bind(signal::Modulated, x̄::AbstractVector)
    offset = 0
    for component in signal.components
        L = Parameters.count(component)
        Parameters.bind(component, x̄[offset+1:offset+L])
        offset += L
    end
end

(signal::Modulated)(t::Real) = prod(component(t) for component in signal.components)

function partial(i::Int, signal::Modulated, t::Real)
    ∂f = 1
    for component in signal.components
        L = Parameters.count(component)
        if 1 <= i <= L
            ∂f *= partial(i, component, t)
        else
            ∂f *= component(t)
        end
        i -= L
    end
    return ∂f
end

function Base.string(signal::Modulated, names::AbstractVector{String})
    texts = String[]
    offset = 0
    for component in signal.components
        L = Parameters.count(component)
        text = string(component, names[offset+1:offset+L])
        push!(texts, "($text)")
        offset += L
    end

    return join(texts, " ⋅ ")
end

