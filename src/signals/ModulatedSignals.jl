import ..Parameters, ..Signals
export ModulatedSignal

import ..Signals: SignalType

"""
    ModulatedSignal(components::AbstractVector{<:SignalType{P,R}})

A signal which is the product of each sub-signal in `components`.

Note that each component must share the same type parameters `P` and `R`.

"""
struct ModulatedSignal{P,R} <: SignalType{P,R}
    components::Vector{SignalType{P,R}}

    function ModulatedSignal(components::AbstractVector{<:SignalType{P,R}}) where {P,R}
        return new{P,R}(convert(Vector{SignalType{P,R}}, components))
    end
end

"""
    ModulatedSignal(components::SignalType...)

Alternate constructor, letting each component be passed as its own argument.

"""
function ModulatedSignal(components::SignalType{P,R}...) where {P,R}
    return ModulatedSignal(SignalType{P,R}[component for component in components])
end

#= `Parameters` INTERFACE =#

function Parameters.count(signal::ModulatedSignal)
    return sum(Parameters.count(component)::Int for component in signal.components)
end

function Parameters.names(signal::ModulatedSignal)
    allnames = String[]
    for (i, component) in enumerate(signal.components)
        for name in Parameters.names(component)::Vector{String}
            push!(allnames, "$name.$i")
        end
    end
    return allnames
end

function Parameters.values(signal::ModulatedSignal{P,R}) where {P,R}
    allvalues = P[]
    for component in signal.components
        append!(allvalues, Parameters.values(component)::Vector{P})
    end
    return allvalues
end

function Parameters.bind(signal::ModulatedSignal{P,R}, x̄::AbstractVector{P}) where {P,R}
    offset = 0
    for component in signal.components
        L = Parameters.count(component)::Int
        Parameters.bind(component, x̄[offset+1:offset+L])
        offset += L
    end
end

#= `Signals` INTERFACE =#

function Signals.valueat(signal::ModulatedSignal{P,R}, t::Real) where {P,R}
    return prod(Signals.valueat(component, t)::R for component in signal.components)
end

function Signals.partial(i::Int, signal::ModulatedSignal{P,R}, t::Real) where {P,R}
    ∂f = one(R)
    for component in signal.components
        L = Parameters.count(component)::Int
        if 1 <= i <= L
            ∂f *= Signals.partial(i, component, t)::R
        else
            ∂f *= Signals.valueat(component, t)::R
        end
        i -= L
    end
    return ∂f
end

function Base.string(signal::ModulatedSignal, names::AbstractVector{String})
    texts = String[]
    offset = 0
    for component in signal.components
        L = Parameters.count(component)::Int
        text = string(component, names[offset+1:offset+L])::String
        push!(texts, "($text)")
        offset += L
    end

    return join(texts, " ⋅ ")
end
