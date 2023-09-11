import ..Parameters, ..Signals
export CompositeSignal

import ..Signals: AbstractSignal

"""
    CompositeSignal(components::AbstractVector{<:AbstractSignal{P,R}})

A signal which is the sum of each sub-signal in `components`.

Note that each component must share the same type parameters `P` and `R`.

"""
struct CompositeSignal{P,R} <: AbstractSignal{P,R}
    components::Vector{AbstractSignal{P,R}}

    function CompositeSignal(components::AbstractVector{<:AbstractSignal{P,R}}) where {P,R}
        return new{P,R}(convert(Vector{AbstractSignal{P,R}}, components))
    end
end

"""
    CompositeSignal(components::AbstractSignal...)

Alternate constructor, letting each component be passed as its own argument.

"""
function CompositeSignal(components::AbstractSignal{P,R}...) where {P,R}
    return CompositeSignal(AbstractSignal{P,R}[component for component in components])
end

#= `Parameters` INTERFACE =#

function Parameters.count(signal::CompositeSignal)
    return sum(Parameters.count(component)::Int for component in signal.components)
end

function Parameters.names(signal::CompositeSignal)
    allnames = String[]
    for (i, component) in enumerate(signal.components)
        for name in Parameters.names(component)::Vector{String}
            push!(allnames, "$name.$i")
        end
    end
    return allnames
end

function Parameters.values(signal::CompositeSignal{P,R}) where {P,R}
    allvalues = P[]
    for component in signal.components
        append!(allvalues, Parameters.values(component)::Vector{P})
    end
    return allvalues
end

function Parameters.bind(signal::CompositeSignal{P,R}, x̄::AbstractVector{P}) where {P,R}
    offset = 0
    for component in signal.components
        L = Parameters.count(component)::Int
        Parameters.bind(component, x̄[offset+1:offset+L])
        offset += L
    end
end

#= `Signals` INTERFACE =#

function (signal::CompositeSignal{P,R})(t::Real) where {P,R}
    total = zero(R)
    for component in signal.components
        total += component(t)::R
    end
    return total
end

function Signals.partial(i::Int, signal::CompositeSignal{P,R}, t::Real) where {P,R}
    for component in signal.components
        L = Parameters.count(component)::Int
        if i <= L
            return Signals.partial(i, component, t)::R
        end
        i -= L
    end
    return zero(R)  # NOTE: This can never happen for valid i, but helps the compiler.
end

function Base.string(signal::CompositeSignal, names::AbstractVector{String})
    texts = String[]
    offset = 0
    for component in signal.components
        L = Parameters.count(component)::Int
        text = string(component, names[offset+1:offset+L])::String
        push!(texts, "($text)")
        offset += L
    end

    return join(texts, " + ")
end
