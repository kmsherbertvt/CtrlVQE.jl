import ...Parameters, ...Signals
import ...Signals: AbstractSignal

#= MODULATED SIGNAL =#

struct ModulatedSignal{P,R} <: AbstractSignal{P,R}
    components::Vector{AbstractSignal{P,R}}

    function ModulatedSignal(components::AbstractVector{<:AbstractSignal{P,R}}) where {P,R}
        return new{P,R}(convert(Vector{AbstractSignal{P,R}}, components))
    end
end

function ModulatedSignal(components::AbstractSignal{P,R}...) where {P,R}
    return ModulatedSignal(AbstractSignal{P,R}[component for component in components])
end

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

function (signal::ModulatedSignal{P,R})(t::Real) where {P,R}
    return prod(component(t)::R for component in signal.components)
end

function Signals.partial(i::Int, signal::ModulatedSignal{P,R}, t::Real) where {P,R}
    ∂f = one(R)
    for component in signal.components
        L = Parameters.count(component)::Int
        if 1 <= i <= L
            ∂f *= Signals.partial(i, component, t)::R
        else
            ∂f *= component(t)::R
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
