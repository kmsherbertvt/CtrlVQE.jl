import ...Parameters, ...Signals
import ...Signals: AbstractSignal

#= COMPOSITE SIGNAL =#

struct CompositeSignal{P,R} <: AbstractSignal{P,R}
    components::Vector{AbstractSignal{P,R}}

    function CompositeSignal(components::AbstractVector{<:AbstractSignal{P,R}}) where {P,R}
        return new{P,R}(convert(Vector{AbstractSignal{P,R}}, components))
    end
end

function CompositeSignal(components::AbstractSignal{P,R}...) where {P,R}
    return CompositeSignal(AbstractSignal{P,R}[component for component in components])
end

function Parameters.count(signal::CompositeSignal)
    return sum(Parameters.count(component) for component in signal.components)
end

function Parameters.names(signal::CompositeSignal)
    names(i) = ["$name.$i" for name in Parameters.names(signal.components[i])]
    return vcat((names(i) for i in eachindex(signal.components))...)
end

function Parameters.values(signal::CompositeSignal)
    return vcat((Parameters.values(component) for component in signal.components)...)
end

function Parameters.bind(signal::CompositeSignal{P,R}, x̄::AbstractVector{P}) where {P,R}
    offset = 0
    for component in signal.components
        L = Parameters.count(component)
        Parameters.bind(component, x̄[offset+1:offset+L])
        offset += L
    end
end

function (signal::CompositeSignal{P,R})(t::Real) where {P,R}
    total = zero(R)
    for component in signal.components
        total += component(t)
    end
    return total
end

function Signals.partial(i::Int, signal::CompositeSignal{P,R}, t::Real) where {P,R}
    for component in signal.components
        L = Parameters.count(component)
        if i <= L
            return Signals.partial(i, component, t)
        end
        i -= L
    end
    return zero(R)  # NOTE: This can never happen for valid i, but helps the compiler.
end

function Base.string(signal::CompositeSignal, names::AbstractVector{String})
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
