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
    return sum(Parameters.count(component) for component in signal.components)
end

function Parameters.names(signal::ModulatedSignal)
    names(i) = ["$name.$i" for name in Parameters.names(signal.components[i])]
    return vcat((names(i) for i in eachindex(signal.components))...)
end

function Parameters.values(signal::ModulatedSignal)
    return vcat((Parameters.values(component) for component in signal.components)...)
end

function Parameters.bind(signal::ModulatedSignal{P,R}, x̄::AbstractVector{P}) where {P,R}
    offset = 0
    for component in signal.components
        L = Parameters.count(component)
        Parameters.bind(component, x̄[offset+1:offset+L])
        offset += L
    end
end

(signal::ModulatedSignal)(t::Real) = prod(component(t) for component in signal.components)

function Signals.partial(i::Int, signal::ModulatedSignal{P,R}, t::Real) where {P,R}
    ∂f = one(R)
    for component in signal.components
        L = Parameters.count(component)
        if 1 <= i <= L
            ∂f *= Signals.partial(i, component, t)
        else
            ∂f *= component(t)
        end
        i -= L
    end
    return ∂f
end

function Base.string(signal::ModulatedSignal, names::AbstractVector{String})
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
