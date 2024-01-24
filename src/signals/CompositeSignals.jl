import ..Parameters, ..Signals
export CompositeSignal, WeightedCompositeSignal

import ..Signals: SignalType

"""
    CompositeSignal(components::AbstractVector{<:SignalType{P,R}})

A signal which is the sum of each sub-signal in `components`.

Note that each component must share the same type parameters `P` and `R`.

"""
struct CompositeSignal{P,R} <: SignalType{P,R}
    components::Vector{SignalType{P,R}}

    function CompositeSignal(components::AbstractVector{<:SignalType{P,R}}) where {P,R}
        return new{P,R}(convert(Vector{SignalType{P,R}}, components))
    end
end

"""
    CompositeSignal(components::SignalType...)

Alternate constructor, letting each component be passed as its own argument.

"""
function CompositeSignal(components::SignalType{P,R}...) where {P,R}
    return CompositeSignal(SignalType{P,R}[component for component in components])
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

function Signals.valueat(signal::CompositeSignal{P,R}, t::Real) where {P,R}
    total = zero(R)
    for component in signal.components
        total += Signals.valueat(component, t)::R
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




"""
    WeightedCompositeSignal(components::AbstractVector{<:SignalType{P,R}})

A signal which is the weighted average of each sub-signal in `components`.

Note that each component must share the same type parameters `P` and `R`.

"""
struct WeightedCompositeSignal{P,R} <: SignalType{P,R}
    components::Vector{SignalType{P,R}}
    weights::Vector{P}

    function WeightedCompositeSignal(
        components::AbstractVector{<:SignalType{P,R}},
        weights::AbstractVector{P},
    ) where {P,R}
        # VERIFY CONSISTENCY IN PARAMETER LENGTHS
        @assert length(components) == length(weights)

        return new{P,R}(
            convert(Vector{SignalType{P,R}}, components),
            convert(Vector{P}, weights),
        )
    end
end

"""
    WeightedCompositeSignal(components::SignalType...)

Alternate constructor, letting each component be passed as its own argument.

Using this constructor, weights initialize to zero.

"""
function WeightedCompositeSignal(components::SignalType{P,R}...) where {P,R}
    return WeightedCompositeSignal(
        SignalType{P,R}[component for component in components],
        zeros(P, length(components)),
    )
end

#= `Parameters` INTERFACE =#

function Parameters.count(signal::WeightedCompositeSignal)
    L = length(signal.weights)
    L += sum(Parameters.count(component)::Int for component in signal.components)
    return L
end

function Parameters.names(signal::WeightedCompositeSignal)
    allnames = String[]
    for (i, component) in enumerate(signal.components)
        push!(allnames, "c$i")  # USE LETTER 'c' TO DENOTE WEIGHT
        for name in Parameters.names(component)::Vector{String}
            push!(allnames, "$name.$i")
        end
    end
    return allnames
end

function Parameters.values(signal::WeightedCompositeSignal{P,R}) where {P,R}
    allvalues = P[]
    for (i, component) in enumerate(signal.components)
        push!(allvalues, signal.weights[i])
        append!(allvalues, Parameters.values(component)::Vector{P})
    end
    return allvalues
end

function Parameters.bind(
    signal::WeightedCompositeSignal{P,R},
    x̄::AbstractVector{P},
) where {P,R}
    offset = 0
    for (i, component) in enumerate(signal.components)
        # BIND WEIGHT
        signal.weights[i] = x̄[offset+1]
        offset += 1

        # BIND COMPONENT
        L = Parameters.count(component)::Int
        Parameters.bind(component, x̄[offset+1:offset+L])
        offset += L
    end
end

#= `Signals` INTERFACE =#

function Signals.valueat(
    signal::WeightedCompositeSignal{P,R},
    t::Real,
) where {P,R}
    total = zero(R)
    for (i, component) in enumerate(signal.components)
        total += signal.weights[i] * Signals.valueat(component, t)::R
    end
    return total
end

# TODO
function Signals.partial(
    i::Int,
    signal::WeightedCompositeSignal{P,R},
    t::Real,
) where {P,R}
    offset = 0
    for (j, component) in enumerate(signal.components)
        # HANDLE WEIGHT PARTIAL - JUST EVALUATING THE COMPONENT
        if i == offset + 1
            return Signals.valueat(component, t)::R
        end
        offset += 1

        # HANDLE COMPONENT PARTIAL - JUST DELEGATE
        L = Parameters.count(component)::Int
        if i-offset <= L
            return signal.weights[j] * Signals.partial(i-offset, component, t)::R
        end
        offset += L
    end
    return zero(R)  # NOTE: This can never happen for valid i, but helps the compiler.
end

function Base.string(signal::WeightedCompositeSignal, names::AbstractVector{String})
    texts = String[]
    offset = 0
    for component in signal.components
        # FETCH WEIGHT NAME
        weight = names[offset+1]
        offset += 1

        # CONSTRUCT COMPONENT STRINGS
        L = Parameters.count(component)::Int
        text = string(component, names[offset+1:offset+L])::String
        offset += L

        # COMBINE WEIGHT WITH COMPONENT
        push!(texts, "$weight ($text)")
    end

    return join(texts, " + ")
end
