module CompositeSignals
    export CompositeSignal, Composed

    import ..CtrlVQE: Parameters
    import ..CtrlVQE: Signals

    """
        CompositeSignal(components::AbstractVector{<:SignalType{P,R}})

    A signal which is the sum of each sub-signal in `components`.

    Each component should be the same type of signal, for type stability.
    If you need to compose different types of signals,
        you should probably implement your own custom `SignalType`.

    """
    struct CompositeSignal{P,R,S<:Signals.SignalType{P,R}} <: Signals.SignalType{P,R}
        components::Vector{S}

        function CompositeSignal(
            components::AbstractVector{<:Signals.SignalType{P,R}},
        ) where {P,R}
            S = eltype(components)
            components = convert(Vector{S}, components)
            return new{P,R,S}(components)
        end
    end

    #= `Parameters` INTERFACE =#

    function Parameters.count(signal::CompositeSignal)
        return sum(
            Parameters.count(component)::Int for component in signal.components;
            init=0,
        )
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

    function Parameters.values(signal::CompositeSignal{P,R,S}) where {P,R,S}
        allvalues = P[]
        for component in signal.components
            append!(allvalues, Parameters.values(component)::Vector{P})
        end
        return allvalues
    end

    function Parameters.bind!(
        signal::CompositeSignal{P,R,S},
        x::AbstractVector{P}
    ) where {P,R,S}
        offset = 0
        for component in signal.components
            L = Parameters.count(component)::Int
            Parameters.bind!(component, x[offset+1:offset+L])
            offset += L
        end
        return signal
    end

    #= `Signals` INTERFACE =#

    function Signals.valueat(signal::CompositeSignal{P,R,S}, t::Real) where {P,R,S}
        total = zero(R)
        for component in signal.components
            total += Signals.valueat(component, t)::R
        end
        return total
    end

    function Signals.partial(
        i::Int,
        signal::CompositeSignal{P,R,S},
        t::Real,
    ) where {P,R,S}
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
        Composed(components::SignalType...)

    Convenience constructor to combine multiple signals into a `CompositeSignal`.

    """
    function Composed(components::Signals.SignalType{P,R}...) where {P,R}
        return CompositeSignal(collect(components))
    end

    #= TODO: We've removed WeightedCompositeSignal here. Where ought it go? =#
end