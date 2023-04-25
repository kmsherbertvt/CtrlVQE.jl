using Memoization: @memoize
import ..Parameters

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

##########################################################################################
#=                                  ABSTRACT INTERFACES
=#




"""
NOTE: Implements `Parameters` interface.
"""
abstract type AbstractSignal{P<:AbstractFloat,R<:Number} end

# In addition to `Parameters` interface:
(::AbstractSignal)(t::Real)::Number = error("Not Implemented")
partial(i::Int, ::AbstractSignal, t::Real)::Number = error("Not Implemented")
    # NOTE: Return type is R

Base.string(::AbstractSignal, ::AbstractVector{String})::String = error("Not Implemented")



# VECTORIZED METHODS
function (signal::AbstractSignal{P,R})(
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    isnothing(result) && return signal(t̄; result=Vector{R}(undef, size(t̄)))
    result .= signal.(t̄)
    return result
end

function partial(
    i::Int,
    signal::AbstractSignal{P,R},
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    isnothing(result) && return partial(i, signal, t̄; result=Vector{R}(undef, size(t̄)))
    result .= partial.(i, Ref(signal), t̄)
    return result
end

# CONVENIENCE FUNCTIONS
function Base.string(signal::AbstractSignal)
    return string(signal, Parameters.names(signal))
end

function integrate_partials(
    signal::AbstractSignal{P,R},
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractVector;
    result=nothing,
) where {P,R}
    # NOTE: Calculates ∫τ ℜ(∂k⋅ϕ) for each parameter k. Let ϕ = ϕα - 𝑖 ϕβ to get desired gradient calculation.
    isnothing(result) && return integrate_gradient_signal!(
        signal, τ̄, t̄, ϕα, ϕβ;
        result=Vector{P}(undef, Parameters.count(signal))
    )

    # TEMPORARY VARIABLES NEEDED IN GRADIENT INTEGRALS
    ∂̄ = array(R, size(t̄), (LABEL, :signal))
    integrand = array(P, size(t̄), (LABEL, :integrand))

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    for k in 1:Parameters.count(signal)

        ∂̄ = Signals.partial(k, signal, t̄; result=∂̄)
        integrand .= τ̄ .* real.(∂̄ .* ϕ̄)
        result[k] = sum(integrand)
    end

    return result
end

function integrate_signal(
    signal::AbstractSignal{P,R},
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractVector,
) where {P,R}
    # NOTE: Calculates ∫τ ℜ(Ω⋅ϕ). Let ϕ = t⋅(ϕβ + 𝑖 ϕα) to get frequency gradient.

    # USE PRE-ALLOCATED ARRAYS TO EXPLOIT DOT NOTATION WITHOUT ASYMPTOTIC PENALTY
    Ω̄ = array(R, size(t̄), (LABEL, :signal))
    integrand = array(P, size(t̄), (LABEL, :integrand))

    # CALCULATE GRADIENT FOR SIGNAL PARAMETERS
    Ω̄ = signal(t̄; result=Ω̄)
    integrand .= τ̄ .* real.(Ω̄ .* ϕ̄)

    return sum(integrand)
end








"""
    abstract type ParametricSignal{P,R} ... end

# Implementation
- Subtypes S are mutable structs.
- All differentiable parameters of S are of type P.
- Subtypes implement the following methods:
    (::S)(t::Real)::R
    partial(i::Int, ::S, t::Real)::R

By default, all fields of type P are treated as differentiable parameters.
You can narrow this selection by implementing:
    parameters(::S)::Vector{Symbol}
But before you do that, consider using a `Constrained` signal instead.

"""
abstract type ParametricSignal{P,R} <: AbstractSignal{P,R} end

@memoize Dict function parameters(::Type{S}) where {P,R,S<:ParametricSignal{P,R}}
    return [field for (i, field) in enumerate(fieldnames(S)) if S.types[i] == P]
end

function parameters(signal::S) where {P,R,S<:ParametricSignal{P,R}}
    return parameters(S)
end

Parameters.count(signal::S) where {S<:ParametricSignal} = length(parameters(S))

function Parameters.names(signal::S) where {S<:ParametricSignal}
    return [string(field) for field in parameters(S)]
end

function Parameters.values(signal::S) where {P,R,S<:ParametricSignal{P,R}}
    return P[getfield(signal, field) for field in parameters(S)]
end

function Parameters.bind(
    signal::S,
    x̄::AbstractVector{P}
) where {P,R,S<:ParametricSignal{P,R}}
    for (i, field) in enumerate(parameters(signal))
        setfield!(signal, field, x̄[i])
    end
end

#= TO BE IMPLEMENTED BY SUB-CLASSES:
    (::S)(t::Real)::R
    partial(i::Int, ::S, t::Real)::R
    Base.string(::S, ::AbstractVector{String})::String
=#








##########################################################################################
#=                              SPECIAL SIGNAL TYPES
=#


#= CONSTRAINED SIGNAL =#

struct Constrained{P,R,S<:ParametricSignal{P,R}} <: AbstractSignal{P,R}
    constrained::S
    constraints::Vector{Symbol}
    _map::Vector{Int}

    function Constrained(
        constrained::S,
        constraints::Vector{Symbol}
    ) where {P,R,S<:ParametricSignal{P,R}}
        fields = parameters(constrained)
        _map = [j for (j, field) in enumerate(fields) if field ∉ constraints]
        return new{P,R,S}(constrained, constraints, _map)
    end
end

function Constrained(constrained::ParametricSignal, constraints::Symbol...)
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

function Parameters.bind(signal::Constrained{P,R,S}, x̄::AbstractVector{P}) where {P,R,S}
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



#= COMPOSITE SIGNAL =#

struct Composite{P,R} <: AbstractSignal{P,R}
    components::Vector{AbstractSignal{P,R}}

    function Composite(components::AbstractVector{<:AbstractSignal{P,R}}) where {P,R}
        return new{P,R}(convert(Vector{AbstractSignal{P,R}}, components))
    end
end

function Composite(components::AbstractSignal{P,R}...) where {P,R}
    return Composite(AbstractSignal{P,R}[component for component in components])
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

function Parameters.bind(signal::Composite{P,R}, x̄::AbstractVector{P}) where {P,R}
    offset = 0
    for component in signal.components
        L = Parameters.count(component)
        Parameters.bind(component, x̄[offset+1:offset+L])
        offset += L
    end
end

function (signal::Composite{P,R})(t::Real) where {P,R}
    total = zero(R)
    for component in signal.components
        total += component(t)
    end
    return total
end

function partial(i::Int, signal::Composite{P,R}, t::Real) where {P,R}
    for component in signal.components
        L = Parameters.count(component)
        if i <= L
            return partial(i, component, t)
        end
        i -= L
    end
    return zero(R)  # NOTE: This can never happen for valid i, but helps the compiler.
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

struct Modulated{P,R} <: AbstractSignal{P,R}
    components::Vector{AbstractSignal{P,R}}

    function Modulated(components::AbstractVector{<:AbstractSignal{P,R}}) where {P,R}
        return new{P,R}(convert(Vector{AbstractSignal{P,R}}, components))
    end
end

function Modulated(components::AbstractSignal{P,R}...) where {P,R}
    return Modulated(AbstractSignal{P,R}[component for component in components])
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

function Parameters.bind(signal::Modulated{P,R}, x̄::AbstractVector{P}) where {P,R}
    offset = 0
    for component in signal.components
        L = Parameters.count(component)
        Parameters.bind(component, x̄[offset+1:offset+L])
        offset += L
    end
end

(signal::Modulated)(t::Real) = prod(component(t) for component in signal.components)

function partial(i::Int, signal::Modulated{P,R}, t::Real) where {P,R}
    ∂f = one(R)
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



#= WINDOWED SIGNAL =#

struct Windowed{P,R} <: AbstractSignal{P,R}
    windows::Vector{AbstractSignal{P,R}}
    starttimes::Vector{P}

    offsets::Vector{Int}        # CONTAINS A CUMULATIVE SUM OF PARAMETER COUNTS

    function Windowed(
        windows::AbstractVector{<:AbstractSignal{P,R}},
        starttimes::AbstractVector{<:Real},
    ) where {P,R}
        # CHECK THAT NUMBER OF WINDOWS AND STARTTIMES ARE COMPATIBLE
        if length(windows) != length(starttimes)
            error("Number of windows must match number of starttimes.")
        end

        # CONVERT windows TO VECTOR OF ABSTRACT TYPE
        windows = convert(Vector{AbstractSignal{P,R}}, windows)

        # ENSURE THAT starttimes ARE SORTED, AND MAKE TYPE CONSISTENT WITH WINDOWS
        starttimes = convert(Vector{P}, sort(starttimes))

        # CONSTRUCT `offsets` VECTOR
        offsets = Int[0]
        for (i, window) in enumerate(windows[1:end-1])
            push!(offsets, Parameters.count(window) + offsets[i])
        end

        return new{P,R}(windows, starttimes, offsets)
    end
end

function Parameters.count(signal::Windowed)
    return sum(Parameters.count(window) for window in signal.windows)
end

function Parameters.names(signal::Windowed)
    names(i) = ["$name.$i" for name in Parameters.names(signal.windows[i])]
    return vcat((names(i) for i in eachindex(signal.windows))...)
end

function Parameters.values(signal::Windowed)
    return vcat((Parameters.values(window) for window in signal.windows)...)
end

function Parameters.bind(signal::Windowed{P,R}, x̄::AbstractVector{P}) where {P,R}
    for (k, window) in enumerate(signal.windows)
        L = Parameters.count(window)
        Parameters.bind(window, x̄[1+signal.offsets[k]:L+signal.offsets[k]])
    end
end

function get_window_from_time(signal::Windowed, t::Real)
    k = findlast(starttime -> starttime ≤ t, signal.starttimes)
    isnothing(k) && error("Time $t does not fit into any window.")
    return k
end

function get_window_from_parameter(signal::Windowed, i::Int)
    k = findlast(offset -> offset < i, signal.offsets)
    isnothing(k) && error("Parameter $i does not fit into any window.")
    return k
end

function (signal::Windowed)(t::Real)
    k = get_window_from_time(signal,t)
    return signal.windows[k](t)
end

function partial(i::Int, signal::Windowed{P,R}, t::Real) where {P,R}
    kt = get_window_from_time(signal,t)
    ki = get_window_from_parameter(signal,i)
    return (kt == ki ?  partial(i-signal.offsets[ki], signal.windows[kt], t)
        :               zero(R)
    )
end

function Base.string(signal::Windowed, names::AbstractVector{String})
    texts = String[]
    for window in signal.windows
        L = Parameters.count(window)
        text = string(window, names[1+signal.offsets[k]:L+signal.offsets[k]])
        push!(texts, "($text) | t∊[$s1,$s2)")
    end

    return join(texts, "; ")
end

# VECTORIZED METHODS
function (signal::Windowed{P,R})(
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    isnothing(result) && return signal(t̄; result=Vector{R}(undef, size(t̄)))
    k = 0
    for (i, t) in enumerate(t̄)
        while k < length(signal.windows) && t ≥ signal.starttimes[k+1]
            k += 1
        end
        result[i] = signal.windows[k](t)
    end
    return result
end

function partial(
    i::Int,
    signal::Windowed{P,R},
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    isnothing(result) && return partial(i, signal, t̄; result=Vector{R}(undef, size(t̄)))
    result .= 0

    ki = get_window_from_parameter(signal,i)
    kt = 0
    for (j, t) in enumerate(t̄)
        while kt < length(signal.windows) && t ≥ signal.starttimes[kt+1]
            kt += 1
        end

        if      ki > kt; continue
        elseif  ki < kt; break
        else
            result[j] = partial(i-signal.offsets[ki], signal.windows[kt], t)
        end
    end
    return result
end

function integrate_partials(
    signal::Windowed{P,R},
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractVector,;
    result=nothing,
) where {P,R}
    isnothing(result) && return integrate_gradient_signal!(
        signal, τ̄, t̄, ϕα, ϕβ;
        result=Vector{P}(undef, Parameters.count(signal))
    )
    result .= 0

    k = 0                                                   # k INDEXES WINDOW
    for (j, t) in enumerate(t̄)                              # j INDEXES TIME
        while k < length(signal.windows) && t ≥ signal.starttimes[k+1]
            k += 1
        end

        for i in 1:Parameters.count(signal.windows[k])     # i INDEXES PARAMETER
            ∂ = partial(i, signal.windows[k], t)
            result[i+signal.offsets[k]] += τ̄[j] * real(∂ * ϕ̄[j])
        end
    end

    return result
end


#= TODO: Annotate return types for Composite, Modulated, and Windowed.

=#
