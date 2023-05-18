import ...Parameters, ...Signals
import ...Signals: AbstractSignal


#= WINDOWED SIGNAL =#

struct WindowedSignal{P,R} <: AbstractSignal{P,R}
    windows::Vector{AbstractSignal{P,R}}
    starttimes::Vector{P}

    offsets::Vector{Int}        # CONTAINS A CUMULATIVE SUM OF PARAMETER COUNTS

    function WindowedSignal(
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

function Parameters.count(signal::WindowedSignal)
    return sum(Parameters.count(window) for window in signal.windows)
end

function Parameters.names(signal::WindowedSignal)
    names(i) = ["$name.$i" for name in Parameters.names(signal.windows[i])]
    return vcat((names(i) for i in eachindex(signal.windows))...)
end

function Parameters.values(signal::WindowedSignal)
    return vcat((Parameters.values(window) for window in signal.windows)...)
end

function Parameters.bind(signal::WindowedSignal{P,R}, x̄::AbstractVector{P}) where {P,R}
    for (k, window) in enumerate(signal.windows)
        L = Parameters.count(window)
        Parameters.bind(window, x̄[1+signal.offsets[k]:L+signal.offsets[k]])
    end
end

function get_window_from_time(signal::WindowedSignal, t::Real)
    k = findlast(starttime -> starttime ≤ t, signal.starttimes)
    isnothing(k) && error("Time $t does not fit into any window.")
    return k
end

function get_window_from_parameter(signal::WindowedSignal, i::Int)
    k = findlast(offset -> offset < i, signal.offsets)
    isnothing(k) && error("Parameter $i does not fit into any window.")
    return k
end

function (signal::WindowedSignal)(t::Real)
    k = get_window_from_time(signal,t)
    return signal.windows[k](t)
end

function Signals.partial(i::Int, signal::WindowedSignal{P,R}, t::Real) where {P,R}
    kt = get_window_from_time(signal,t)
    ki = get_window_from_parameter(signal,i)
    return (kt == ki ?  Signals.partial(i-signal.offsets[ki], signal.windows[kt], t)
        :               zero(R)
    )
end

function Base.string(signal::WindowedSignal, names::AbstractVector{String})
    texts = String[]
    for (k, window) in enumerate(signal.windows)
        L = Parameters.count(window)
        text = string(window, names[1+signal.offsets[k]:L+signal.offsets[k]])

        s1 = signal.starttimes[k]
        s2 = k+1 > length(signal.starttimes) ? "∞" : signal.starttimes[k+1]
        push!(texts, "($text) | t∊[$s1,$s2)")
    end

    return join(texts, "; ")
end

# VECTORIZED METHODS
function (signal::WindowedSignal{P,R})(
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

function Signals.partial(
    i::Int,
    signal::WindowedSignal{P,R},
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    if isnothing(result)
        return Signals.partial(i, signal, t̄; result=Vector{R}(undef, size(t̄)))
    end
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
            result[j] = Signals.partial(i-signal.offsets[ki], signal.windows[kt], t)
        end
    end
    return result
end

function Signals.integrate_partials(
    signal::WindowedSignal{P,R},
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractVector,;
    result=nothing,
) where {P,R}
    isnothing(result) && return Signals.integrate_partials(
        signal, τ̄, t̄, ϕ̄;
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