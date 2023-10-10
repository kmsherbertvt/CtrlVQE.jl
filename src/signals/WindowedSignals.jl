import ..Parameters, ..Signals
export WindowedSignal

import ..Signals: SignalType


"""
    WindowedSignal(windows, starttimes)

A signal which applies a different function for each window.

# Arguments
- windows: a vector of signals
- starttimes: a vector of times transitioning each window

Both `windows` and `starttimes` have the same length;
    `starttimes[i]` indicates when `windows[i]` begins.

This signal is undefined for times `t < starttimes[1]`.
Normally, `starttimes[1] == 0`.

Note that each window must share the same type parameters `P` and `R`.

"""
struct WindowedSignal{P,R} <: SignalType{P,R}
    windows::Vector{SignalType{P,R}}
    starttimes::Vector{P}

    offsets::Vector{Int}        # CONTAINS A CUMULATIVE SUM OF PARAMETER COUNTS

    function WindowedSignal(
        windows::AbstractVector{<:SignalType{P,R}},
        starttimes::AbstractVector{<:Real},
    ) where {P,R}
        # CHECK THAT NUMBER OF WINDOWS AND STARTTIMES ARE COMPATIBLE
        if length(windows) != length(starttimes)
            error("Number of windows must match number of starttimes.")
        end

        # CONVERT windows TO VECTOR OF ABSTRACT TYPE
        windows = convert(Vector{SignalType{P,R}}, windows)

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

"""
    get_window_from_time(signal::WindowedSignal, t::Real)

Identify the window index given the time (by inspecting `starttimes`).

"""
function get_window_from_time(signal::WindowedSignal, t::Real)
    k = findlast(starttime -> starttime ≤ t, signal.starttimes)
    isnothing(k) && error("Time $t does not fit into any window.")
    return k
end

"""
    get_window_from_parameter(signal::WindowedSignal, i::Int)

Identify the window index given a parameter index (by counting parameters in `windows`).

"""
function get_window_from_parameter(signal::WindowedSignal, i::Int)
    k = findlast(offset -> offset < i, signal.offsets)
    isnothing(k) && error("Parameter $i does not fit into any window.")
    return k
end

#= `Parameters` INTERFACE =#

function Parameters.count(signal::WindowedSignal)
    return sum(Parameters.count(window)::Int for window in signal.windows)
end

function Parameters.names(signal::WindowedSignal)
    allnames = String[]
    for (i, window) in enumerate(signal.windows)
        for name in Parameters.names(window)::Vector{String}
            push!(allnames, "$name.$i")
        end
    end
    return allnames
end

function Parameters.values(signal::WindowedSignal{P,R}) where {P,R}
    allvalues = P[]
    for window in signal.windows
        append!(allvalues, Parameters.values(window)::Vector{P})
    end
    return allvalues
end

function Parameters.bind(signal::WindowedSignal{P,R}, x̄::AbstractVector{P}) where {P,R}
    for (k, window) in enumerate(signal.windows)
        L = Parameters.count(window)::Int
        Parameters.bind(window, x̄[1+signal.offsets[k]:L+signal.offsets[k]])
    end
end

#= `Signals` INTERFACE =#

function Signals.valueat(signal::WindowedSignal{P,R}, t::Real) where {P,R}
    k = get_window_from_time(signal,t)
    return Signals.valueat(signal.windows[k], t)::R
end

function Signals.partial(i::Int, signal::WindowedSignal{P,R}, t::Real) where {P,R}
    kt = get_window_from_time(signal,t)
    ki = get_window_from_parameter(signal,i)
    return (kt == ki ?  Signals.partial(i-signal.offsets[ki], signal.windows[kt], t)::R
        :               zero(R)
    )
end

function Base.string(signal::WindowedSignal, names::AbstractVector{String})
    texts = String[]
    for (k, window) in enumerate(signal.windows)
        L = Parameters.count(window)::Int
        text = string(window, names[1+signal.offsets[k]:L+signal.offsets[k]])

        s1 = signal.starttimes[k]
        s2 = k+1 > length(signal.starttimes) ? "∞" : signal.starttimes[k+1]
        push!(texts, "($text) | t∊[$s1,$s2)")
    end

    return join(texts, "; ")
end

#= `Signals` OVERRIDES

More efficient implementations of functions over a full timegrid.

=#

function Signals.valueat(
    signal::WindowedSignal{P,R},
    t̄::AbstractVector{<:Real};
    result=nothing,
) where {P,R}
    if isnothing(result)
        return Signals.valueat(signal, t̄; result=Vector{R}(undef, size(t̄)))
    end

    k = 0
    for (i, t) in enumerate(t̄)
        while k < length(signal.windows) && t ≥ signal.starttimes[k+1]
            k += 1
        end
        result[i] = Signals.valueat(signal.windows[k], t)::R
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
            result[j] = Signals.partial(i-signal.offsets[ki], signal.windows[kt], t)::R
        end
    end
    return result
end
