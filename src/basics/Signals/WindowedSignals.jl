module WindowedSignals
    export WindowedSignal, Windowed

    import ..CtrlVQE: Parameters
    import ..CtrlVQE: Integrations, Signals


    #= TODO:
        I've determined offsets are not robust to adaptive methods,
            nor are they necessary for static methods.
        They can be computed in negligible time at the beginning of a vectorized call,
            so they're only useful for non-vectorized calls,
            which are never used in high-performance code.
        Instead, use something like `parseindex` from `SmoothWindowedSignals`. =#

    #= TODO:
        It seems you can cut out those dumb while loops in the vectorized calls,
            while you're at it...
        Model after solution in `SmoothWindowedSignals`
            (but windows here are disjoint so the logic is even simpler). =#

    #= TODO:
        I've also determined that all the typecasts should no longer be needed,
            since windows are constrained to be a consistent type now.
        (I mean you can in principle manually construct a WindowedSignal with mixed windows,
            but there's no reason for me to design compiler-friendly code
            against such a brazen abuse of syntax. :) ) =#

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

    Windows with dynamically changing numbers of parameters are unsupported.

    """
    struct WindowedSignal{P,R,S<:Signals.SignalType{P,R}} <: Signals.SignalType{P,R}
        windows::Vector{S}
        starttimes::Vector{P}

        offsets::Vector{Int}        # CONTAINS A CUMULATIVE SUM OF PARAMETER COUNTS

        function WindowedSignal(
            windows::AbstractVector{<:Signals.SignalType{P,R}},
            starttimes::AbstractVector{<:Real},
        ) where {P,R}
            # CHECK THAT NUMBER OF WINDOWS AND STARTTIMES ARE COMPATIBLE
            if length(windows) != length(starttimes)
                error("Number of windows must match number of starttimes.")
            end

            # CONVERT WINDOWS TO A CONCRETE VECTOR
            S = eltype(windows)
            windows = convert(Vector{S}, windows)

            # ENSURE THAT starttimes ARE SORTED, AND MAKE TYPE CONSISTENT WITH WINDOWS
            starttimes = convert(Vector{P}, sort(starttimes))

            # CONSTRUCT `offsets` VECTOR
            offsets = Int[0]
            for (i, window) in enumerate(windows[1:end-1])
                push!(offsets, Parameters.count(window) + offsets[i])
            end

            return new{P,R,S}(windows, starttimes, offsets)
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

    function Parameters.values(signal::WindowedSignal{P,R,S}) where {P,R,S}
        allvalues = P[]
        for window in signal.windows
            append!(allvalues, Parameters.values(window)::Vector{P})
        end
        return allvalues
    end

    function Parameters.bind!(
        signal::WindowedSignal{P,R,S},
        x̄::AbstractVector{P}
    ) where {P,R,S}
        for (k, window) in enumerate(signal.windows)
            L = Parameters.count(window)::Int
            Parameters.bind!(window, x̄[1+signal.offsets[k]:L+signal.offsets[k]])
        end
        return signal
    end

    #= `Signals` INTERFACE =#

    function Signals.valueat(signal::WindowedSignal{P,R,S}, t::Real) where {P,R,S}
        k = get_window_from_time(signal,t)
        return Signals.valueat(signal.windows[k], t)::R
    end

    function Signals.partial(i::Int, signal::WindowedSignal{P,R,S}, t::Real) where {P,R,S}
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
        signal::WindowedSignal{P,R,S},
        grid::Integrations.IntegrationType;
        result=nothing,
    ) where {P,R,S}
        isnothing(result) && (result=Vector{R}(undef, length(grid)))

        k = 0
        for i in eachindex(grid)
            t = Integrations.timeat(grid, i)
            while k < length(signal.windows) && t ≥ signal.starttimes[k+1]
                k += 1
            end
            result[1+i] = Signals.valueat(signal.windows[k], t)::R
        end
        return result
    end

    function Signals.partial(
        i::Int,
        signal::WindowedSignal{P,R,S},
        grid::Integrations.IntegrationType;
        result=nothing,
    ) where {P,R,S}
        isnothing(result) && (result=Vector{R}(undef, length(grid)))
        result .= 0

        ki = get_window_from_parameter(signal,i)
        kt = 0
        for j in eachindex(grid)
            t = Integrations.timeat(grid, j)
            while kt < length(signal.windows) && t ≥ signal.starttimes[kt+1]
                kt += 1
            end

            if      ki > kt; continue
            elseif  ki < kt; break
            else
                result[1+j] = Signals.partial(
                    i-signal.offsets[ki],
                    signal.windows[kt],
                    t,
                )::R
            end
        end
        return result
    end

    """
        Windowed(signal, starttimes)
        Windowed(signal, T, W)

    Convenience constructors to segment a single `signal` into a `WindowedSignal`.

    Feed in `starttimes` directly,
        or make `W` uniformly spaced windows up to maximum time `T`.

    """
    function Windowed end

    function Windowed(signal::Signals.SignalType, starttimes::AbstractVector)
        windows = [deepcopy(signal) for _ in starttimes]
        return WindowedSignal(windows, starttimes)
    end

    function Windowed(signal::Signals.SignalType, T::Real, W::Int)
        windows = [deepcopy(signal) for _ in 1:W]
        starttimes = range(0.0, T, W+1)[begin:end-1]
        return WindowedSignal(windows, starttimes)
    end

end