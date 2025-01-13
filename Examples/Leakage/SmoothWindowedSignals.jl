module SmoothWindowedSignals
    import CtrlVQE
    import CtrlVQE: Parameters
    import CtrlVQE: Integrations, Signals

    import TemporaryArrays: @temparray

    # SMOOTH KERNEL
    f(x) = x ≤ -1 ? zero(x) : exp(-2/(1+x))
    # TRANSITION FUNCTION
    g(x) = x ≤ -1 ? zero(x) : x ≥ 1 ? one(x) : f(x) / ( f(x) + f(-x) )
    # DERIVATIVE OF TRANSITION FUNCTION
    h(x) = x ≤ -1 ? zero(x) : x ≥ 1 ? zero(x) : 4*(1+x^2)/(1-x^2)^2 * g(x) * g(-x)

    struct SmoothWindowedSignal{P,R,S<:Signals.SignalType{P,R}} <: Signals.SignalType{P,R}
        windows::Vector{S}
        starttimes::Vector{P}
        σ::P                    # HALF A TRANSITION WIDTH
    end

    """
        parseindex(signal::SmoothWindowedSignal, i::Int)

    Identify the window index, given a parameter index.

    Returns a tuple (w,k),
        where `w` is the window index and `k` is the index within the window.
    If k==0, `w` indexes `signal.starttimes` rather than `signal.windows`.

    """
    function parseindex(signal::SmoothWindowedSignal, i::Int)
        W = length(signal.windows)
        for w in 1:W
            L = Parameters.count(signal.windows[w])
            i ≤ L && return (w, i)
            i -= L
        end
        # i MUST INDEX A STARTTIME
        return (i+1, 0)  # Add one because starttimes[1] is not variational.
    end

    """
        isoverlapped(signal)

    Checks whether any windows in this signal are backwards.

    Backward windows result in opposite amplitudes (which may not be expected)
        and overlapping windows (which requires a slower vectorized loop).

    Note this function has no analog in `WindowedSignals`,
        because WindowedSignals are supposed to be ordered by definition.

    """
    function isordered(signal::SmoothWindowedSignal)
        W = length(signal.windows)
        for w in 1:W
            signal.starttimes[w+1] < signal.starttimes[w] && return false
        end
        return true
    end

    """
        isoverlapped(signal)

    Checks whether any window transitions overlap in this signal.

    Overlapping windows require a slower vectorized loop,
        so you may wish to optimize with a penalty term when starttimes get too close.

    Note this function has no analog in `WindowedSignals`,
        because WindowedSignals have no overlap by definition.

    """
    function isoverlapped(signal::SmoothWindowedSignal)
        W = length(signal.windows)
        σ = signal.σ
        for w in 1:W
            signal.starttimes[w+1] < signal.starttimes[w] + 2σ && return true
        end
        return false
    end

    #= `Parameters` INTERFACE =#

    function Parameters.count(signal::SmoothWindowedSignal)
        W = length(signal.windows)  # There are W-1 variable starttimes.
        return sum(Parameters.count(window)::Int for window in signal.windows) + W-1
    end

    function Parameters.names(signal::SmoothWindowedSignal)
        W = length(signal.windows)
        allnames = String[]
        for w in 1:W
            window = signal.windows[w]
            for name in Parameters.names(window)::Vector{String}
                push!(allnames, "$name.$w")
            end
        end
        for w in 2:W
            push!(allnames, "s$w")
        end
        return allnames
    end

    function Parameters.values(signal::SmoothWindowedSignal{P,R,S}) where {P,R,S}
        W = length(signal.windows)
        allvalues = P[]
        for window in signal.windows
            append!(allvalues, Parameters.values(window)::Vector{P})
        end
        append!(allvalues, @view(signal.starttimes[2:W]))
        return allvalues
    end

    function Parameters.bind!(
        signal::SmoothWindowedSignal{P,R,S},
        x::AbstractVector{P}
    ) where {P,R,S}
        W = length(signal.windows)
        Δ = 0   # PARAMETER OFFSET
        for w in 1:W
            L = Parameters.count(signal.windows[w])
            Parameters.bind!(signal.windows[w], @view(x[Δ+1:Δ+L]))
            Δ += L
        end
        signal.starttimes[2:W] .= @view(x[Δ+1:Δ+W-1])
        return signal
    end

    #= `Signals` INTERFACE =#

    function Signals.valueat(signal::SmoothWindowedSignal{P,R,S}, t::Real) where {P,R,S}
        W = length(signal.windows)

        u = @temparray(P, W+1, :valueat)
        u .= (t .- signal.starttimes) ./ signal.σ

        total = zero(R)
        for w in 1:W
            u[w]*u[w+1] > 0 && abs(u[w]) > 1 && abs(u[w+1]) > 1 && continue
            total += Signals.valueat(signal.windows[w], t) * (g(u[w]) - g(u[w+1]))
        end
        return total
    end

    function Signals.partial(
        i::Int,
        signal::SmoothWindowedSignal{P,R,S},
        t::Real,
    ) where {P,R,S}
        # WORK OUT WHICH WINDOW THIS `i` CORRESPONDS TO
        w, k = parseindex(signal, i)
        u = (t - signal.starttimes[w]) / signal.σ

        # HANDLE CASE WHERE `i` CORRESPONDS TO A STARTTIME
        if iszero(k)
            total = zero(R)
            total -= Signals.valueat(signal.windows[w],   t)
            total += Signals.valueat(signal.windows[w-1], t)
            total *= h(u) / signal.σ
            return total
        end

        # HANDLE CASE WHERE `t` IS OUTSIDE THE INTERVAL CORRESPONDING TO `w`
        v = (t - signal.starttimes[w+1]) / signal.σ
        u*v > 0 && abs(u) > 1 && abs(v) > 1 && return zero(R)

        # DELEGATE `partial` TO THE APPROPRIATE WINDOW
        return (g(u) - g(v)) * Signals.partial(k, signal.windows[w], t)
    end

    function Base.string(signal::SmoothWindowedSignal, names::AbstractVector{String})
        W = length(signal.windows)

        windowstrings = String[]
        Δ = 0               # INDEX OFFSETS
        for w in 1:W
            L = Parameters.count(signal.windows[w])
            push!(windowstrings, string(signal.windows[w], @view(names[Δ+1:Δ+L])))
            Δ += L
        end

        s = [
            "$(first(signal.starttimes))";
            @view(names[Δ+1:Δ+W-1]);
            "$(last(signal.starttimes))";
        ]

        return join((
            "($(windowstrings[w])) | t∊($(s[w]),$(s[w+1]))±σ" for w in 1:W
        ), "; ")
    end

    #= `Signals` OVERRIDES

    More efficient implementations of functions over a full timegrid.

    =#

    function Signals.valueat(
        signal::SmoothWindowedSignal{P,R,S},
        grid::Integrations.IntegrationType;
        result=nothing,
    ) where {P,R,S}
        isnothing(result) && (result=Vector{R}(undef, length(grid)))
        W = length(signal.windows)

        # IF STARTTIMES OVERLAP, WE CAN DO NO BETTER THAN LOOP OVER THE SCALAR FORM
        if isoverlapped(signal)
            for j in eachindex(grid)
                t = Integrations.timeat(grid,j)
                result[1+j] = Signals.valueat(signal, t)
            end
            return result
        end

        # WITH NO OVERLAP, EACH TIME IS EITHER A SINGLE WINDOW OR A TRANSITION BETWEEN TWO
        w = 1                       # INDEX THE TRANSITION
        for j in eachindex(grid)
            t = Integrations.timeat(grid, j)

            # FIND WHERE WE ARE WITH RESPECT TO THE NEXT (OR CURRENT) TRANSITION
            while w ≤ W && t ≥ signal.starttimes[w] + signal.σ
                w += 1
            end
            u = (t - signal.starttimes[w]) / signal.σ

            # HANDLE CASE WHERE WE ARE IN A SINGLE WINDOW
            if u < -1
                result[1+j] = Signals.valueat(signal.windows[w-1], t)
                continue
            end

            # HANDLE CASE WHERE WE ARE IN A TRANSITION BETWEEN TWO WINDOWS
            result[1+j] = 0
            (w ≤ W) && (result[1+j] += Signals.valueat(signal.windows[w],   t) * g(u))
            (w > 1) && (result[1+j] += Signals.valueat(signal.windows[w-1], t) * g(-u))
        end
        return result
    end

    function Signals.partial(
        i::Int,
        signal::SmoothWindowedSignal{P,R,S},
        grid::Integrations.IntegrationType;
        result=nothing,
    ) where {P,R,S}
        isnothing(result) && (result=Vector{R}(undef, length(grid)))

        w, k = parseindex(signal, i)

        # HANDLE CASE WHERE `i` CORRESPONDS TO A STARTTIME
        if iszero(k)
            result .= 0
            for j in eachindex(grid)
                t = Integrations.timeat(grid, j)
                u = (t - signal.starttimes[w]) / signal.σ
                # SKIP TIME IF IT IS OUTSIDE THE TARGET INTERVAL
                u < -1 && continue
                u >  1 && break
                # COMPUTE THE PARTIAL
                total = zero(R)
                total -= Signals.valueat(signal.windows[w],   t)
                total += Signals.valueat(signal.windows[w-1], t)
                total *= h(u) / signal.σ
                result[1+j] = total
            end
            return result
        end

        # HANDLE CASE WHERE `i` CORRESPONDS TO A SIGNAL PARAMETER:

        # IF STARTTIMES AREN'T ORDERED, WE CAN DO NO BETTER THAN LOOP OVER THE SCALAR FORM
        # (Well, we've already done better by calling `parseindex` only once for all t...)
        if !isordered(signal)
            for j in eachindex(grid)
                t = Integrations.timeat(grid, j)
                result[1+j] = Signals.partial(k, signal.windows[w], t)
            end
            return result
        end

        # IF STARTTIMES ARE ORDERED, WE CAN CUT OUT SOME WORK FOR `t` OUTSIDE THE INTERVAL
        result .= 0
        for j in eachindex(grid)
            t = Integrations.timeat(grid, j)
            u = (t - signal.starttimes[w])   / signal.σ;    u < -1 && continue
            v = (t - signal.starttimes[w+1]) / signal.σ;    v >  1 && break
            result[1+j] = (g(u) - g(v)) * Signals.partial(k, signal.windows[w], t)
        end
        return result
    end

    #= RECIPES =#

    """
        SmoothWindowed(template, T, W, σ)

    Convenience constructor to segment a single signal into a `SmoothWindowedSignal`.

    Makes `W` uniformly spaced windows up to a maximum time `T`.

    The actual start and end times are offset by `σ`, so that ``Ω(0)=Ω(T)=0``.
    This means the breadth of each window will be slightly less than when using `Windowed`.

    """
    function SmoothWindowed(
        template::Signals.SignalType{P,R},
        T::P,
        W::Int,
        σ::P,
    ) where {P,R}
        windows = [deepcopy(template) for _ in 1:W]
        starttimes = collect(range(σ, T-σ, W+1))
        return SmoothWindowedSignal(windows, starttimes, σ)
    end

end