module WindowOverlapPenalties
    import CtrlVQE: Parameters
    import CtrlVQE: Integrations, Signals, CostFunctions

    import CtrlVQE.Integrations: IntegrationType
    import CtrlVQE.Signals: SignalType
    import CtrlVQE.CostFunctions: CostFunctionType

    # NOTE: Implicitly use smooth bounding function.
    wall(u) = u ≤ 0 ? zero(u) : exp(u - 1/u)
    grad(u) = u ≤ 0 ? zero(u) : exp(u - 1/u) * (1 + 1/u^2)

    """
        WindowOverlapPenalty(grid, signal, A, σ, λ)

    Compute penalties for a signal's windows getting too close together.

    This penalty is designed for `SmoothWindowedSignal`.
    It requires the `starttimes` field with length `W>1`,
        assumes that the first and last of these are non-variational,
        and that the last `W-1` parameters of `x`
        correspond directly to the remaining starttimes.

    Let ``u(t)=\\frac{|Ω(t)|-A}{σ}``, where `Ω(t)` is defined by `signal`.
    When ``u>0`` the instantaneous penalty is computed as ``Λ(t) = λ⋅\\exp[u(t) - 1/u(t)]``.
    The total penalty is ``\\frac{1}{T}⋅\\int_0^T Λ(t)⋅dt``,
        where the integration is defined by `grid`.

    # Parameters
    - `grid`: defines the integration (including time bounds) over which to penalize.
    - `signal`: the signal to penalize.
    - `A`: the maximum modulus of the signal.
    - `σ`: the steepness parameter of the penalty function.
    - `λ`: the strength parameter of the penalty function.

    Note that, if you do not wish the penalty to be normalized per unit time,
        `λ` should be selected to be proportional to the duration of `grid`.

    """
    struct WindowOverlapPenalty{
        F,
        R,
        S<:SignalType{F,R},
    } <: CostFunctionType{F}
        signal::S       # SIGNAL TYPE
        A::F            # MAXIMUM MODULUS
        σ::F            # STEEPNESS OF PENALTY
        λ::F            # STRENGTH OF PENALTY PER UNIT TIME
    end

    """
        WindowOverlapPenalty(device; A=1.0, σ=A, λ=1.0)

    Alternate constructor, with kwargs and sensible defaults.

    We have selected σ defaulting to A because heuristically it seems to work well.
    Don't feel too attached to that choice.

    """
    function WindowOverlapPenalty(
        signal::SignalType{F,R};
        A::Real=one(F),
        σ::Real=A,
        λ::Real=one(F),
    ) where {F,R}
        return WindowOverlapPenalty(signal, F(A), F(σ), F(λ))
    end

    function CostFunctions.cost_function(costfn::WindowOverlapPenalty)
        W = length(costfn.signal.starttimes) - 1
        s = costfn.signal.starttimes

        return (x) -> (
            Parameters.bind!(costfn.signal, x);
            total = zero(eltype(s));
            for w in 1:W;
                u = (costfn.A - (s[w+1] - s[w])) / costfn.σ;
                total += wall(u);
            end;
            costfn.λ * total
        )
    end

    function CostFunctions.grad!function(costfn::WindowOverlapPenalty)
        W = length(costfn.signal.starttimes) - 1
        s = costfn.signal.starttimes

        return (∇f, x) -> (
            Parameters.bind!(costfn.signal, x);
            ∇f .= 0;
            for w in 2:W;
                u = (costfn.A - (s[w+1] - s[w])) / costfn.σ;
                v = (costfn.A - (s[w] - s[w-1])) / costfn.σ;
                ∇f[end-W+w] = costfn.λ / costfn.σ * (grad(u) - grad(v));
            end;
            ∇f
        )
    end

    function Base.length(costfn::WindowOverlapPenalty)
        return Parameters.count(costfn.signal)
    end
end