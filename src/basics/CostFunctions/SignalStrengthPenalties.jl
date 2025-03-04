module SignalStrengthPenalties
    import CtrlVQE: Parameters
    import CtrlVQE: Integrations, Signals, CostFunctions

    import CtrlVQE.Integrations: IntegrationType
    import CtrlVQE.Signals: SignalType
    import CtrlVQE.CostFunctions: CostFunctionType

    # NOTE: Implicitly use smooth bounding function.
    wall(u) = exp(u - 1/u)
    grad(u) = exp(u - 1/u) * (1 + 1/u^2)

    """
        SignalStrengthPenalty(grid, signal, A, σ, λ)

    Compute penalties for a signal exceeding a maximum absolute value over a time interval.

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
    struct SignalStrengthPenalty{
        F,
        R,
        G<:IntegrationType{F},
        S<:SignalType{F,R},
    } <: CostFunctionType{F}
        grid::G         # INTEGRATION TYPE
        signal::S       # SIGNAL TYPE
        A::F            # MAXIMUM MODULUS
        σ::F            # STEEPNESS OF PENALTY
        λ::F            # STRENGTH OF PENALTY PER UNIT TIME
    end

    """
        SignalStrengthPenalty(device; A=1.0, σ=A, λ=1.0)

    Alternate constructor, with kwargs and sensible defaults.

    We have selected σ defaulting to A because heuristically it seems to work well.
    Don't feel too attached to that choice.

    ```jldoctests
    julia> grid = TemporalLattice(20.0, 400);

    julia> signal = Windowed(Constant(2.0), 20.0, 5);

    julia> costfn = SignalStrengthPenalty(grid, signal; A=0.8);

    julia> x = collect(range(0.0, 1.0, length(costfn)))
    5-element Vector{Float64}:
     0.0
     0.25
     0.5
     0.75
     1.0

    julia> validate(costfn; x=x, rms=1e-6);

    julia> costfn(x)
    0.00473294635352182
    julia> grad_function(costfn)(x)
    5-element Vector{Float64}:
     0.0
     0.0
     0.0
     0.0
     0.10057511001233899

    ```

    """
    function SignalStrengthPenalty(
        grid::IntegrationType{F},
        signal::SignalType{F,R};
        A::Real=one(F),
        σ::Real=A,
        λ::Real=one(F),
    ) where {F,R}
        return SignalStrengthPenalty(grid, signal, F(A), F(σ), F(λ))
    end

    function CostFunctions.cost_function(costfn::SignalStrengthPenalty)
        T = Integrations.duration(costfn.grid)
        R = Signals.returntype(costfn.signal)
        Ω = Vector{R}(undef, length(costfn.grid))   # TO FILL, FOR EACH DRIVE

        Φ(t, Ω) = (
            u = (abs(Ω) - costfn.A) / costfn.σ;
            u ≤ 0 ? zero(u) : wall(u)
        )

        return (x) -> (
            Parameters.bind!(costfn.signal, x);
            Signals.valueat(costfn.signal, costfn.grid; result=Ω);  # FILLS Ω
            costfn.λ * real(Integrations.integrate(costfn.grid, Φ, Ω)) / T
        )
    end

    function CostFunctions.grad!function(costfn::SignalStrengthPenalty)
        T = Integrations.duration(costfn.grid)
        R = Signals.returntype(costfn.signal)
        Ω = Vector{R}(undef, length(costfn.grid))   # TO FILL, FOR EACH DRIVE
        ∂ = Vector{R}(undef, length(costfn.grid))   # TO FILL, FOR EACH PARAMETER

        Φ(t, Ω, ∂) = (
            u = (abs(Ω) - costfn.A) / costfn.σ;
            u ≤ 0 ? zero(u) : grad(u) * real(conj(Ω)*∂) / (abs(Ω)*costfn.σ)
        )

        return (∇f, x) -> (
            Parameters.bind!(costfn.signal, x);
            ∇f .= 0;
            Ω = Signals.valueat(costfn.signal, costfn.grid; result=Ω);      # FILLS Ω
            for k in 1:Parameters.count(costfn.signal);
                Signals.partial(k, costfn.signal, costfn.grid; result=∂);   # FILLS ∂
                ∇f[k] = costfn.λ * real(Integrations.integrate(costfn.grid, Φ, Ω, ∂)) / T;
            end;
            ∇f
        )
    end

    function Base.length(costfn::SignalStrengthPenalty)
        return Parameters.count(costfn.signal)
    end
end