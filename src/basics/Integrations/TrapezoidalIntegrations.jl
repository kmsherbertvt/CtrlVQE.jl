module TrapezoidalIntegrations
    export TrapezoidalIntegration, TemporalLattice

    import ..CtrlVQE: Integrations

    """
        TrapezoidalIntegration(t0::F, tf::F, r::Int)

    First-order grid using a trapezoidal rule time integration.

    # Arguments
    - `t0`, `tf`: lower and upper bounds of the integral
    - `r`: the number of time steps (remember number of lattice points is `r+1`)

    # Explanation

    This grid gives a first-order quadrature,
        meaning every step takes you forward in time.

    Additionally, this grid uses uniform spacing.
    You'd *think* that would mean `stepat(grid, i)` would give ``τ_i = τ`` for every `i`,
        where ``τ ≡ T / r`` and ``T ≡ t_f - t_0``.
    But careful! The sum of all `τ_i` must match the length of the integral, ie. `T`.
    But there are `r+1` points, and `(r+1)⋅τ > T`. How do we reconcile this?
    A "Left Hand Sum" would omit `t=T` from the time points;
        a "Right Hand Sum" would omit `t=0`.
    The trapezoidal rule omits half a point from each.

    That sounds awfully strange, but it's mathematically sound!
    We only integrate through *half* of each boundary time point `t=0` and `t=T`.
    Thus, those points, and only those points, have a spacing of `τ/2`.

    """
    struct TrapezoidalIntegration{F} <: Integrations.IntegrationType{F}
        t0::F
        tf::F
        r::Int
    end

    Integrations.starttime(grid::TrapezoidalIntegration) = grid.t0
    Integrations.endtime(grid::TrapezoidalIntegration) = grid.tf
    Integrations.nsteps(grid::TrapezoidalIntegration) = grid.r

    function Integrations.timeat(grid::TrapezoidalIntegration, i::Int)
        return grid.t0 + (Integrations.stepsize(grid) * i)
    end

    function Integrations.stepat(grid::TrapezoidalIntegration, i::Int)
        return Integrations.stepsize(grid) / ((0 < i < grid.r) ? 1 : 2)
    end

    """
        Prototype(::Type{TrapezoidalIntegration{F}}, r::Int; T=10.0)

    A prototypical time grid from t=0.0 to `T`.

    """
    function Integrations.Prototype(
        ::Type{TrapezoidalIntegration{F}}, r::Int;
        T=10.0,
    ) where {F}
        return TrapezoidalIntegration(zero(F), F(T), r)
    end

    """
        TemporalLattice(T::Real, r::Int)

    Convenience constructor to make a trapezoidal time grid from t=0.0 to `T`.

    ```jldoctests
    julia> grid = TemporalLattice(20.0, 400);

    julia> validate(grid);

    julia> Integrations.duration(grid)
    20.0
    julia> Integrations.nsteps(grid)
    400
    ```

    """
    function TemporalLattice(T::Real, r::Int)
        return TrapezoidalIntegration(zero(T), T, r)
    end

end