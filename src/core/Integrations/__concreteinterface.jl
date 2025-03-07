import .Integrations: IntegrationType
import .Integrations: nsteps, timeat, stepat

"""
    starttime(grid::IntegrationType)

Lower bound of a time integral.

"""
function starttime(grid::IntegrationType)
    return timeat(grid, 0)
end

"""
    endtime(grid::IntegrationType)

Upper bound of a time integral.

"""
function endtime(grid::IntegrationType)
    return timeat(grid, nsteps(grid))
end

"""
    duration(grid::IntegrationType)

The total duration, ie. the last time point minus the first.

"""
function duration(grid::IntegrationType)
    return endtime(grid) - starttime(grid)
end

"""
    stepsize(grid::IntegrationType)

The average step size, ie. duration divided by number of steps.

"""
function stepsize(grid::IntegrationType)
    return duration(grid) / nsteps(grid)
end

"""
    lattice(grid::IntegrationType; result=nothing)

A vector of all time points.

This is equivalent to `collect(grid)` except for the `result` kwarg,
    which allows the caller to provide a pre-allocated array.

"""
function lattice(grid::IntegrationType; result=nothing)
    isnothing(result) && (result = Vector{eltype(grid)}(undef, length(grid)))
    for i in eachindex(grid)
        result[1+i] = grid[i]
    end
    return result
end

"""
    integrate(grid::IntegrationType{F}, Φ)

Compute the integral ``∫_0^T Φ(t)⋅dt``.
`Φ` is a univariate scalar function of time, returning type F.

    integrate(grid::IntegrationType, f̄::AbstractVector)

Treat the elements of `f̄` as the function evaluations `Φ(t)` above.
The length of `f̄` must be `length(grid)` and its eltype must be real.

    integrate(grid::IntegrationType, Φ, f̄s::AbstractVector...)

Compute the integral ``∫_0^T Φ(t, f1, f2...)⋅dt``,
    where `f1` is the value of `f̄s[1]` at the index corresponding to time `t`, etc.
`Φ` is a multivariate scalar function,
    of time and one argument for each f̄, returning type F.
The length of each element in `f̄s` must be `length(grid)`.

"""
function integrate end

function integrate(grid::IntegrationType, f̄::AbstractVector)
    I = zero(promote_type(eltype(grid), eltype(f̄)))
    for i in eachindex(grid)
        I += f̄[1+i] * stepat(grid, i)
    end
    return I
end

function integrate(grid::IntegrationType, Φ)
    I = zero(eltype(grid))
    for i in eachindex(grid)
        I += Φ(timeat(grid,i)) * stepat(grid, i)
    end
    return I
end

function integrate(grid::IntegrationType, Φ, f̄s::AbstractVector...)
    I = zero(promote_type(eltype(grid), Base.promote_eltype(f̄s...)))
    for i in eachindex(grid)
        I += Φ(timeat(grid,i), (f̄[1+i] for f̄ in f̄s)...) * stepat(grid, i)
    end
    return I
end

# `AbstractVector` INTERFACE, customised to index from 0.
Base.size(grid::IntegrationType) = (1+nsteps(grid),)
Base.getindex(grid::IntegrationType, i::Int) = timeat(grid,i)
Base.axes(grid::IntegrationType) = (0:nsteps(grid),)