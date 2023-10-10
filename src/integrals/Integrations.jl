export IntegrationType
export nsteps, timeat, stepat, lattice, integrate
export starttime, endtime, duration, stepsize

using Memoization: @memoize

"""
    IntegrationType{F}

Super-type for all time grids, which encapsulate everything needed to integrate over time.

# Type Parameters
- `F` denotes the type for time values. Must be a real float.

# Implementation

Any concrete sub-type `G` must implement the following methods:
- `nsteps(::G)`: the total number of finite jumps within this time grid.

    This is the maximum index of the `timeat` and `stepat` functions,
        but it is NOT the length of the lattice, since the minimum index is 0.

- `timeat(::G, i::Int)`: time at index i (which may be zero)
- `stepat(::G, i::Int)`: stepsize at index i (which may be zero)

    Let ``r`` be the number of steps, ``T`` be the duration,
        and ``τ_i`` the step at index ``i``.
    Then ``∑_{i=0}^r τ_i = T``.

- `Base.reverse(::G)`: a new integration, identical except bounds are reversed

"""
abstract type IntegrationType{F<:AbstractFloat} end


"""
    nsteps(grid::IntegrationType)

The total number of finite jumps within this time grid.

This is the maximum index of the `timeat` and `stepat` functions,
    but it is NOT the length of the lattice, since the minimum index is 0.

"""
function nsteps(::IntegrationType)
    error("Not Implemented")
    return 0
end

"""
    timeat(grid::IntegrationType, i::Int)

The time at index i (which may be zero).

"""
function timeat(::IntegrationType{F}, i::Int) where {F}
    error("Not Implemented")
    return zero(F)
end

"""
    stepat(grid::IntegrationType, i::Int)

The stepsize at index i (which may be zero).

Let ``r`` be the number of steps, ``T`` be the duration,
    and ``τ_i`` the step at index ``i``.
Then ``∑_{i=0}^r τ_i = T``.

"""
function stepat(::IntegrationType{F}, i::Int) where {F}
    error("Not Implemented")
    return zero(F)
end


"""
    Base.reverse(grid::IntegrationType)

Construct a new integration, identical except bounds are reversed.
"""
function Base.reverse(grid::IntegrationType)
    error("Not Implemented")
    return grid
end

"""
    starttime(grid::IntegrationType)

Lower bound of time integral.

"""
function starttime(grid::IntegrationType)
    return timeat(grid, 0)
end

"""
    endtime(grid::IntegrationType)

Upper bound of time integral.

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
    lattice(grid::IntegrationType)

A vector of time points.

This is a regular vector, meaning it indexes from 1.
This is offset by one from the `timeat` and `stepat` functions!

NOTE: This function is memoized, so DO NOT mutate its return value!

"""
@memoize Dict function lattice(grid::IntegrationType)
    t̄ = Vector{eltype(grid)}(undef, 1+nsteps(grid))
    for i in 0:nsteps(grid); t̄[1+i] = timeat(grid, i); end
    return t̄
end


"""
    integrate(grid::IntegrationType{F}, Φ)

Compute the integral ``∫_0^T Φ(t)⋅dt``.
`Φ` is a univariate scalar function of time, returning type F.

    integrate(grid::IntegrationType, f̄::AbstractVector)

Treat the elements of `f̄` as the function evaluations `Φ(t)` above.
The length of `f̄` must be `nsteps(grid) + 1` and its eltype must be real.

    integrate(grid::IntegrationType, Φ, f̄s::AbstractVector...)

Compute the integral ``∫_0^T Φ(t, f1, f2...)⋅dt``,
    where `f1` is the value of `f̄s[1]` at the index corresponding to time `t`, etc.
`Φ` is a multivariate scalar function,
    of time and one argument for each f̄, returning type F.
The length of each element in `f̄s` must be `nsteps(grid) + 1`.

"""
function integrate end

function integrate(grid::IntegrationType, f̄::AbstractVector)
    I = zero(promote_type(eltype(grid), eltype(f̄)))
    for i in 0:nsteps(grid)
        I += f̄[i+1] * stepat(grid, i)
    end
    return I
end

function integrate(grid::IntegrationType, Φ, f̄s::AbstractVector...)
    I = zero(eltype(grid))
    for i in 0:nsteps(grid)
        I += Φ(timeat(grid,i), (f̄[i+1] for f̄ in f̄s)...) * stepat(grid, i)
    end
    return I
end


#= TODO (mid): Implement the whole AbstractVector interface.

https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array

Importantly, firstindex probably needs to be overwritten..?
Not really sure how that works.

=#

Base.eltype(::IntegrationType{F}) where {F} = F