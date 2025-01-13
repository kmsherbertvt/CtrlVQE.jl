"""
    IntegrationType{F}

Encapsulates a time-grid, used to decide how to integrate over time.

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

- `Prototype(::Type{G}, r::Int; T, kwargs...)`:
    construct a prototypical grid of type `G` with `r` steps.

# `AbstractVector` Interface

This type implements the AbstractVector interface,
    defined so that `grid[i] == timeat(grid, i)`, where `grid` is the IntegrationType.
Note that `i` here starts from 0.
The `collect` function produces a concrete vector, in which `i` starts from 1.
The `lattice` function does the same thing, but permits an allocation-free signature.

"""
abstract type IntegrationType{F<:AbstractFloat} <: AbstractVector{F} end


"""
    nsteps(grid::IntegrationType)::Int

The total number of finite jumps within this time grid.

This is the maximum index of the `timeat` and `stepat` functions,
    but it is NOT the length of the lattice, since the minimum index is 0.

"""
function nsteps end

"""
    timeat(grid::IntegrationType{F}, i::Int)::F

The time at index i (which may be zero).

"""
function timeat end

"""
    stepat(grid::IntegrationType{F}, i::Int)::F

The stepsize at index i (which may be zero).

Let ``r`` be the number of steps, ``T`` be the duration,
    and ``τ_i`` the step at index ``i``.
Then ``∑_{i=0}^r τ_i = T``.

"""
function stepat end

"""
    Prototype(gridtype::Type{G}, r::Int; T, kwargs...)

Construct a prototypical grid of type `G` with `r` steps.

The grid should integrate from 0.0 to `T` (which may be given a default value).

# Implementation

The primary purpose of this constructor is to produce time grids
    with arbitrary numbers of steps
    for testing and benchmarking purposes.
Therefore, the constructor should use reasonable values for all unspecified fields,
    but they need not necessarily be scientifically meaningful.

Of course, the more meaningful they are,
    the more useful this constructor can be.
Therefore, the method signature accommodates keyword arguments
    which can be tailored to the specific device.
But standard tests and benchmarks are agnostic to those specifics,
    so these kwargs must have reasonable defaults!

"""
function Prototype end