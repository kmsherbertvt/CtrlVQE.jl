import LinearAlgebra: norm

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

"""
    AbstractCostFunction

Super-type for "cost functions", to be plugged directly into optimization algorithms.

# Implementation

Any concrete sub-type `CF` must be a callable object, accepting a vector of parameters.
That is, it must implement the function `(f::CF)(x̄::AbstractVector)`.

"""
abstract type AbstractCostFunction end

"""
    (f::AbstractCostFunction)(x̄::AbstractVector)

A cost function evaluated at the point in parameter space given by `x̄`.

"""
function (::AbstractCostFunction)(x̄::AbstractVector)
    error("Not Implemented")
    return 0
end

"""
    AbstractGradientFunction

Super-type for "gradient functions", to be plugged directly into optimization algorithms.

Each gradient function is implicitly associated with a cost function.

# Implementation

Any concrete sub-type `GF` must be a callable object,
    accepting a vector of partial derivatives (to be calculated and filled in),
    and a vector of parameters.
That is, it must implement the function `(g::GF)(∇f̄::AbstractVector,x̄::AbstractVector)`.

"""
abstract type AbstractGradientFunction end

"""
    (g::AbstractGradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)

The gradient of a function at the point in parameter space given by `x̄`.

The result is stored in the first argument `∇f̄`, as well as returned.

"""
function (::AbstractGradientFunction)(
    ∇f̄::AbstractVector,
    x̄::AbstractVector,
)::AbstractVector
    error("Not Implemented")
    return ∇f̄
end

"""
    (g::AbstractGradientFunction)(x̄::AbstractVector)

The gradient of a function at the point in parameter space given by `x̄`.

"""
(g::AbstractGradientFunction)(x̄::AbstractVector) = g(copy(x̄),x̄)

"""
    CompositeCostFunction(f̄::Vector{AbstractCostFunction})

The sum of several other cost-functions.

Use this eg. to combine an energy function with one or more penalty functions.

"""
struct CompositeCostFunction <: AbstractCostFunction
    f̄::Vector{AbstractCostFunction}

    counter::Ref{Int}
    values::Vector{Float64}

    function CompositeCostFunction(f̄::Vector{AbstractCostFunction})
        return new(f̄, 0, Vector{Float64}(undef, size(f̄)))
    end
end

"""
    CompositeCostFunction(f̄...)

Alternate constructor, letting each function be passed as its own argument.

"""
function CompositeCostFunction(f̄...)
    return CompositeCostFunction(AbstractCostFunction[f for f in f̄])
end

function (f::CompositeCostFunction)(x̄::AbstractVector)
    f.counter[] += 1
    f.values .= [f_(x̄) for f_ in f.f̄]
    return sum(f.values)
end





"""
    CompositeGradientFunction(ḡ::Vector{AbstractCostFunction})

The sum of several other gradient-functions.

"""
struct CompositeGradientFunction <: AbstractGradientFunction
    ḡ::Vector{AbstractGradientFunction}

    counter::Ref{Int}
    norms::Vector{Float64}

    function CompositeGradientFunction(ḡ::Vector{AbstractGradientFunction})
        return new(ḡ, 0, Vector{Float64}(undef, size(ḡ)))
    end
end


"""
    CompositeGradientFunction(ḡ...)

Alternate constructor, letting each function be passed as its own argument.

"""
function CompositeGradientFunction(ḡ...)
    return CompositeGradientFunction(AbstractGradientFunction[g for g in ḡ])
end

function (g::CompositeGradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)
    g.counter[] += 1
    ∇f̄_ = array(eltype(x̄), size(x̄), LABEL)
    ∇f̄ .= 0
    for (i, g_) in enumerate(g.ḡ)
        ∇f̄_ = g_(∇f̄_, x̄)
        g.norms[i] = norm(∇f̄_)
        ∇f̄ .+= ∇f̄_
    end
    return ∇f̄
end
