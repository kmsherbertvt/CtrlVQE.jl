import LinearAlgebra: norm

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

abstract type AbstractCostFunction end
(::AbstractCostFunction)(x̄::AbstractVector)::Real = error("Not Implemented")

abstract type AbstractGradientFunction end
function (::AbstractGradientFunction)(
    ∇f̄::AbstractVector,
    x̄::AbstractVector,
)::AbstractVector
    return error("Not Implemented")
end

(g::AbstractGradientFunction)(x̄::AbstractVector) = g(copy(x̄),x̄)

#= ARBITRARY FUNCTIONS - hack providing concrete type for arbitrary functionality. =#

struct ArbitraryCostFunction <: AbstractCostFunction
    f::AbstractCostFunction
end
(f::ArbitraryCostFunction)(x̄::AbstractVector) = f.f(x̄)

struct ArbitraryGradientFunction <: AbstractGradientFunction
    g::AbstractGradientFunction
end
function (g::ArbitraryGradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)
    return g.g(∇f̄, x̄)
end

#= COMPOSITE FUNCTIONS - useful for combining energy with penalties =#

struct CompositeCostFunction <: AbstractCostFunction
    f̄::Vector{ArbitraryCostFunction}

    counter::Ref{Int}
    values::Vector{Float64}

    function CompositeCostFunction(f̄::Vector{ArbitraryCostFunction})
        return new(f̄, 0, Vector{Float64}(undef, size(f̄)))
    end
end

function CompositeCostFunction(f̄...)
    return CompositeCostFunction([ArbitraryCostFunction(f) for f in f̄])
end

function (f::CompositeCostFunction)(x̄::AbstractVector)
    f.counter[] += 1
    f.values .= [f_(x̄) for f_ in f.f̄]
    return sum(f.values)
end

struct CompositeGradientFunction <: AbstractGradientFunction
    ḡ::Vector{ArbitraryGradientFunction}

    counter::Ref{Int}
    norms::Vector{Float64}

    function CompositeGradientFunction(ḡ::Vector{ArbitraryGradientFunction})
        return new(ḡ, 0, Vector{Float64}(undef, size(ḡ)))
    end
end

function CompositeGradientFunction(ḡ...)
    return CompositeGradientFunction([ArbitraryGradientFunction(g) for g in ḡ])
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
