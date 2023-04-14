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
end

function CompositeCostFunction(f̄...)
    return CompositeCostFunction([ArbitraryCostFunction(f) for f in f̄])
end

(f::CompositeCostFunction)(x̄::AbstractVector) = sum(f_(x̄) for f_ in f.f̄)

struct CompositeGradientFunction <: AbstractGradientFunction
    ḡ::Vector{ArbitraryGradientFunction}
end

function CompositeGradientFunction(ḡ...)
    return CompositeGradientFunction([ArbitraryGradientFunction(g) for g in ḡ])
end

function (g::CompositeGradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)
    ∇f̄_ = array(eltype(x̄), size(x̄), LABEL)
    ∇f̄ .= 0
    for g_ in g.ḡ
        ∇f̄_ = g_(∇f̄_, x̄)
        ∇f̄ .+= ∇f̄_
    end
    return ∇f̄
end

#= TODO (lo):

Alternate composite cost/gradient function with io and delimiter.
Writes each of components' evaluations as a row in io.

=#