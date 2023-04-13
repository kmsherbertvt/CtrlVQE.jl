import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

abstract type AbstractCostFunction end
(::CostFunction)(x̄::AbstractVector)::Real = error("Not Implemented")

abstract type AbstractGradientFunction end
function (::GradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)::AbstractVector
    return error("Not Implemented")
end

(g::GradientFunction)(x̄::AbstractVector) = g(copy(x̄),x̄)

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
CompositeCostFunction(f̄...) = CompositeCostFunction(collect(f̄))
(f::CompositeCostFunction)(x̄::AbstractVector) = sum(f_(x̄) for f_ in f.f̄)

struct CompositeGradientFunction <: AbstractCostFunction
    ḡ::Vector{ArbitraryGradientFunction}
end
CompositeGradientFunction(ḡ...) = CompositeGradientFunction(collect(ḡ))
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