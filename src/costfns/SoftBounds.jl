import ..AbstractCostFunction, ..AbstractGradientFunction

"""
    functions(λ̄, x̄L, x̄R, σ̄)

A shallow quadratic penalty for each parameter's deviation from mean.

# Arguments
- `λ̄`: vector of weights for each penalty
        Set `λ̄[i]=0` to skip penalties for the ith parameter.

- `μ̄`: vector of lower bounds for each parameter
- `σ̄`: vector of scalings (smaller=steeper) for each penalty

# Returns
- `f`: the cost function
- `g`: the gradient function

"""
function functions(λ̄, μ̄, σ̄)
    f = CostFunction(λ̄, μ̄, σ̄)
    g = GradientFunction(f)
    return f, g
end

struct CostFunction{F<:AbstractFloat} <: AbstractCostFunction
    λ̄::Vector{F}
    μ̄::Vector{F}
    σ̄::Vector{F}

    function CostFunction(λ̄::AbstractVector, μ̄::AbstractVector, σ̄::AbstractVector)
        F = promote_type(Float16, eltype(λ̄), eltype(μ̄), eltype(σ̄))
        return new{F}(
            convert(Array{F}, λ̄),
            convert(Array{F}, μ̄),
            convert(Array{F}, σ̄),
        )
    end
end

function (f::CostFunction)(x̄::AbstractVector)
    λ̄ = f.λ̄
    χ̄ = (x̄ .- f.μ̄) ./ f.σ̄
    return sum(λ̄ .* χ̄.^2)
end

struct GradientFunction{F<:AbstractFloat} <: AbstractGradientFunction
    f::CostFunction{F}
end

function (g::GradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)
    λ̄ = g.f.λ̄
    χ̄ = (x̄ .- g.f.μ̄) ./ g.f.σ̄
    ∇f̄ .= 2 .* λ̄ .* χ̄ ./ g.f.σ̄
    return ∇f̄
end