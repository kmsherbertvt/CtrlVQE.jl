import ..AbstractCostFunction, ..AbstractGradientFunction

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
    ∇f̄ .= 2 .* λ̄ .* χ̄
    return ∇f̄
end