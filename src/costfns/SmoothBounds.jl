import ..AbstractCostFunction, ..AbstractGradientFunction


smoothwall(u) = u ≤ 0 ? 0 : exp(u - 1/u)
smoothgrad(u) = u ≤ 0 ? 0 : (1 + 1/u^2) * smoothwall(u)

"""
    functions(λ̄, x̄L, x̄R, σ̄)

A smooth exponential penalty for each parameter exceeding its bounds.

# Arguments
- `λ̄`: vector of weights for each penalty
        Set `λ̄[i]=0` to skip penalties for the ith parameter.

- `μ̄R`: vector of lower bounds for each parameter
- `μ̄L`: vector of upper bounds for each parameter
- `σ̄`: vector of scalings (smaller=steeper) for each penalty

# Returns
- `f`: the cost function
- `g`: the gradient function

"""
function functions(λ̄, μ̄R, μ̄L, σ̄)
    f = CostFunction(λ̄, μ̄R, μ̄L, σ̄)
    g = GradientFunction(f)
    return f, g
end


struct CostFunction{F<:AbstractFloat} <: AbstractCostFunction
    λ̄::Vector{F}
    μ̄R::Vector{F}
    μ̄L::Vector{F}
    σ̄::Vector{F}

    ūR::Vector{F}
    ūL::Vector{F}

    function CostFunction(
        λ̄::AbstractVector,
        μ̄R::AbstractVector,
        μ̄L::AbstractVector,
        σ̄::AbstractVector,
    )
        L = length(λ̄)
        F = promote_type(Float16, eltype(λ̄), eltype(μ̄R), eltype(μ̄L), eltype(σ̄))

        return new{F}(
            convert(Array{F}, λ̄),
            convert(Array{F}, μ̄R),
            convert(Array{F}, μ̄L),
            convert(Array{F}, σ̄),
            Array{F}(undef, L),
            Array{F}(undef, L),
        )
    end
end

function (f::CostFunction)(x̄::AbstractVector)
    f.ūR .= (x̄ .- f.μ̄R) ./ f.σ̄
    f.ūL .= (f.μ̄L .- x̄) ./ f.σ̄

    value = 0
    for i in eachindex(x̄)
        f.λ̄[i] > 0 && (value += f.λ̄[i] * smoothwall(f.ūR[i]))
        f.λ̄[i] > 0 && (value += f.λ̄[i] * smoothwall(f.ūL[i]))
    end
    return value
end

struct GradientFunction{F<:AbstractFloat} <: AbstractGradientFunction
    f::CostFunction{F}
end

function (g::GradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)
    g.f.ūR .= (x̄ .- g.f.μ̄R) ./ g.f.σ̄
    g.f.ūL .= (g.f.μ̄L .- x̄) ./ g.f.σ̄

    ∇f̄ .= 0
    for i in eachindex(x̄)
        g.f.λ̄[i] > 0 && (∇f̄[i] += g.f.λ̄[i]/g.f.σ̄[i] * smoothgrad(g.f.ūR[i]))
        g.f.λ̄[i] > 0 && (∇f̄[i] -= g.f.λ̄[i]/g.f.σ̄[i] * smoothgrad(g.f.ūL[i]))
    end
    return ∇f̄
end
