import ..CostFunctions
export SmoothBound

wall(u) = exp(u - 1/u)
grad(u) = exp(u - 1/u) * (1 + 1/u^2)

"""
    SmoothBound(λ̄, x̄R, x̄L, σ̄)

A smooth exponential penalty for each parameter exceeding its bounds.

# Arguments
- `λ̄`: vector of weights for each penalty
        Set `λ̄[i]=0` to skip penalties for the ith parameter.

- `x̄R`: vector of upper bounds for each parameter
- `x̄L`: vector of lower bounds for each parameter
- `σ̄`: vector of scalings (smaller=steeper) for each penalty

"""
struct SmoothBound{F} <: CostFunctions.CostFunctionType{F}
    λ̄::Vector{F}
    μ̄R::Vector{F}
    μ̄L::Vector{F}
    σ̄::Vector{F}

    function SmoothBound(
        λ̄::AbstractVector,
        μ̄R::AbstractVector,
        μ̄L::AbstractVector,
        σ̄::AbstractVector,
    )
        F = promote_type(Float16, eltype(λ̄), eltype(μ̄R), eltype(μ̄L), eltype(σ̄))

        return new{F}(
            convert(Array{F}, λ̄),
            convert(Array{F}, μ̄R),
            convert(Array{F}, μ̄L),
            convert(Array{F}, σ̄),
        )
    end
end

Base.length(fn::SmoothBound) = length(fn.λ̄)

function CostFunctions.cost_function(fn::SmoothBound)
    return (x̄) -> (
        total = 0;
        for i in 1:length(fn);
            x, λ, μR, μL, σ = x̄[i], fn.λ̄[i], fn.μ̄R[i], fn.μ̄L[i], fn.σ̄[i];
            u = (x-μR)/σ;
            λ > 0 && u > 0 && (total += λ * wall(u));
            u = (μL-x)/σ;
            λ > 0 && u > 0 && (total += λ * wall(u));
        end;
        total
    )
end

function CostFunctions.grad_function_inplace(fn::SmoothBound)
    return (∇f̄, x̄) -> (
        ∇f̄ .= 0;
        for i in 1:length(fn);
            x, λ, μR, μL, σ = x̄[i], fn.λ̄[i], fn.μ̄R[i], fn.μ̄L[i], fn.σ̄[i];
            u = (x-μR)/σ;
            λ > 0 && u > 0 && (∇f̄[i] += λ * grad(u) / σ);
            u = (μL-x)/σ;
            λ > 0 && u > 0 && (∇f̄[i] -= λ * grad(u) / σ);
        end;
        ∇f̄
    )
end