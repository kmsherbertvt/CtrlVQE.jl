import ..CostFunctions
export SoftBound

wall(u) = u^2
grad(u) = 2u

"""
    SoftBound(λ̄, μ̄, σ̄)

A shallow quadratic penalty for each parameter's deviation from mean.

# Arguments
- `λ̄`: vector of weights for each penalty
        Set `λ̄[i]=0` to skip penalties for the ith parameter.

- `μ̄`: vector of means for each parameter
- `σ̄`: vector of scalings (smaller=steeper) for each penalty

"""
struct SoftBound{F} <: CostFunctions.CostFunctionType{F}
    λ̄::Vector{F}
    μ̄::Vector{F}
    σ̄::Vector{F}

    function SoftBound(λ̄::AbstractVector, μ̄::AbstractVector, σ̄::AbstractVector)
        F = promote_type(Float16, eltype(λ̄), eltype(μ̄), eltype(σ̄))
        return new{F}(
            convert(Array{F}, λ̄),
            convert(Array{F}, μ̄),
            convert(Array{F}, σ̄),
        )
    end
end

Base.length(fn::SoftBound) = length(fn.λ̄)

function CostFunctions.cost_function(fn::SoftBound)
    return (x̄) -> (
        total = 0;
        for i in 1:length(fn);
            x, λ, μ, σ = x̄[i], fn.λ̄[i], fn.μ̄[i], fn.σ̄[i];
            u = (x-μ)/σ;
            λ > 0 && (total += λ * wall(u));
        end;
        total
    )
end

function CostFunctions.grad_function_inplace(fn::SoftBound)
    return (∇f̄, x̄) -> (
        ∇f̄ .= 0;
        for i in 1:length(fn);
            x, λ, μ, σ = x̄[i], fn.λ̄[i], fn.μ̄[i], fn.σ̄[i];
            u = (x-μ)/σ;
            λ > 0 && (∇f̄[i] = λ * grad(u) / σ);
        end;
        ∇f̄
    )
end
