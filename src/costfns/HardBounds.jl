import ..CostFunctions
export HardBound

wall(u) = exp(log(2) * u^2) - 1
grad(u) = exp(log(2) * u^2) * 2*log(2) * u

"""
    HardBound(λ̄, x̄L, x̄R, σ̄)

A steep exponential penalty for each parameter exceeding its bounds.

# Arguments
- `λ̄`: vector of weights for each penalty
        Set `λ̄[i]=0` to skip penalties for the ith parameter.

- `x̄L`: vector of lower bounds for each parameter
- `x̄R`: vector of upper bounds for each parameter
- `σ̄`: vector of scalings (smaller=steeper) for each penalty

"""
struct HardBound{F} <: CostFunctions.CostFunctionType{F}
    λ̄::Vector{F}
    x̄L::Vector{F}
    x̄R::Vector{F}
    σ̄::Vector{F}

    function HardBound(
        λ̄::AbstractVector,
        x̄L::AbstractVector,
        x̄R::AbstractVector,
        σ̄::AbstractVector,
    )
        F = promote_type(Float16, eltype(λ̄), eltype(x̄L), eltype(x̄R), eltype(σ̄))

        return new{F}(
            convert(Array{F}, λ̄),
            convert(Array{F}, x̄L),
            convert(Array{F}, x̄R),
            convert(Array{F}, σ̄),
        )
    end
end

Base.length(fn::HardBound) = length(fn.λ̄)

function CostFunctions.cost_function(fn::HardBound)
    return (x̄) -> (
        total = 0;
        for i in 1:length(fn);
            x, λ, xL, xR, σ = x̄[i], fn.λ̄[i], fn.x̄L[i], fn.x̄R[i], fn.σ̄[i];
            u = (x-xR)/σ;
            λ > 0 && u > 0 && (total += λ * wall(u));
            u = (xL-x)/σ;
            λ > 0 && u > 0 && (total += λ * wall(u));
        end;
        total
    )
end

function CostFunctions.grad_function_inplace(fn::HardBound)
    return (∇f̄, x̄) -> (
        ∇f̄ .= 0;
        for i in 1:length(fn);
            x, λ, xL, xR, σ = x̄[i], fn.λ̄[i], fn.x̄L[i], fn.x̄R[i], fn.σ̄[i];
            u = (x-xR)/σ;
            λ > 0 && u > 0 && (∇f̄[i] += λ * grad(u) / σ);
            u = (xL-x)/σ;
            λ > 0 && u > 0 && (∇f̄[i] -= λ * grad(u) / σ);
        end;
        ∇f̄
    )
end