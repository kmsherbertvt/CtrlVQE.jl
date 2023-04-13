import ...CostFunctions: AbstractCostFunction, AbstractGradientFunction

function functions(λ̄, x̄L, x̄R, σ̄)
    f = CostFunction(λ̄, x̄L, x̄R, σ̄)
    g = GradientFunction(f)
    return f, g
end

struct CostFunction{F<:AbstractFloat} <: AbstractCostFunction
    λ̄::Vector{F}
    x̄L::Vector{F}
    x̄R::Vector{F}
    σ̄::Vector{F}

    function CostFunction(
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

function (f::CostFunction)(x̄::AbstractVector)
    χ̄L = (f.x̄L .- x̄) ./ f.σ̄
    χ̄R = (x̄ .- f.x̄R) ./ f.σ̄
    λ̄ = vcat(f.λ̄, f.λ̄)
    χ̄ = vcat(χ̄L, χ̄R)
    Θ(x) = x > 0 ? 1 : 0

    return sum(λ̄ .* Θ.(χ̄) .* (exp.( log(2) .* χ̄.^2 ) .- 1))
end

struct GradientFunction{F<:AbstractFloat} <: AbstractGradientFunction
    f::CostFunction{F}
end

function (g::GradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)
    χ̄L = (f.x̄L .- x̄) ./ f.σ̄
    χ̄R = (x̄ .- f.x̄R) ./ f.σ̄
    λ̄ = vcat(f.λ̄, f.λ̄)
    χ̄ = vcat(χ̄L, χ̄R)
    Θ(x) = x > 0 ? 1 : 0

    ∇f̄ .= (2log(2)) .* λ̄ .* χ̄ .* Θ.(χ̄) .* exp.( log(2) .* χ̄.^2 )
    # DERIVATIVE OF Θ(x) THROWS IN A -1 ON χ̄L TERMS
    ∇f̄[1:end÷2] .*= -1
    return ∇f̄
end
