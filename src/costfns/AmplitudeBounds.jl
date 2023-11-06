import ..CostFunctions
export AmplitudeBound

# NOTE: Implicitly use smooth bounding function.
wall(u) = exp(u - 1/u)
grad(u) = exp(u - 1/u) * (1 + 1/u^2)

"""
    AmplitudeBound(ΩMAX, λ, σ, L, Ω, paired)

Smooth bounds for explicitly windowed amplitude parameters.

# Parameters
- `ΩMAX`: Maximum allowable amplitude on a device.
- `λ`: Penalty strength.
- `σ`: Penalty effective width: smaller means steeper.
- `L`: total number of parameters in cost function.
- `Ω`: array of indices corresponding to amplitudes (only these are penalized).
- `paired`: whether or not adjacent pairs of parameters give real+imag parts

"""
struct AmplitudeBound{F} <: CostFunctions.CostFunctionType{F}
    ΩMAX::F             # MAXIMUM PERMISSIBLE AMPLITUDE
    λ::F                # STRENGTH OF BOUNDS
    σ::F                # STEEPNESS OF BOUNDS
    L::Int              # TOTAL NUMBER OF PARAMETERS IN COST FUNCTION
    Ω::Vector{Int}      # LIST OF INDICES THAT CORRESPOND TO AMPLITUDES
    paired::Bool        # FLAG THAT ADJACENT ITEMS IN Ω ARE REAL AND IMAGINARY PARTS

    function AmplitudeBound(
        ΩMAX::Real,
        λ::Real,
        σ::Real,
        L::Int,
        Ω::AbstractVector{Int},
        paired::Bool,
    )
        F = promote_type(Float16, eltype(ΩMAX), eltype(λ), eltype(σ))
        return new{F}(ΩMAX, λ, σ, L, convert(Array{Int}, Ω), paired)
    end
end

Base.length(fn::AmplitudeBound) = fn.L

function CostFunctions.cost_function(fn::AmplitudeBound)
    if fn.paired
        Ωα = @view(fn.Ω[1:2:end])
        Ωβ = @view(fn.Ω[2:2:end])
    else
        Ωα = fn.Ω
    end

    return (x̄) -> (
        total = 0;
        for i in eachindex(Ωα);
            α = x̄[Ωα[i]]
            β = fn.paired ? x̄[Ωβ[i]] : zero(α);
            r = sqrt(α^2 + β^2);
            u = (r - fn.ΩMAX) / fn.σ;
            u > 0 && (total += fn.λ * wall(u));
        end;
        total
    )
end

function CostFunctions.grad_function_inplace(fn::AmplitudeBound)
    if fn.paired
        Ωα = @view(fn.Ω[1:2:end])
        Ωβ = @view(fn.Ω[2:2:end])
    else
        Ωα = fn.Ω
    end

    return (∇f̄, x̄) -> (
        ∇f̄ .= 0;
        for i in eachindex(Ωα);
            α = x̄[Ωα[i]]
            β = fn.paired ? x̄[Ωβ[i]] : zero(α);
            r = sqrt(α^2 + β^2);
            u = (r - fn.ΩMAX) / fn.σ;
            u > 0 && (∇f̄[Ωα[i]] += fn.λ * grad(u) / fn.σ * (α/r));
            fn.paired && u > 0 && (∇f̄[Ωβ[i]] += fn.λ * grad(u) / fn.σ * (β/r));
        end;
        ∇f̄
    )
end