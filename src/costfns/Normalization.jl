import ...EnergyFunctions, ...AbstractGradientFunction

import ....Parameters, ....LinearAlgebraTools, ....Devices, ....Evolutions
import ....QubitOperators
import ....Bases: BasisType, OCCUPATION

function functions(
    ψ0::AbstractVector,
    T::Real,
    device::Devices.Device,
    r::Int;
    algorithm=Evolutions.Rotate(r),
    basis=OCCUPATION,
)
    f = CostFunction(ψ0, T, device, algorithm, basis)
    g = GradientFunction(f, r)
    return f, g
end

struct CostFunction{
    F<:AbstractFloat,
    D<:Devices.Device,
    A<:Evolutions.Algorithm,
    B<:BasisType,
} <: EnergyFunctions.AbstractEnergyFunction
    ψ0::Vector{Complex{F}}
    T::F
    device::D
    algorithm::A
    basis::B

    ψ::Vector{Complex{F}}
    π̄::Vector{Matrix{Bool}}
    # TODO (mid): convert to 3d array

    function CostFunction(
        ψ0::AbstractVector,
        T::Real,
        device::D,
        algorithm::A,
        basis::B,
    ) where {D, A, B}
        # INFER FLOAT TYPE AND CONVERT ARGUMENTS
        F = real(promote_type(Float16, eltype(ψ0), eltype(T)))
        ψ0 = convert(Array{Complex{F}}, ψ0)
        T = F(T)

        # CONSTRUCT PRE-ALLOCATED VARIABLES
        ψ = Array{LinearAlgebraTools.cis_type(F)}(undef, size(ψ0))
        π̄ = QubitOperators.localqubitprojectors(device)

        # CREATE OBJECT
        return new{F,D,A,B}(ψ0, T, device, algorithm, basis, ψ, π̄)
    end
end

function (f::CostFunction)(x̄::AbstractVector)
    Parameters.bind(f.device, x̄)
    Evolutions.evolve(
        f.algorithm,
        f.device,
        f.basis,
        f.T,
        f.ψ0;
        result=f.ψ,
    )
    return EnergyFunctions.evaluate(f, f.ψ)
end

function EnergyFunctions.evaluate(f::CostFunction, ψ::AbstractVector)
    return real(LinearAlgebraTools.expectation(f.π̄, ψ))
end

function EnergyFunctions.evaluate(f::CostFunction, ψ::AbstractVector, t::Real)
    return EnergyFunctions.evaluate(f, ψ)
end

struct GradientFunction{
    F<:AbstractFloat,
    D<:Devices.Device,
    A<:Evolutions.Algorithm,
    B<:BasisType,
} <: AbstractGradientFunction
    f::CostFunction{F,D,A,B}
    r::Int

    ϕ̄::Matrix{F}
    Π::Matrix{Bool}

    function GradientFunction(
        f::CostFunction{F,D,A,B},
        r::Int,
    ) where {F, D, A, B}
        ϕ̄ = Array{F}(undef, r+1, Devices.ngrades(f.device))
        Π = QubitOperators.qubitprojector(f.device)
        return new{F,D,A,B}(f,r,ϕ̄,Π)
    end
end

function (g::GradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)
    Parameters.bind(g.f.device, x̄)
    Evolutions.gradientsignals(
        g.f.device,
        g.f.basis,
        g.f.T,
        g.f.ψ0,
        g.r,
        g.Π;
        result=g.ϕ̄,
        evolution=g.f.algorithm,
    )
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(g.f.T, g.r)
    ∇f̄ .= Devices.gradient(g.f.device, τ̄, t̄, g.ϕ̄)
    return ∇f̄
end