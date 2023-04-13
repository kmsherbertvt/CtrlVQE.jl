import ...CostFunctions: AbstractCostFunction, AbstractGradientFunction

import ...LinearAlgebraTools, ...Devices
import Bases: BasisType, OCCUPATION
import Evolutions: Algorithm, Rotate

function functions(
    observable::AbstractMatrix,
    ψ0::AbstractVector,
    T::Real,
    device::Devices.Device,
    r::Int;
    algorithm=Rotate(r),
    basis=OCCUPATION,
)
    f = CostFunction(observable, ψ0, T, device, algorithm, basis)
    g = GradientFunction(f, r)
    return f, g
end

struct CostFunction{
    F<:AbstractFloat,
    D<:Devices.Device,
    A<:Evolutions.Algorithm,
    B<:Bases.BasisType,
} <: AbstractCostFunction
    observable::Matrix{Complex{F}}
    ψ0::Vector{Complex{F}}
    T::F
    device::D
    algorithm::A
    basis::B

    ψ::Vector{Complex{F}}

    function CostFunction(
        observable::AbstractMatrix,
        ψ0::AbstractVector,
        T::Real,
        device::D,
        algorithm::A,
        basis::B,
    ) where {D, A, B}
        # INFER FLOAT TYPE AND CONVERT ARGUMENTS
        F = real(promote_type(Float16, eltype(observable), eltype(ψ0), eltype(T)))
        observable = convert(Array{Complex{F}}, observable)
        ψ0 = convert(Array{Complex{F}}, ψ0)
        T = F(T)

        # CONSTRUCT PRE-ALLOCATED VARIABLES
        ψ = Array{LinearAlgebraTools.cis_type(F)}(undef, size(ψ0))

        # CREATE OBJECT
        return new{F,D,A,B}(observable, ψ0, T, device, algorithm, basis, ψ)
    end
end

function (f::CostFunction)(x̄::AbstractVector)
    x̄0 = Devices.values(f.device)
    Devices.bind(f.device, x̄)
    ψ = Evolutions.evolve(
        f.algorithm,
        f.device,
        f.basis,
        f.T,
        f.ψ0;
        result=ψ,
    )
    Devices.bind(f.device, x̄0)
    return real(LinearAlgebraTools.expectation(f.observable, ψ))
end

# TODO (hi): bonus method to calculate energy for given ψ, useful for evolve callback




struct GradientFunction{
    F<:AbstractFloat,
    D<:Devices.Device,
    A<:Evolutions.Algorithm,
    B<:Bases.BasisType,
} <: AbstractGradientFunction
    f::CostFunction{F,D,A,B}
    r::Int

    Φ̄::Matrix{F}

    function GradientFunction(
        f::CostFunction{F,D,A,B},
        r::Int,
    ) where {F, D, A, B}
        Φ̄ = Array{F}(undef, r+1, Devices.ngrades(f.device))
        return new{F,D,A,B}(f,r,Φ̄)
    end
end

function (g::GradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)
    x̄0 = Devices.values(g.f.device)
    Devices.bind(g.f.device, x̄)
    Evolutions.gradientsignals(
        g.f.device,
        g.f.basis,
        g.f.T,
        g.f.ψ0,
        g.r,
        g.f.observable;
        result=g.Φ̄,
        evolution=g.f.algorithm,
    )
    Devices.bind(g.f.device, x̄0)
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(g.f.T, g.r)
    ∇f̄ .= Devices.gradient(g.f.device, τ̄, t̄, g.Φ̄)
    return ∇f̄
end