import ...CostFunctions: AbstractCostFunction, AbstractGradientFunction

import ...LinearAlgebraTools, ...Devices, ...QubitOperators
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
    projected::Matrix{Complex{F}}
    Π::Matrix{Bool}

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
        π̄ = QubitOperators.localqubitprojectors(device)
        projected = copy(observable); LinearAlgebraTools.rotate!(π̄, projected)
        Π = LinearAlgebraTools.kron(π̄)

        # CREATE OBJECT
        return new{F,D,A,B}(observable,ψ0,T,device,algorithm,basis,ψ,projected,Π)
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

    E = real(LinearAlgebraTools.expectation(f.projected, ψ))
    F = real(LinearAlgebraTools.expectation(f.Π, ψ))
    return E / F
end

# TODO (hi): make another costfn devoted to ⟨Π⟩, this one just uses fE, fN!
# TODO (hi): bonus method to calculate energy for given ψ, useful for evolve callback

struct GradientFunction{
    F<:AbstractFloat,
    D<:Devices.Device,
    A<:Evolutions.Algorithm,
    B<:Bases.BasisType,
} <: AbstractGradientFunction
    f::CostFunction{F,D,A,B}
    r::Int

    ψ::Vector{Complex{F}}
    Φ̄::Array{F,3}

    function GradientFunction(
        f::CostFunction{F,D,A,B},
        r::Int,
    ) where {F, D, A, B}
        ψ = copy(f.ψ)
        Φ̄ = Array{F}(undef, r+1, Devices.ngrades(f.device), 2)
        return new{F,D,A,B}(f,r,ψ,Φ̄)
    end
end

function (g::GradientFunction)(∇f̄::AbstractVector, x̄::AbstractVector)
    x̄0 = Devices.values(g.f.device)
    Devices.bind(g.f.device, x̄)
    Evolutions.evolve(
        g.f.algorithm,
        g.f.device,
        g.f.basis,
        g.f.T,
        g.f.ψ0;
        result=g.ψ,
    )

    E = real(LinearAlgebraTools.expectation(g.f.projected, g.ψ))
    F = real(LinearAlgebraTools.expectation(g.f.Π, g.ψ))

    Evolutions.gradientsignals(
        g.f.device,
        g.f.basis,
        g.f.T,
        g.f.ψ0,
        g.r,
        [g.f.projected, g.f.Π];
        result=g.Φ̄,
        evolution=g.algorithm,
    )
    Devices.bind(g.f.device, x̄0)

    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(g.f.T, g.r)
    ∂E = Devices.gradient(g.f.device, τ̄, t̄, g.Φ̄[:,:,1])
    ∂F = Devices.gradient(g.f.device, τ̄, t̄, g.Φ̄[:,:,2])

    ∇f̄ .= (∂E./F) .- (E/F) .* (∂F./F)
    return ∇f̄
end