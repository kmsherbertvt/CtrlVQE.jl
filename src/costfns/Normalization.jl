import ...EnergyFunctions, ...AbstractGradientFunction

import ....Parameters, ....LinearAlgebraTools, ....Devices, ....Evolutions
import ....QubitOperators
import ....Bases: BasisType, OCCUPATION

"""
    functions(ψ0, T, device, r; kwargs...)

Cost and gradient functions for the norm of a statevector in a binary logical space.

# Arguments
- `ψ0`: the reference state, living in the physical Hilbert space of `device`.
- `T::Real`: the total time for the state to evolve under the `device` Hamiltonian.
- `device::Devices.Device`: the device
- `r::Int`: the number of time steps to calculate the gradient signal

# Keyword Arguments
- `algorithm::Evolutions.Algorithm`: which algorithm to evolve `ψ0` with.
        Defaults to `Evolutions.rotate(r)`.

- `basis::Bases.BasisType`: which basis `O0` and `ψ0` are represented in.
        ALSO determines the basis in which time-evolution is carried out.
        Defaults to `Bases.OCCUPATION`.

# Returns
- `f`: the cost function
- `g`: the gradient function

"""
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
    π̄::LinearAlgebraTools.MatrixList{Bool}

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