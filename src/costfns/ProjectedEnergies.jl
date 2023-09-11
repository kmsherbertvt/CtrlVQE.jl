import ..CostFunctions

import ..Parameters, ..LinearAlgebraTools, ..Devices, ..Evolutions
import ..QubitOperators
import ..Operators: StaticOperator, IDENTITY
import ..Bases: BasisType, OCCUPATION

"""
    ProjectedEnergy(O0, ψ0, T, device, r; kwargs...)

Expectation value of a Hermitian observable.

The statevector is projected onto a binary logical space after time evolution,
    modeling an ideal quantum measurement where leakage is fully characterized.

The frame rotation (if provided) is applied to the molecular hamiltonian,
    rather than to the state.

# Arguments
- `O0`: a Hermitian matrix, living in the physical Hilbert space of `device`.
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

- `frame::Operators.StaticOperator`: which frame to measure expecation values in.
        Use `Operators.STATIC` for the drive frame,
            which preserves the reference energy for a zero pulse.
        Use `Operators.UNCOUPLED` for the interaction frame,
            a (presumably) classically tractable approximation to the drive frame.
        Defaults to `Operators.IDENTITY`.
"""
struct ProjectedEnergy{F} <: CostFunctions.CostFunctionType{F}
    O0::Matrix{Complex{F}}
    ψ0::Vector{Complex{F}}
    T::F
    device::Devices.Device
    r::Int
    algorithm::Evolutions.Algorithm
    basis::BasisType
    frame::StaticOperator

    function ProjectedEnergy(
        O0::AbstractMatrix,
        ψ0::AbstractVector,
        T::Real,
        device::Devices.Device,
        r::Int;
        algorithm::Evolutions.Algorithm=Evolutions.Rotate(r),
        basis::BasisType=OCCUPATION,
        frame::StaticOperator=IDENTITY,
    )
        # INFER FLOAT TYPE AND CONVERT ARGUMENTS
        F = real(promote_type(Float16, eltype(O0), eltype(ψ0), eltype(T)))

        # CREATE OBJECT
        return new{F}(
            convert(Array{Complex{F}}, O0),
            convert(Array{Complex{F}}, ψ0),
            F(T), device, r,
            algorithm, basis, frame,
        )
    end
end

Base.length(fn::ProjectedEnergy) = Parameters.count(fn.device)

function CostFunctions.cost_function(fn::ProjectedEnergy)
    # DYNAMICALLY UPDATED STATEVECTOR
    ψ = copy(fn.ψ0)
    # OBSERVABLE, IN MEASUREMENT FRAME
    OT = copy(fn.O0); Devices.evolve!(fn.frame, fn.device, fn.T, OT)
    # INCLUDE PROJECTION ONTO COMPUTATIONAL SUBSPACE IN THE MEASUREMENT
    LinearAlgebraTools.rotate!(QubitOperators.localqubitprojectors(fn.device), OT)

    return (x̄) -> (
        Parameters.bind(fn.device, x̄);
        Evolutions.evolve(
            fn.algorithm,
            fn.device,
            fn.basis,
            fn.T,
            fn.ψ0;
            result=ψ,
        );
        real(LinearAlgebraTools.expectation(OT, ψ))
    )
end

function CostFunctions.grad_function(fn::ProjectedEnergy{F}) where {F}
    # TIME GRID
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(fn.T, fn.r)
    # OBSERVABLE, IN MEASUREMENT FRAME
    OT = copy(fn.O0); Devices.evolve!(fn.frame, fn.device, fn.T, OT)
    # INCLUDE PROJECTION ONTO COMPUTATIONAL SUBSPACE IN THE MEASUREMENT
    LinearAlgebraTools.rotate!(QubitOperators.localqubitprojectors(fn.device), OT)
    # GRADIENT VECTORS
    ϕ̄ = Array{F}(undef, fn.r+1, Devices.ngrades(fn.device))

    return (∇f̄, x̄) -> (
        Parameters.bind(fn.device, x̄);
        Evolutions.gradientsignals(
            fn.device,
            fn.basis,
            fn.T,
            fn.ψ0,
            fn.r,
            OT;
            result=ϕ̄,
            evolution=fn.algorithm,
        );
        ∇f̄ .= Devices.gradient(fn.device, τ̄, t̄, ϕ̄)
    )
end
