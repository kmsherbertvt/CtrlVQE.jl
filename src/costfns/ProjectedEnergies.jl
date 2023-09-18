import ..CostFunctions
export ProjectedEnergy

import ..LinearAlgebraTools, ..QubitOperators
import ..Parameters, ..Devices, ..Evolutions
import ..Bases, ..Operators

"""
    ProjectedEnergy(O0, ψ0, T, device, r; kwargs...)

Expectation value of a Hermitian observable.

The statevector is projected onto a binary logical space after time evolution,
    modeling an ideal quantum measurement where leakage is fully characterized.

# Arguments

- `evolution::Evolutions.TrotterEvolution`: which algorithm to evolve `ψ0` with
        A sensible choice is `ToggleEvolutions.Toggle(r)`,
            where `r` is the number of Trotter steps.
        Must be a TrotterEvolution because the gradient signal is inherently Trotterized.

- `device::Devices.DeviceType`: the device, which determines the time-evolution of `ψ0`

- `basis::Bases.BasisType`: the measurement basis
        ALSO determines the basis which `ψ0` and `O0` are understood to be given in.
        An intuitive choice is `Bases.OCCUPATION`, aka. the qubits' Z basis.
        That said, there is some doubt whether, experimentally,
            projective measurement doesn't actually project on the device's eigenbasis,
            aka `Bases.DRESSED`.
        Note that you probably want to rotate `ψ0` and `O0` if you change this argument.

- `frame::Operators.StaticOperator`: the measurement frame
        Think of this as a time-dependent basis rotation, which is applied to `O0`.
        A sensible choice is `Operators.STATIC` for the "drive frame",
            which ensures a zero pulse (no drive) system retains the same energy for any T.
        Alternatively, use `Operators.UNCOUPLED` for the interaction frame,
            a (presumably) classically tractable approximation to the drive frame,
            or `Operators.IDENTITY` to omit the time-dependent rotation entirely.

- `T::Real`: the total time for the state to evolve under the `device` Hamiltonian.

- `ψ0`: the reference state, living in the physical Hilbert space of `device`.

- `O0`: a Hermitian matrix, living in the physical Hilbert space of `device`.

"""
struct ProjectedEnergy{F} <: CostFunctions.CostFunctionType{F}
    evolution::Evolutions.TrotterEvolution
    device::Devices.DeviceType
    basis::Bases.BasisType
    frame::Operators.StaticOperator
    T::F
    ψ0::Vector{Complex{F}}
    O0::Matrix{Complex{F}}

    function ProjectedEnergy(
        evolution::Evolutions.TrotterEvolution,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        frame::Operators.StaticOperator,
        T::Real,
        ψ0::AbstractVector,
        O0::AbstractMatrix,
    )
        # INFER FLOAT TYPE AND CONVERT ARGUMENTS
        F = real(promote_type(Float16, eltype(O0), eltype(ψ0), eltype(T)))

        # CREATE OBJECT
        return new{F}(
            evolution, device, basis, frame,
            F(T),
            convert(Array{Complex{F}}, ψ0),
            convert(Array{Complex{F}}, O0),
        )
    end
end

Base.length(fn::ProjectedEnergy) = Parameters.count(fn.device)

function CostFunctions.cost_function(fn::ProjectedEnergy)
    # DYNAMICALLY UPDATED STATEVECTOR
    ψ = copy(fn.ψ0)
    # OBSERVABLE, IN MEASUREMENT FRAME
    OT = copy(fn.O0); Devices.evolve!(fn.frame, fn.device, fn.basis, fn.T, OT)
    # INCLUDE PROJECTION ONTO COMPUTATIONAL SUBSPACE IN THE MEASUREMENT
    LinearAlgebraTools.rotate!(QubitOperators.localqubitprojectors(fn.device), OT)

    return (x̄) -> (
        Parameters.bind(fn.device, x̄);
        Evolutions.evolve(
            fn.evolution,
            fn.device,
            fn.basis,
            fn.T,
            fn.ψ0;
            result=ψ,
        );
        real(LinearAlgebraTools.expectation(OT, ψ))
    )
end

function CostFunctions.grad_function_inplace(fn::ProjectedEnergy{F}) where {F}
    # TIME GRID
    r = Evolutions.nsteps(fn.evolution)
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(fn.T, r)
    # OBSERVABLE, IN MEASUREMENT FRAME
    OT = copy(fn.O0); Devices.evolve!(fn.frame, fn.device, fn.basis, fn.T, OT)
    # INCLUDE PROJECTION ONTO COMPUTATIONAL SUBSPACE IN THE MEASUREMENT
    LinearAlgebraTools.rotate!(QubitOperators.localqubitprojectors(fn.device), OT)
    # GRADIENT VECTORS
    ϕ̄ = Array{F}(undef, r+1, Devices.ngrades(fn.device))

    return (∇f̄, x̄) -> (
        Parameters.bind(fn.device, x̄);
        Evolutions.gradientsignals(
            fn.evolution,
            fn.device,
            fn.basis,
            fn.T,
            fn.ψ0,
            OT;
            result=ϕ̄,
        );
        ∇f̄ .= Devices.gradient(fn.device, τ̄, t̄, ϕ̄)
    )
end
