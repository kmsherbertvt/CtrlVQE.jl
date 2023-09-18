import ..CostFunctions
export Normalization

import ..LinearAlgebraTools, ..QubitOperators
import ..Parameters, ..Devices, ..Evolutions
import ..Bases, ..Operators

"""
    Normalization(ψ0, T, device, r; kwargs...)

The norm of a statevector in a binary logical space.

# Arguments

- `evolution::Evolutions.TrotterEvolution`: which algorithm to evolve `ψ0` with
        A sensible choice is `ToggleEvolutions.Toggle(r)`,
            where `r` is the number of Trotter steps.
        Must be a TrotterEvolution because the gradient signal is inherently Trotterized.

- `device::Devices.DeviceType`: the device, which determines the time-evolution of `ψ0`

- `basis::Bases.BasisType`: the measurement basis
        ALSO determines the basis which `ψ0` is understood to be given in.
        An intuitive choice is `Bases.OCCUPATION`, aka. the qubits' Z basis.
        That said, there is some doubt whether, experimentally,
            projective measurement doesn't actually project on the device's eigenbasis,
            aka `Bases.DRESSED`.
        Note that you probably want to rotate `ψ0` if you change this argument.

- `T::Real`: the total time for the state to evolve under the `device` Hamiltonian.

- `ψ0`: the reference state, living in the physical Hilbert space of `device`.

"""
struct Normalization{F} <: CostFunctions.CostFunctionType{F}
    evolution::Evolutions.TrotterEvolution
    device::Devices.DeviceType
    basis::Bases.BasisType
    T::F
    ψ0::Vector{Complex{F}}

    function Normalization(
        evolution::Evolutions.TrotterEvolution,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        T::Real,
        ψ0::AbstractVector,
    )
        # INFER FLOAT TYPE AND CONVERT ARGUMENTS
        F = real(promote_type(Float16, eltype(ψ0), eltype(T)))

        # CREATE OBJECT
        return new{F}(
            evolution, device, basis,
            F(T),
            convert(Array{Complex{F}}, ψ0),
        )
    end
end

Base.length(fn::Normalization) = Parameters.count(fn.device)

function CostFunctions.cost_function(fn::Normalization)
    # DYNAMICALLY UPDATED STATEVECTOR
    ψ = copy(fn.ψ0)
    # OBSERVABLE - IT'S THE PROJECTION OPERATOR
    π̄ = QubitOperators.localqubitprojectors(fn.device)

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
        real(LinearAlgebraTools.expectation(π̄, ψ))
    )
end

function CostFunctions.grad_function_inplace(fn::Normalization{F}) where {F}
    # TIME GRID
    r = Evolutions.nsteps(fn.evolution)
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(fn.T, r)
    # OBSERVABLE - IT'S THE PROJECTION OPERATOR
    Π = QubitOperators.qubitprojector(fn.device)
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
            Π;
            result=ϕ̄,
        );
        ∇f̄ .= Devices.gradient(fn.device, τ̄, t̄, ϕ̄)
    )
end