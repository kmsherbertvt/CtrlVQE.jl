import ..CostFunctions
export NormalizedEnergy

import ..LinearAlgebraTools, ..QubitOperators
import ..Parameters, ..Devices, ..Evolutions
import ..Bases, ..Operators

"""
    NormalizedEnergy(O0, ψ0, T, device, r; kwargs...)

Expectation value of a Hermitian observable.

The statevector is projected onto a binary logical space after time evolution,
    and then renormalized,
    modeling quantum measurement where leakage is completely obscured.

The frame rotation (if provided) is applied to the molecular hamiltonian,
    rather than to the state.

# Arguments
- `O0`: a Hermitian matrix, living in the physical Hilbert space of `device`.
- `ψ0`: the reference state, living in the physical Hilbert space of `device`.
- `T::Real`: the total time for the state to evolve under the `device` Hamiltonian.
- `device::Devices.DeviceType`: the device
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
struct NormalizedEnergy{F} <: CostFunctions.CostFunctionType{F}
    O0::Matrix{Complex{F}}
    ψ0::Vector{Complex{F}}
    T::F
    device::Devices.DeviceType
    r::Int
    algorithm::Evolutions.Algorithm
    basis::Bases.BasisType
    frame::Operators.StaticOperator

    function NormalizedEnergy(
        O0::AbstractMatrix,
        ψ0::AbstractVector,
        T::Real,
        device::Devices.DeviceType,
        r::Int;
        algorithm::Evolutions.Algorithm=Evolutions.Rotate(r),
        basis::Bases.BasisType=Bases.OCCUPATION,
        frame::Operators.StaticOperator=Operators.IDENTITY,
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

Base.length(fn::NormalizedEnergy) = Parameters.count(fn.device)

function CostFunctions.cost_function(fn::NormalizedEnergy)
    # DYNAMICALLY UPDATED STATEVECTOR
    ψ = copy(fn.ψ0)
    # OBSERVABLE, IN MEASUREMENT FRAME
    OT = copy(fn.O0); Devices.evolve!(fn.frame, fn.device, fn.T, OT)
    # INCLUDE PROJECTION ONTO COMPUTATIONAL SUBSPACE IN THE MEASUREMENT
    LinearAlgebraTools.rotate!(QubitOperators.localqubitprojectors(fn.device), OT)
    # THE PROJECTION OPERATOR
    π̄ = QubitOperators.localqubitprojectors(fn.device)

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
        E = real(LinearAlgebraTools.expectation(OT, ψ));
        F = real(LinearAlgebraTools.expectation(π̄, ψ));
        E / F
    )
end

function CostFunctions.grad_function(fn::NormalizedEnergy{F}) where {F}
    # TIME GRID
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(fn.T, fn.r)
    # THE PROJECTION OPERATOR, FOR COMPONENT COST FUNCTION EVALUATIONS
    π̄ = QubitOperators.localqubitprojectors(fn.device)
    # DYNAMICALLY UPDATED STATEVECTOR
    ψ = copy(fn.ψ0)

    # THE "MATRIX LIST" (A 3D ARRAY), FOR EACH GRADIENT SIGNAL
    Ō = Array{eltype(ψ)}(undef, (size(fn.O0)..., 2))
    # FIRST MATRIX: THE OBSERVABLE, IN MEASUREMENT FRAME
    OT = @view(Ō[:,:,1])
    OT .= fn.O0; Devices.evolve!(fn.frame, fn.device, fn.T, OT)
    # INCLUDE PROJECTION ONTO COMPUTATIONAL SUBSPACE IN THE MEASUREMENT
    LinearAlgebraTools.rotate!(π̄, OT)
    # SECOND MATRIX: PROJECTION OPERATOR, AS A GLOBAL OPERATOR
    LinearAlgebraTools.kron(π̄; result=@view(Ō[:,:,2]))

    # GRADIENT VECTORS
    ϕ̄ = Array{F}(undef, fn.r+1, Devices.ngrades(fn.device), 2)
    ∂E = Array{F}(undef, length(fn))
    ∂N = Array{F}(undef, length(fn))

    return (∇f̄, x̄) -> (
        Parameters.bind(fn.device, x̄);
        Evolutions.evolve(
            fn.algorithm,
            fn.device,
            fn.basis,
            fn.T,
            fn.ψ0;
            result=ψ,
        );
        E = real(LinearAlgebraTools.expectation(OT, ψ));
        N = real(LinearAlgebraTools.expectation(π̄, ψ));

        Parameters.bind(fn.device, x̄);
        Evolutions.gradientsignals(
            fn.device,
            fn.basis,
            fn.T,
            fn.ψ0,
            fn.r,
            Ō;
            result=ϕ̄,
            evolution=fn.algorithm,
        );
        ∂E .= Devices.gradient(fn.device, τ̄, t̄, @view(ϕ̄[:,:,1]));
        ∂N .= Devices.gradient(fn.device, τ̄, t̄, @view(ϕ̄[:,:,2]));

        ∇f̄ .= (∂E./N) .- (E/N) .* (∂N./N)
    )
end
