import ..CostFunctions

import ...Parameters, ...LinearAlgebraTools, ...Devices, ...Evolutions
import ...QubitOperators
import ...Operators: StaticOperator, IDENTITY
import ...Bases: BasisType, OCCUPATION

"""
    Normalization(ψ0, T, device, r; kwargs...)

The norm of a statevector in a binary logical space.

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
"""
struct Normalization{F} <: CostFunctions.CostFunctionType{F}
    ψ0::Vector{Complex{F}}
    T::F
    device::Devices.Device
    r::Int
    algorithm::Evolutions.Algorithm
    basis::BasisType

    function Normalization(
        ψ0::AbstractVector,
        T::Real,
        device::Devices.Device,
        r::Int;
        algorithm::Evolutions.Algorithm=Evolutions.Rotate(r),
        basis::BasisType=OCCUPATION,
    )
        # INFER FLOAT TYPE AND CONVERT ARGUMENTS
        F = real(promote_type(Float16, eltype(ψ0), eltype(T)))

        # CREATE OBJECT
        return new{F}(
            convert(Array{Complex{F}}, ψ0),
            F(T), device, r,
            algorithm, basis,
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
            fn.algorithm,
            fn.device,
            fn.basis,
            fn.T,
            fn.ψ0;
            result=ψ,
        );
        real(LinearAlgebraTools.expectation(π̄, ψ))
    )
end

function CostFunctions.grad_function(fn::Normalization{F}) where {F}
    # TIME GRID
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(fn.T, fn.r)
    # OBSERVABLE - IT'S THE PROJECTION OPERATOR
    Π = QubitOperators.qubitprojector(fn.device)
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
            Π;
            result=ϕ̄,
            evolution=fn.algorithm,
        );
        ∇f̄ .= Devices.gradient(fn.device, τ̄, t̄, ϕ̄)
    )
end