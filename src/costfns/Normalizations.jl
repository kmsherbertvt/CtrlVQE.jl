import ..CostFunctions
export Normalization

import ..LinearAlgebraTools, ..QubitOperators
import ..Parameters, ..Integrations, ..Devices, ..Evolutions
import ..Bases, ..Operators

import ..TrapezoidalIntegrations: TrapezoidalIntegration

"""
    Normalization(evolution, device, basis, grid, ψ0; kwargs...)

The norm of a statevector in a binary logical space.

# Arguments

- `evolution::Evolutions.EvolutionType`: the algorithm with which to evolve `ψ0`
        A sensible choice is `ToggleEvolutions.TOGGLE`

- `device::Devices.DeviceType`: the device, which determines the time-evolution of `ψ0`

- `basis::Bases.BasisType`: the measurement basis
        ALSO determines the basis which `ψ0` is understood to be given in.
        An intuitive choice is `Bases.OCCUPATION`, aka. the qubits' Z basis.
        That said, there is some doubt whether, experimentally,
            projective measurement doesn't actually project on the device's eigenbasis,
            aka `Bases.DRESSED`.
        Note that you probably want to rotate `ψ0` if you change this argument.

- `grid::TrapezoidalIntegration`: defines the time integration bounds (eg. from 0 to `T`)

- `ψ0`: the reference state, living in the physical Hilbert space of `device`.

"""
struct Normalization{F} <: CostFunctions.EnergyFunction{F}
    evolution::Evolutions.EvolutionType
    device::Devices.DeviceType
    basis::Bases.BasisType
    grid::TrapezoidalIntegration
    ψ0::Vector{Complex{F}}

    function Normalization(
        evolution::Evolutions.EvolutionType,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        grid::TrapezoidalIntegration,
        ψ0::AbstractVector,
    )
        # INFER FLOAT TYPE AND CONVERT ARGUMENTS
        F = real(promote_type(Float16, eltype(ψ0), eltype(grid)))

        # CREATE OBJECT
        return new{F}(
            evolution, device, basis, grid,
            convert(Array{Complex{F}}, ψ0),
        )
    end
end

Base.length(fn::Normalization) = Parameters.count(fn.device)

function CostFunctions.trajectory_callback(
    fn::Normalization,
    F::AbstractVector;
    callback=nothing
)
    workbasis = Evolutions.workbasis(fn.evolution)      # BASIS OF CALLBACK ψ
    U = Devices.basisrotation(fn.basis, workbasis, fn.device)
    π̄ = QubitOperators.localqubitprojectors(fn.device)
    ψ_ = similar(fn.ψ0)

    return (i, t, ψ) -> (
        ψ_ .= ψ;
        LinearAlgebraTools.rotate!(U, ψ_);  # ψ_ IS NOW IN MEASUREMENT BASIS
        F[i] = real(LinearAlgebraTools.expectation(π̄, ψ_));
        !isnothing(callback) && callback(i, t, ψ)
    )
end

function CostFunctions.cost_function(fn::Normalization; callback=nothing)
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
            fn.grid,
            fn.ψ0;
            result=ψ,
            callback=callback,
        );
        real(LinearAlgebraTools.expectation(π̄, ψ))
    )
end

function CostFunctions.grad_function_inplace(fn::Normalization{F}; ϕ=nothing) where {F}
    r = Integrations.nsteps(fn.grid)

    if isnothing(ϕ)
        return CostFunctions.grad_function_inplace(
            fn;
            ϕ=Array{F}(undef, r+1, Devices.ngrades(fn.device))
        )
    end

    # OBSERVABLE - IT'S THE PROJECTION OPERATOR
    Π = QubitOperators.qubitprojector(fn.device)

    return (∇f̄, x̄) -> (
        Parameters.bind(fn.device, x̄);
        Evolutions.gradientsignals(
            fn.evolution,
            fn.device,
            fn.basis,
            fn.grid,
            fn.ψ0,
            Π;
            result=ϕ,   # NOTE: This writes the gradient signal as needed.
        );
        ∇f̄ .= Devices.gradient(fn.device, fn.grid, ϕ)
    )
end