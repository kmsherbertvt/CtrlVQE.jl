import ..CostFunctions
export NormalizedEnergy

import ..LinearAlgebraTools, ..QubitOperators
import ..Parameters, ..Integrations, ..Devices, ..Evolutions
import ..Bases, ..Operators

import ..TrapezoidalIntegrations: TrapezoidalIntegration

"""
    NormalizedEnergy(evolution, device, basis, frame, grid, ψ0, O0; kwargs...)

Expectation value of a Hermitian observable.

The statevector is projected onto a binary logical space after time evolution,
    and then renormalized,
    modeling quantum measurement where leakage is completely obscured.

# Arguments

- `evolution::Evolutions.EvolutionType`: the algorithm with which to evolve `ψ0`
        A sensible choice is `ToggleEvolutions.TOGGLE`

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

- `grid::TrapezoidalIntegration`: defines the time integration bounds (eg. from 0 to `T`)

- `ψ0`: the reference state, living in the physical Hilbert space of `device`.

- `O0`: a Hermitian matrix, living in the physical Hilbert space of `device`.

"""
struct NormalizedEnergy{F} <: CostFunctions.EnergyFunction{F}
    evolution::Evolutions.EvolutionType
    device::Devices.DeviceType
    basis::Bases.BasisType
    frame::Operators.StaticOperator
    grid::TrapezoidalIntegration
    ψ0::Vector{Complex{F}}
    O0::Matrix{Complex{F}}

    function NormalizedEnergy(
        evolution::Evolutions.EvolutionType,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        frame::Operators.StaticOperator,
        grid::TrapezoidalIntegration,
        ψ0::AbstractVector,
        O0::AbstractMatrix,
    )
        # INFER FLOAT TYPE AND CONVERT ARGUMENTS
        F = real(promote_type(Float16, eltype(O0), eltype(ψ0), eltype(grid)))

        # CREATE OBJECT
        return new{F}(
            evolution, device, basis, frame, grid,
            convert(Array{Complex{F}}, ψ0),
            convert(Array{Complex{F}}, O0),
        )
    end
end

Base.length(fn::NormalizedEnergy) = Parameters.count(fn.device)

function CostFunctions.trajectory_callback(
    fn::NormalizedEnergy,
    En::AbstractVector;
    callback=nothing
)
    workbasis = Evolutions.workbasis(fn.evolution)      # BASIS OF CALLBACK ψ
    U = Devices.basisrotation(fn.basis, workbasis, fn.device)
    π̄ = QubitOperators.localqubitprojectors(fn.device)
    ψ_ = similar(fn.ψ0)

    return (i, t, ψ) -> (
        ψ_ .= ψ;
        LinearAlgebraTools.rotate!(U, ψ_);  # ψ_ IS NOW IN MEASUREMENT BASIS
        LinearAlgebraTools.rotate!(π̄, ψ_);  # ψ_ IS NOW "MEASURED"
        # APPLY FRAME ROTATION TO STATE RATHER THAN OBSERVABLE
        Devices.evolve!(fn.frame, fn.device, fn.basis, -t, ψ_);
            # NOTE: Rotating observable only makes sense when t is always the same.
        E = real(LinearAlgebraTools.expectation(fn.O0, ψ_));
        F = real(LinearAlgebraTools.expectation(π̄, ψ_));
        En[i] = E / F;
        !isnothing(callback) && callback(i, t, ψ)
    )
end

function CostFunctions.cost_function(fn::NormalizedEnergy; callback=nothing)
    # DYNAMICALLY UPDATED STATEVECTOR
    ψ = copy(fn.ψ0)
    # OBSERVABLE, IN MEASUREMENT FRAME
    T = Integrations.endtime(fn.grid)
    OT = copy(fn.O0); Devices.evolve!(fn.frame, fn.device, fn.basis, T, OT)
    # INCLUDE PROJECTION ONTO COMPUTATIONAL SUBSPACE IN THE MEASUREMENT
    LinearAlgebraTools.rotate!(QubitOperators.localqubitprojectors(fn.device), OT)
    # THE PROJECTION OPERATOR
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
        E = real(LinearAlgebraTools.expectation(OT, ψ));
        F = real(LinearAlgebraTools.expectation(π̄, ψ));
        E / F
    )
end

function CostFunctions.grad_function_inplace(fn::NormalizedEnergy{F}; ϕ=nothing) where {F}
    r = Integrations.nsteps(fn.grid)

    if isnothing(ϕ)
        return CostFunctions.grad_function_inplace(
            fn;
            ϕ=Array{F}(undef, r+1, Devices.ngrades(fn.device), 2)
        )
    end

    # THE PROJECTION OPERATOR, FOR COMPONENT COST FUNCTION EVALUATIONS
    π̄ = QubitOperators.localqubitprojectors(fn.device)
    # DYNAMICALLY UPDATED STATEVECTOR
    ψ = copy(fn.ψ0)

    # THE "MATRIX LIST" (A 3D ARRAY), FOR EACH GRADIENT SIGNAL
    Ō = Array{eltype(ψ)}(undef, (size(fn.O0)..., 2))
    # FIRST MATRIX: THE OBSERVABLE, IN MEASUREMENT FRAME
    OT = @view(Ō[:,:,1])
    T = Integrations.endtime(fn.grid)
    OT .= fn.O0; Devices.evolve!(fn.frame, fn.device, fn.basis, T, OT)
    # INCLUDE PROJECTION ONTO COMPUTATIONAL SUBSPACE IN THE MEASUREMENT
    LinearAlgebraTools.rotate!(π̄, OT)
    # SECOND MATRIX: PROJECTION OPERATOR, AS A GLOBAL OPERATOR
    LinearAlgebraTools.kron(π̄; result=@view(Ō[:,:,2]))

    # GRADIENT VECTORS
    ∂E = Array{F}(undef, length(fn))
    ∂N = Array{F}(undef, length(fn))

    return (∇f̄, x̄) -> (
        Parameters.bind(fn.device, x̄);
        Evolutions.evolve(
            fn.evolution,
            fn.device,
            fn.basis,
            fn.grid,
            fn.ψ0;
            result=ψ,
        );
        E = real(LinearAlgebraTools.expectation(OT, ψ));
        N = real(LinearAlgebraTools.expectation(π̄, ψ));

        Parameters.bind(fn.device, x̄);
        Evolutions.gradientsignals(
            fn.evolution,
            fn.device,
            fn.basis,
            fn.grid,
            fn.ψ0,
            Ō;
            result=ϕ,   # NOTE: This writes the gradient signal as needed.
        );
        ∂E .= Devices.gradient(fn.device, fn.grid, @view(ϕ[:,:,1]));
        ∂N .= Devices.gradient(fn.device, fn.grid, @view(ϕ[:,:,2]));

        ∇f̄ .= (∂E./N) .- (E/N) .* (∂N./N)
    )
end