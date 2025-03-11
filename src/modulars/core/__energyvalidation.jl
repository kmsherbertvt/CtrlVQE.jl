import CtrlVQE: Validation
import CtrlVQE.Validation: @withresult

import CtrlVQE: BARE, DRESSED
import CtrlVQE: Bases, Operators, Devices, CostFunctions

function Validation.validate(
    reference::ReferenceType;
    device=nothing,
)
    basis = initbasis(reference);                   @assert basis isa Bases.BasisType
    isnothing(device) && return

    # CHECK DEFAULTS
    ψ = @withresult prepare(reference, device)
        @assert ψ ≈ @withresult prepare(reference, device, basis)

    # CHECK BASIS INVARIANCE
    ψB = prepare(reference, device, BARE)
    ψD = prepare(reference, device, DRESSED)
    U = Devices.basisrotation(DRESSED, BARE, device)
        @assert ψD ≈ U * ψB
end

function Validation.validate(
    measurement::MeasurementType;
    grid=nothing,
    device=nothing,
    t=1.0,
)
    basis = initbasis(measurement);             @assert basis isa Bases.BasisType
    frame = initframe(measurement);             @assert frame isa Operators.StaticOperator
    nK = CostFunctions.nobservables(measurement);   @assert nK isa Int
    isnothing(device) && return

    F = eltype(device)
    N = Devices.nstates(device)
    U = Devices.basisrotation(DRESSED, BARE, device)
    ψB = LAT.basisvector(Complex{F}, N, 1)
    ψD = LAT.rotate!(U, ψB)

    # CHECK DEFAULTS
    E = measure(measurement, device, ψB)
        @assert E ≈ measure(measurement, device, ψB, 0.0)
        @assert E ≈ measure(measurement, device, basis, ψB)
        @assert E ≈ measure(measurement, device, basis, ψB, 0.0)

    O = @withresult observables(measurement, device)
        @assert O ≈ @withresult observables(measurement, device, 0.0)
        @assert O ≈ @withresult observables(measurement, device, basis)
        @assert O ≈ @withresult observables(measurement, device, basis, 0.0)

    # CHECK FRAME ROTATION
    O_ = deepcopy(O)
    for k in 1:nK; Devices.evolve!(frame, device, basis, t, @view(O_[:,:,k])); end
        @assert O_ ≈ observables(measurement, device, t)

    # CHECK BASIS INVARIANCE OF MEASUREMENT
    EB = measure(measurement, device, BARE, ψB, t)
    ED = measure(measurement, device, DRESSED, ψD, t)
        @assert EB ≈ ED

    # CHECK BASIS INVARIANCE OF OBSERVABLES
    OB = observables(measurement, device, BARE, t)
    OD = observables(measurement, device, DRESSED, t)
    OBD = deepcopy(OB); for k in 1:nK; LAT.rotate!(U, @view(OBD[:,:,k])); end
        @assert OBD ≈ OD

    # CHECK THAT `gradient` IS DEFINED
    isnothing(grid) && return
    nG = Devices.ngrades(device)
    ϕ = ones(F, length(grid), nG, nK)
    @withresult Devices.gradient(measurement, device, grid, ϕ, ψ)
end
