import CtrlVQE: Validation
import CtrlVQE.Validation: @withresult

import CtrlVQE.Bases: BARE, DRESSED
import CtrlVQE: Devices, CostFunctions

function Validation.validate(
    reference::ReferenceType;
    device=nothing,
)
    isnothing(device) && return

    ψB = @withresult prepare(reference, device, BARE)
    ψD = prepare(reference, device, DRESSED)
    U = Devices.basisrotation(DRESSED, BARE, device)
        @assert ψD ≈ U * ψB
end

function Validation.validate(
    measurement::MeasurementType;
    grid=nothing,
    device=nothing,
    t=0.0,
)
    nK = CostFunctions.nobservables(measurement);   @assert nK isa Int
    isnothing(device) && return

    F = eltype(device)
    N = Devices.nstates(device)
    U = Devices.basisrotation(DRESSED, BARE, device)

    # CHECK BASIS INVARIANCE OF MEASUREMENT
    ψB = LAT.basisvector(Complex{F}, N, 1)
    ψD = LAT.rotate!(U, ψB)
    EB = measure(measurement, device, BARE, ψB, t)
    ED = measure(measurement, device, DRESSED, ψD, t)
        @assert EB ≈ ED

    # CHECK BASIS INVARIANCE OF OBSERVABLES
    OB = @withresult observables(measurement, device, BARE, t)
    OD = @withresult observables(measurement, device, DRESSED, t)
    OBD = deepcopy(OB); for k in 1:nK; LAT.rotate!(U, @view(OB[:,:,k])); end
        @assert OBD ≈ OD

    # CHECK THAT `gradient` IS DEFINED
    isnothing(grid) && return
    nG = Devices.ngrades(device)
    ϕ = ones(F, length(grid), nG, nK)
    @withresult Devices.gradient(measurement, device, grid, ϕ, ψ)
end
