using .Devices: DeviceType, LocallyDrivenDevice

import ..CtrlVQE: Validation

function Validation.validate(
    device::LocallyDrivenDevice{F};
    grid=nothing,
    t=zero(F),
) where {F}
    invoke(Validation.validate, Tuple{DeviceType}, device; grid=grid, t=t)

    # REPRODUCE NAMESPACE FROM INVOCATION
    nD = ndrives(device)
    nG = ngrades(device)
    nO = noperators(device)
    n = nqubits(device)
    τ = one(F)

    ā = globalalgebra(device)
    ā0 = localalgebra(device)
    for o in 1:nO
        for q in 1:n
            @assert ā[:,:,o,q] ≈ LAT.globalize(@view(ā0[:,:,o,q]), n, q)
        end
    end
    v̄ = [driveoperator(device, ā, i, t) for i in 1:nD]
    Ā = [gradeoperator(device, ā, j, t) for j in 1:nG]

    # MODEL METHODS
    v̄0 = [driveoperator(device, ā0, i, t) for i in 1:nD]
    for i in 1:nD
        q = drivequbit(device, i)
        @assert v̄[i] ≈ LAT.globalize(v̄0[i], n, q)
    end

    Ā0 = [gradeoperator(device, ā0, j, t) for j in 1:nG]
    for j in 1:nG
        q = gradequbit(device, j)
        @assert Ā[j] ≈ LAT.globalize(Ā0[j], n, q)
    end

    # LOCAL OPERATORS
    v̄L = @withresult localdriveoperators(device, t)
        @assert all(v̄L[:,:,q] ≈ v̄0[q] for q in 1:n)
    ūL = @withresult localdrivepropagators(device, τ, t)
        @assert all(ūL[:,:,q] ≈ cis((-τ).*v̄L[:,:,q]) for q in 1:n)
end