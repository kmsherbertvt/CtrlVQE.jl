import CtrlVQE: Validation
import CtrlVQE.Validation: @withresult

import CtrlVQE: LAT, Parameters, Devices

function Validation.validate(algebra::AlgebraType{m,n}) where {m,n}
    # CHECK TYPE PARAMETERS
    A = algebratype(algebra);               @assert A == typeof(algebra)
    @assert m == Devices.nlevels(algebra) == m
    @assert n == Devices.nqubits(algebra) == n

    # CHECK REMAINING FEATURES
    nO = Devices.noperators(algebra);       @assert nO isa Int
    ā0 = @withresult Devices.localalgebra(algebra)
        @assert size(ā0) == (m, m, nO, n)
end

function Validation.validate(drift::DriftType{A}; algebra=nothing) where {A}
    @assert A == algebratype(drift)
    isnothing(algebra) && return
    n = Devices.nqubits(algebra)

    ā0 = Devices.localalgebra(algebra)
    ā = _globalizealgebra(ā0)

    h̄ = @withresult Devices.qubithamiltonian(drift, ā, n)
    G = @withresult Devices.staticcoupling(drift, ā)
end

function Validation.validate(
    drive::DriveType{A};
    algebra=nothing,
    grid=nothing,
    t=1.0,
) where {A}
    @assert A == algebratype(drive)
    isnothing(algebra) && return

    ā0 = Devices.localalgebra(algebra)
    ā = _globalizealgebra(ā0)

    nG = Devices.ngrades(drive);            @assert nG isa Int


    v = @withresult Devices.driveoperator(drive, ā, t)
    G = @withresult Devices.gradeoperator(drive, ā, nG, t)

    if !isnothing(grid)
        ϕ = ones(eltype(grid), length(grid), nG)
        @withresult Devices.gradient(drive, grid, ϕ)
        # Just make sure it can be called. Accuracy must be checked with a cost function.
    end
end

function Validation.validate(drive::LocalDrive{A}; kwargs...) where {A}
    invoke(Validation.validate, Tuple{DriveType}, drive; kwargs...)
    @assert Devices.drivequbit(drive) isa Int
end

function Validation.validate(pmap::ParameterMap; device=nothing)
    isnothing(device) && return
    # We can't actually do anything without the device...
    F = eltype(device)
    nD = Devices.ndrives(device)
    # Actually, there really isn't much to do at all.
    # Basically just make sure everything is implemented...

    # CHECK TYPING
    namelist = Parameters.names(pmap, device);          @assert eltype(namelist) == String
    x = @withresult map_values(pmap, device, nD);       @assert F == eltype(x)
    g = @withresult map_gradients(pmap, device, nD);    @assert F == eltype(g)

    # CHECK MUTATING FUNCTION RETURNS MUTATED OBJECT
    device_ = sync!(pmap, device);                      @assert device_ == device
end


function _globalizealgebra(ops)
    m = size(ops, 1)
    o = size(ops, 3)
    n = size(ops, 4)
    N = m^n

    result = Array{eltype(ops)}(undef, N, N, o, n)
    for q in 1:n
        for σ in 1:o
            LAT.globalize(@view(ops[:,:,σ,q]), n, q; result=@view(result[:,:,σ,q]))
        end
    end
    return result
end
