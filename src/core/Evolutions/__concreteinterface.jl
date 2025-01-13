using .Evolutions: EvolutionType
using .Evolutions: workbasis

import ..CtrlVQE: LAT
import ..CtrlVQE: Bases
import ..CtrlVQE: Integrations, Devices, Evolutions

import TemporaryArrays: @temparray

"""
    evolve(evolution, device, grid, ψ0; result=nothing, kwargs...)
    evolve(evolution, device, basis, grid, ψ0; result=nothing, kwargs...)

Evolve a quantum computational state under a device Hamiltonian.

If `basis` is provided, `ψ0` is taken to be represented in that basis.
Otherwise, the workbasis of `evolution` is assumed.

This method simply copies `ψ0` (to `result` if provided, or else to a new array),
    then calls the mutating function `evolve!` on the copy.
Please see `evolve!` for detailed documentation.

"""
function evolve end

function evolve(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    grid::Integrations.IntegrationType,
    ψ0::AbstractVector;
    result=nothing,
    kwargs...
)
    isnothing(result) && (result = Vector{LAT.cis_type(ψ0)}(undef, length(ψ0)))
    result .= ψ0
    return Evolutions.evolve!(evolution, device, grid, result; kwargs...)
end

function evolve(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    grid::Integrations.IntegrationType,
    ψ0::AbstractVector;
    result=nothing,
    kwargs...
)
    isnothing(result) && (result = Vector{LAT.cis_type(ψ0)}(undef, length(ψ0)))
    result .= ψ0
    return Evolutions.evolve!(evolution, device, basis, grid, result; kwargs...)
end



#= Implement certain method signatures from the abstract interface. =#

function Evolutions.evolve!(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    grid::Integrations.IntegrationType,
    ψ0::AbstractVector;
    kwargs...
)
    U = Devices.basisrotation(workbasis(evolution), basis, device)
    ψ0 = LAT.rotate!(U, ψ0)      # ROTATE INTO WORK BASIS
    ψ0 = Evolutions.evolve!(evolution, device, grid, ψ0; kwargs...)
    ψ0 = LAT.rotate!(U', ψ0)     # ROTATE BACK INTO GIVEN BASIS
    return ψ0
end

function Evolutions.gradientsignals(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    basis::Bases.BasisType,
    grid::Integrations.IntegrationType,
    ψ::AbstractVector,
    Ō::Union{LAT.MatrixList,AbstractMatrix};
    kwargs...
)
    # FETCH THE BASIS ROTATION
    U = Devices.basisrotation(workbasis(evolution), basis, device)

    # ALLOCATE NEW ARRAYS SO WE DON'T MUTATE ψ AND Ō
    ψ_ = @temparray(eltype(ψ), size(ψ), :gradientsignals); ψ_ .= ψ
    Ō_ = @temparray(eltype(Ō), size(Ō), :gradientsignals); Ō_ .= Ō

    # ROTATE INTO WORK BASIS
    ψ_ = LAT.rotate!(U, ψ_)
    if Ō isa LAT.MatrixList     # TAKE CARE IF Ō IS MULTI-DIMENSIONAL
        for k in axes(Ō,3)
            LAT.rotate!(U, @view(Ō_[:,:,k]))
        end
    else
        LAT.rotate!(U, Ō_)
    end

    # CALL BASIS-LESS VERSION
    return Evolutions.gradientsignals(evolution, device, grid, ψ_, Ō_; kwargs...)
end

function Evolutions.gradientsignals(
    evolution::EvolutionType,
    device::Devices.DeviceType,
    grid::Integrations.IntegrationType,
    ψ0::AbstractVector,
    O::AbstractMatrix;
    result=nothing,
    kwargs...
)
    # `O` AND `result` GIVEN AS 2D ARRAYS BUT MUST BE 3D FOR DELEGATION
    !isnothing(result) && (result = reshape(result, size(result)..., 1))
    Ō = reshape(O, size(O)..., 1)

    # PERFORM THE DELEGATION
    result = Evolutions.gradientsignals(
        evolution, device, grid, ψ0, Ō;
        result=result, kwargs...
    )   # NOTE: `result` may or may not have started nothing, but it is now something.

    # NOW RESHAPE `result` BACK TO 2D ARRAY
    result = reshape(result, size(result, 1), size(result, 2))
    return result
end