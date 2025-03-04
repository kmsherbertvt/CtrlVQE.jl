import ..CtrlVQE: Validation
import ..CtrlVQE.Validation: @withresult

import ..CtrlVQE.LinearAlgebraTools as LAT

import ..CtrlVQE.Bases: BasisType, BARE, DRESSED
import ..CtrlVQE: Devices

function Validation.validate(
    evolution::EvolutionType;
    grid=nothing, device=nothing,
    skipgradient=false,
)
    # CHECK `workbasis`
    basis = workbasis(evolution);                   @assert basis isa BasisType

    # Well...that's all we can do without the device...
    (isnothing(grid) || isnothing(device)) && return
    N = Devices.nstates(device)
    arr_type = Array{Complex{eltype(device)}}
    ψ = convert(arr_type, LAT.basisvector(N,1))     # Dummy reference state.
    I = convert(arr_type, LAT.basisvectors(N))      # Dummy single observable.
    Ō = reshape(I, N, N, 1)                         # Dummy multi-observables.

    # SET UP CALLBACK TESTING
    counter = Ref(0)
    count! = (i,t,ψ) -> (counter[] += 1)


    # CHECK THAT `evolve` MATCHES `evolve!`
    ψ_ = copy(ψ); evolve!(evolution, device, grid, ψ_; callback=count!)
    ψ__ = @withresult evolve(evolution, device, grid, ψ)
        @assert ψ_ ≈ ψ__
        @assert counter[] == length(grid)

    # CHECK THAT EVOLUTION MATCHES IN DIFFERENT BASES
    U = Devices.basisrotation(DRESSED, BARE, device)
    ψB = evolve(evolution, device, BARE, grid, ψ)
    ψD = evolve(evolution, device, DRESSED, grid, U*ψ)
        @assert U * ψB ≈ ψD

    if !skipgradient
        # CHECK THE GRADIENT SIGNAL MATCHES WITH DIFFERENT OBSERVABLE SHAPES
        counter[] = 0
        ϕ̄ = gradientsignals(evolution, device, grid, ψ, Ō; callback=count!)
        ϕ = @withresult gradientsignals(evolution, device, grid, ψ, I)
            @assert ϕ ≈ reshape(ϕ̄, length(grid), :)
            @assert counter[] == length(grid)

        # CHECK THE GRADIENT SIGNAL IS INDEPENDENT OF BASIS
        ϕB = gradientsignals(evolution, device, BARE, grid, ψ, Ō)
        ϕD = gradientsignals(evolution, device, DRESSED, grid, U*ψ, I)
            @assert ϕB ≈ ϕD
    end
end
