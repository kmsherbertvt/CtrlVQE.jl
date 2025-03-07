module RotatingFrameEvolutions
    export ROTATING_FRAME

    import CtrlVQE: LAT
    import CtrlVQE: Integrations, Devices, Evolutions

    import CtrlVQE.Bases: DRESSED
    import CtrlVQE.Operators: STATIC, Drive

    import TemporaryArrays: @temparray

    import LinearAlgebra: norm

    """
        ROTATING_FRAME

    A Trotterization method calculating drive terms in the rotating frame of the device.

    The work basis for this algorithm is `Bases.DRESSED`.
    This ensures the rotating-frame evolution ``U_t ≡ exp(-itH_0)`` is quite cheap.
    Even so, this algorithm exponentiates the matrix ``U_t' V(t) U_t`` at each time step,
        so it is not terribly efficient.

    A `gradientsignals` method is not currently supported for this evolution algorithm.

    ```jldoctests
    julia> grid = TemporalLattice(20.0, 400);

    julia> device = Prototype(TransmonDevice{Float64,2}; n=2);

    julia> evolution = ROTATING_FRAME;

    julia> validate(evolution; grid=grid, device=device, skipgradient=true);

    julia> workbasis(evolution)
    CtrlVQE.Bases.Dressed()

    ```

    """
    struct RotatingFrameEvolution <: Evolutions.EvolutionType end
    ROTATING_FRAME = RotatingFrameEvolution()

    Evolutions.workbasis(::RotatingFrameEvolution) = DRESSED

    function Evolutions.evolve!(
        ::RotatingFrameEvolution,
        device::Devices.DeviceType,
        grid::Integrations.IntegrationType,
        ψ::AbstractVector{<:Complex{<:AbstractFloat}};
        callback=nothing,
    )
        # REMEMBER NORM FOR NORM-PRESERVING STEP
        A = norm(ψ)

        # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
        N = Devices.nstates(device)
        U = @temparray(Complex{eltype(device)}, (N,N), :framerotation)
        V = @temparray(Complex{eltype(device)}, (N,N), :driveoperator)

        # ROTATE INTO INTERACTION PICTURE
        t0 = Integrations.starttime(grid)
        ψ = Devices.evolve!(STATIC, device, DRESSED, -t0, ψ)

        # RUN EVOLUTION
        for i in eachindex(grid)
            τ = Integrations.stepat(grid, i)
            t = Integrations.timeat(grid, i)

            callback !== nothing && callback(i, t, ψ)
            U = Devices.evolver(STATIC, device, DRESSED, t; result=U)
            V = Devices.operator(Drive(t), device, DRESSED; result=V)
            V = LAT.rotate!(U', V)
            V = LAT.cis!(V, -τ)
            ψ = LAT.rotate!(V, ψ)
        end

        # ROTATE OUT OF INTERACTION PICTURE
        T = Integrations.endtime(grid)
        ψ = Devices.evolve!(STATIC, device, DRESSED, T, ψ)

        # ENFORCE NORM-PRESERVING TIME EVOLUTION
        ψ .*= A / norm(ψ)

        return ψ
    end
end