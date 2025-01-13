module QubitFrameEvolutions
    export QUBIT_FRAME

    import ..CtrlVQE: LAT
    import ..CtrlVQE: Integrations, Devices, Evolutions

    import ..CtrlVQE.Bases: BARE
    import ..CtrlVQE.Operators: STATIC, Drive, Gradient

    import ..CtrlVQE.TrapezoidalIntegrations: TrapezoidalIntegration

    import TemporaryArrays: @temparray

    import LinearAlgebra: norm

    """
        QUBIT_FRAME

    A Trotterization method alternately propagating static and drive terms.

    The work basis for this algorithm is `Bases.BARE`.
    The static term propagator is expensive but only computed once.
    If the drive terms are local (as for a `LocallyDrivenDevice`),
        the drive propagator is relatively cheap.

    This algorithm assumes a trapezoidal rule,
        so only `TrapezoidalIntegration` grids are allowed.

    """
    struct QubitFrameEvolution <: Evolutions.EvolutionType end
    QUBIT_FRAME = QubitFrameEvolution()

    Evolutions.workbasis(::QubitFrameEvolution) = BARE

    function Evolutions.evolve!(
        ::QubitFrameEvolution,
        device::Devices.DeviceType,
        grid::TrapezoidalIntegration,
        ψ::AbstractVector{<:Complex{<:AbstractFloat}};
        callback=nothing,
    )
        # REMEMBER NORM FOR NORM-PRESERVING STEP
        A = norm(ψ)

        # TRAPEZOIDAL-RULE SPECIFICS
        τ = Integrations.stepsize(grid)         # AVERAGE TIME STEP
        t = Ref(Integrations.starttime(grid))   # STORE PREVIOUS TIME STEP

        # RUN EVOLUTION
        for i in firstindex(grid):lastindex(grid)-1
            callback !== nothing && callback(i, t[], ψ)
            ψ = Devices.propagate!(Drive(t[]), device, BARE, τ/2, ψ)
            ψ = Devices.propagate!(STATIC, device, BARE, τ, ψ)
            t[] = Integrations.timeat(grid, i+1)        # SHIFT TIME STEP
            ψ = Devices.propagate!(Drive(t[]), device, BARE, τ/2, ψ)
        end
        callback !== nothing && callback(lastindex(grid), t[], ψ)

        #= NOTE:

        This implementation applies the drive about twice as many times as strictly necessary,
            since the latter propagation of step i can be combined with the first of i+1.
        But this symmetric form gives access to a "truer" intermediate state ψ(t).
        This doesn't matter for pure evolution, but it is meaningful for the callback,
            and more importantly to me it matches the `gradientsignals` method,
            which *needs* the true intermediate state to evaluate the gradient signal.
        For locally driven devices (which is what this evolution algorithm is designed for)
            there is no major cost to the drive propagations,
            so we can afford to favor parllel code structures.

        =#

        # ENFORCE NORM-PRESERVING TIME EVOLUTION
        ψ .*= A / norm(ψ)

        return ψ
    end


    function Evolutions.gradientsignals(
        evolution::QubitFrameEvolution,
        device::Devices.DeviceType,
        grid::TrapezoidalIntegration,
        ψ0::AbstractVector,
        Ō::LAT.MatrixList;
        result=nothing,
        callback=nothing,
    )
        # PREPARE SIGNAL ARRAYS ϕ[i,j,k]
        if result === nothing
            F = real(LAT.cis_type(ψ0))
            result = Array{F}(undef, length(grid), Devices.ngrades(device), size(Ō,3))
        end

        # PREPARE STATE AND CO-STATES
        ψTYPE = LAT.cis_type(ψ0)
        ψ = @temparray(ψTYPE, size(ψ0), :gradientsignals); ψ .= ψ0
        ψ = Evolutions.evolve!(evolution, device, grid, ψ)

        λ̄ = @temparray(ψTYPE, (size(ψ0,1), size(Ō,3)), :gradientsignals)
        for k in axes(Ō,3)
            λ̄[:,k] .= ψ
            LAT.rotate!(@view(Ō[:,:,k]), @view(λ̄[:,k]))
        end

        # TRAPEZOIDAL-RULE SPECIFICS
        τ = Integrations.stepsize(grid)         # AVERAGE TIME STEP
        t = Ref(Integrations.endtime(grid))     # STORE PREVIOUS TIME STEP

        # LAST GRADIENT SIGNALS
        callback !== nothing && callback(lastindex(grid), t[], ψ)
        for k in axes(Ō,3)
            λ = @view(λ̄[:,k])
            for j in 1:Devices.ngrades(device)
                z = Devices.braket(Gradient(j, t[]), device, BARE, λ, ψ)
                result[length(grid),j,k] = 2 * imag(z)  # ϕ[i,j,k] = -𝑖z + 𝑖z̄
            end
        end

        # ITERATE OVER TIME
        for i in reverse(firstindex(grid):lastindex(grid)-1)
            # COMPLETE THE PREVIOUS DRIVE STEP
            ψ = Devices.propagate!(Drive(t[]), device, BARE, -τ/2, ψ)
            for k in axes(Ō,3)
                λ = @view(λ̄[:,k])
                Devices.propagate!(Drive(t[]), device, BARE, -τ/2, λ)
            end

            # PROPAGATE THE STATIC HAMILTONIAN
            ψ = Devices.propagate!(STATIC, device, BARE, -τ, ψ)
            for k in axes(Ō,3)
                λ = @view(λ̄[:,k])
                Devices.propagate!(STATIC, device, BARE, -τ, λ)
            end

            t[] = Integrations.timeat(grid, i)          # SHIFT TIME STEP

            # START THE NEXT DRIVE STEP
            ψ = Devices.propagate!(Drive(t[]),   device, BARE, -τ/2, ψ)
            for k in axes(Ō,3)
                λ = @view(λ̄[:,k])
                Devices.propagate!(Drive(t[]),   device, BARE, -τ/2, λ)
            end

            # CALCULATE GRADIENT SIGNAL BRAKETS
            callback !== nothing && callback(i, t[], ψ)
            for k in axes(Ō,3)
                λ = @view(λ̄[:,k])
                for j in 1:Devices.ngrades(device)
                    z = Devices.braket(Gradient(j, t[]), device, BARE, λ, ψ)
                    result[1+i,j,k] = 2 * imag(z) # ϕ[i,j,k] = -𝑖z + 𝑖z̄
                end
            end
        end

        return result
    end
end