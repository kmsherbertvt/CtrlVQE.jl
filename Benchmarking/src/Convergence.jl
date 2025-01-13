"""
    Convergence

Measure convergence with increasing step count.

"""
module Convergence
    import CtrlVQE: Bases
    import CtrlVQE: Integrations, Devices, Evolutions, CostFunctions

    import NPZ
    import Plots

    import LinearAlgebra: norm

    """
        statevector(evolution, device, gridtype, ψ0; maxk=12, T=1.0)

    Evolve a statevector in time with different resolutions in the time grid.

    # Parameters
    - `evolution::EvolutionType`: algorithm to perform time evolution.
    - `device::DeviceType`: the device whose Hamiltonian the state will evolve under.
    - `gridtype::Type{<:IntegrationType}`: a concrete type for constructing the time grid.
    - `ψ0::AbstractVector`: the initial state to evolve.

    # Keyword Arguments
    - `maxk`: this function performs a trial with ``r=T⋅2^k`` time steps,
        where `k` counts up from 0 to `maxk`.
    - `T`: maximum pulse duration.
        Should be chosen to match the pulses programmed into `device`.

    Note that `k` controls a per-unit-time quantity of steps,
        so if `T` is large, this function can take awhile!

    # Returns
    - A matrix with `1+maxk` columns.
      Column `j` is the evolution of `ψ0` with ``k=j-1``, ``r=T⋅2^k`` time steps.

    """
    function statevector(
        evolution::Evolutions.EvolutionType,
        device::Devices.DeviceType{F},
        basis::Bases.BasisType,
        gridtype::Type{<:Integrations.IntegrationType{F}},
        ψ0::AbstractVector;
        maxk::Int=12,
        T::F=1.0,
    ) where {F}
        N = Devices.nstates(device)
        ψ  = Array{Complex{F}}(undef, (N, 1+maxk))
        # EXPERIMENT
        for k in 0:maxk
            r = round(Int, T * 2^k)
            grid = Integrations.Prototype(gridtype, r; T=T)
            println("Solving r=$r...")
            Evolutions.evolve(evolution, device, basis, grid, ψ0; result=@view(ψ[:,1+k]))
        end
        return ψ
    end

    """
        energy(gridtype, costfnfactory, x; maxk=12, T=1.0)

    Compute an energy function with different resolutions in the time grid.

    # Parameters
    - `gridtype::Type{<:IntegrationType}`: a concrete type for constructing the time grid.
        The `Prototype` constructor will be used to generate grids with specific `r`.
    - `costfnfactory` a callable (grid) -> (costfn) function,
        to construct a `CostFunctionType` object for each grid.
        This function must implicilty knows the `DeviceType`, reference state, etc.
    - `x::AbstractVector`: the parameters to evaluate the cost function at.

    # Keyword Arguments
    - `T`: maximum pulse duration.
    - `maxk`: this function performs a trial with ``r=T⋅2^k`` time steps,
        where `k` counts up from 0 to `maxk`.

    Note that `k` controls a per-unit-time quantity of steps,
        so if `T` is large, this function can take awhile!

    # Returns
    - An array with `1+maxk` values.
      Index `j` is the energy with ``k=j-1``, ``r=T⋅2^k`` time steps.

    """
    function energy(
        gridtype::Type{<:Integrations.IntegrationType{F}},
        costfnfactory::Function,
        x::AbstractVector{F};
        maxk::Int=12,
        T::F=1.0,
    ) where {F}
        E = Array{F}(undef, 1+maxk)
        # EXPERIMENT
        for k in 0:maxk
            r = round(Int, T * 2^k)
            grid = Integrations.Prototype(gridtype, r; T=T)
            costfn = costfnfactory(grid)
            f = CostFunctions.cost_function(costfn)
            println("Solving r=$r...")
            E[1+k] = f(x)
        end
        return E
    end

    """
        gradient(gridtype, costfnfactory, x; maxk=12, T=1.0)

    Compute a gradient with different resolutions in the time grid.

    # Parameters
    - `gridtype::Type{<:IntegrationType}`: a concrete type for constructing the time grid.
    - `costfnfactory` a callable (grid) -> (costfn) function,
        to construct a `CostFunctionType` object for each grid.
        This function implicilty knows things like the `DeviceType` and reference state.
    - `x::AbstractVector`: the parameters to evaluate the cost function at.

    # Keyword Arguments
    - `T`: maximum pulse duration.
    - `maxk`: this function performs a trial with ``r=T⋅2^k`` time steps,
        where `k` counts up from 0 to `maxk`.

    Note that `k` controls a per-unit-time quantity of steps,
        so if `T` is large, this function can take awhile!

    # Returns
    - A matrix with `1+maxk` columns.
      Column `j` is the gradient with ``k=j-1``, ``r=T⋅2^k`` time steps.

    """
    function gradient(
        gridtype::Type{<:Integrations.IntegrationType{F}},
        costfnfactory::Function,
        x::AbstractVector{F};
        maxk::Int=12,
        T::F=1.0,
    ) where {F}
        ∇f = Array{F}(undef, (length(x), 1+maxk))
        # EXPERIMENT
        for k in 0:maxk
            r = round(Int, T * 2^k)
            grid = Integrations.Prototype(gridtype, r; T=T)
            costfn = costfnfactory(grid)
            g! = CostFunctions.grad!function(costfn)
            println("Solving r=$r...")
            g!(@view(∇f[:,1+k]), x)
        end
        return ∇f
    end




    """
    """
    function analyze(
        evolution::Evolutions.EvolutionType,
        device::Devices.DeviceType{F},
        basis::Bases.BasisType,
        gridtype::Type{<:Integrations.IntegrationType{F}},
        ψ0::AbstractVector,
        costfnfactory::Function,
        x::AbstractVector{F};
        outdir::String=".",
        maxk::Int=12,
        T::F=1.0,
    ) where {F}
        # OBTAIN CONVERGENCE DATA
        outputsfile = "$outdir/outputs.npz"
        if isfile(outputsfile)
            outputs = NamedTuple(Symbol(k) => v for (k,v) in NPZ.npzread(outputsfile))
        else
            outputs = (
                ψ = statevector(evolution, device, basis, gridtype, ψ0; maxk=maxk, T=T),
                E = energy(gridtype, costfnfactory, x; maxk=maxk, T=T),
                g = gradient(gridtype, costfnfactory, x; maxk=maxk, T=T),
            )
            NPZ.npzwrite(outputsfile; outputs...)
        end

        # COMPUTE DISTANCES FOR EACH OUTPUT
        #= For the statevector, let distance be the norm of the difference vector. =#
        dψ(ψ,ψp) = norm(ψp .- ψ)
        #= For the energy, let distance be simply the absolute difference. =#
        dE(E,Ep) = abs(Ep-E)
        #= For the gradient, let distance be the RMS difference. =#
        dg(g,gp) = sqrt(sum(abs2.(gp.-g))./length(x))

        metrics = (
            ψ = F[],
            E = F[],
            g = F[],
        )

        for k in 1:maxk
            push!(metrics.ψ, dψ(outputs.ψ[:,k], outputs.ψ[:,1+k]))
            push!(metrics.E, dE(outputs.E[k], outputs.E[1+k]))
            push!(metrics.g, dg(outputs.g[:,k], outputs.g[:,1+k]))
        end

        # SAVE RESULTS
        NPZ.npzwrite("$outdir/metrics.npz"; metrics...)

        # PLOT RESULTS
        r = 2 .^ (1:maxk)
        plt = Plots.plot(;
            xlabel = "Trotter Steps per T",
            xscale = :log10,
            ylabel = "Convergence Metric",
            yscale = :log10,
            ylims = [1e-17,1e2],
            yticks = 10.0 .^ (-16:2:2),
            palette = :roma10,
            legend = :topright,
        )
        Plots.plot!(plt, r, metrics.ψ;
            color=1,
            linewidth=3,
            shape=:star,
            label="|ψₖ₋₁ - ψₖ|",
        )
        Plots.plot!(plt, r, metrics.E;
            color=4,
            linewidth=3,
            shape=:circle,
            label="|Eₖ₋₁ - Eₖ|",
        )
        Plots.plot!(plt, r, metrics.g;
            color=7,
            linewidth=3,
            shape=:square,
            label="√[∑|gₖ₋₁-gₖ|²]",
        )
        Plots.savefig(plt, "$outdir/metrics.pdf")
    end

end