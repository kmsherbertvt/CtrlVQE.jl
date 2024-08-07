module ModularEnergies
    import ...CostFunctions
    import ..Algebras

    import ...LinearAlgebraTools
    import ...Parameters, ...Integrations, ...Devices, ...Evolutions
    import ...Bases, ...Operators

    import ...TrapezoidalIntegrations: TrapezoidalIntegration

    import ..ModularDevice
    import ..PreparationProtocolType, ..initialstate
    import ..MeasurementProtocolType, ..measure, ..observables, ..gradient

    struct ModularEnergy{F} <: CostFunctions.EnergyFunction{F}
        evolution::Evolutions.EvolutionType
        grid::TrapezoidalIntegration{F}
        device::ModularDevice                   # Must be eltype F
        preparer::PreparationProtocolType       # Algebra must match device
        measurer::MeasurementProtocolType       # Algebra must match device

        function ModularEnergy(
            evolution::Evolutions.EvolutionType,
            grid::TrapezoidalIntegration{F},
            device::ModularDevice,
            preparer::PreparationProtocolType,
            measurer::MeasurementProtocolType,
        ) where {F}
            @assert F == eltype(device)

            @assert Algebras.algebratype(device) == Algebras.algebratype(preparer)
            @assert Algebras.algebratype(device) == Algebras.algebratype(measurer)

            return new{F}(evolution, grid, device, preparer, measurer)
        end
    end

    Algebras.algebratype(fn::ModularEnergy) = Algebras.algebratype(fn.device)

    function Base.length(fn::ModularEnergy)
        return Parameters.count(fn.device)
    end

    function CostFunctions.trajectory_callback(
        fn::ModularEnergy,
        E::AbstractVector;
        callback=nothing,
    )
        workbasis = Evolutions.workbasis(fn.evolution)
        return (i, t, ψ) -> (
            E[i] = measure(fn.measurer, fn.device, workbasis, ψ, t);
            !isnothing(callback) && callback(i, t, ψ)
        )
    end

    function CostFunctions.cost_function(fn::ModularEnergy; callback=nothing)
        x̄0 = Vector{eltype(fn.device)}(undef, Parameters.count(fn.device))

        workbasis = Evolutions.workbasis(fn.evolution)
        T = Integrations.endtime(fn.grid)               # Time when measurement occurs.
        ψ0 = initialstate(fn.preparer, fn.device, workbasis)
                                                        # Initial state, in the workbasis.
        ψ = copy(ψ0)                                    # Result array for evolution.

        return (x̄) -> (
            x̄0 .= Parameters.values(fn.device);
            Parameters.bind!(fn.device, x̄);
            Evolutions.evolve(
                fn.evolution,
                fn.device,
                workbasis,
                fn.grid,
                ψ0;
                result=ψ,
                callback=callback,
            );
            Parameters.bind!(fn.device, x̄0);
            measure(fn.measurer, fn.device, workbasis, ψ, T);
        )
    end

    function CostFunctions.grad_function_inplace(fn::ModularEnergy{F}; ϕ=nothing) where {F}
        r = Integrations.nsteps(fn.grid)
        nG = Devices.ngrades(fn.device)
        nK = nobservables(fn.measurer)
        isnothing(ϕ) && (ϕ = Array{F}(undef, (r+1, nG, nK)))

        x̄0 = Vector{eltype(fn.device)}(undef, Parameters.count(fn.device))

        workbasis = Evolutions.workbasis(fn.evolution)
        T = Integrations.endtime(fn.grid)               # Time when measurement occurs.
        ψ0 = initialstate(fn.preparer, fn.device, workbasis)
                                                        # Initial state, in the workbasis.
        Ō = observables(fn.measurer, fn.device, workbasis, T)
                                                        # Observables, in the workbasis.

        return (∇f̄, x̄) -> (
            x̄0 .= Parameters.values(fn.device);
            Parameters.bind!(fn.device, x̄);
            Evolutions.gradientsignals(
                fn.evolution,
                fn.device,
                workbasis,
                fn.grid,
                ψ0,
                Ō;
                result=ϕ,
            );
            Parameters.bind!(fn.device, x̄0);
            gradient(fn.measurer, fn.device, fn.grid, ϕ; ∇f̄)
        )
    end
end