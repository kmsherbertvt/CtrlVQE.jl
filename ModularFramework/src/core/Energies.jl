module Energies
    import ..ModularFramework as Modular
    import ..ModularFramework: ReferenceType, MeasurementType

    import CtrlVQE
    import CtrlVQE.LinearAlgebraTools as LAT
    import CtrlVQE: Parameters
    import CtrlVQE: Devices, Evolutions, CostFunctions
    import CtrlVQE.Integrations: IntegrationType
    import CtrlVQE.Devices: DeviceType, gradient
    import CtrlVQE.Evolutions: EvolutionType

    struct Energy{
        F,
        E <: EvolutionType,
        D <: DeviceType{F},
        G <: IntegrationType{F},
        R <: ReferenceType,
        M <: MeasurementType,
    } <: CostFunctions.EnergyFunction{F}
        evolution::E
        device::D
        grid::G
        reference::R
        measurement::M
    end

    Base.length(costfn::Energy) = Parameters.count(costfn.device)

    function CostFunctions.nobservables(costfn::Energy)
        return CostFunctions.nobservables(costfn.measurement)
    end

    function CostFunctions.trajectory_callback(
        costfn::Energy,
        E::AbstractVector;
        callback=nothing,
    )
        workbasis = Evolutions.workbasis(costfn.evolution)
        return (i, t, ψ) -> (
            E[1+i] = Modular.measure(costfn.measurement, costfn.device, workbasis, ψ, t);
            isnothing(callback) || callback(i, t, ψ)
        )
    end

    function CostFunctions.cost_function(costfn::Energy; callback=nothing)
        workbasis = Evolutions.workbasis(costfn.evolution)
        ψ0 = Modular.prepare(costfn.reference, costfn.device, workbasis)
                                                        # Initial state, in the workbasis.
        ψ = copy(ψ0)                                    # Result array for evolution.
        T = last(costfn.grid)                           # Time when measurement occurs.

        return (x) -> (
            Parameters.bind!(costfn.device, x);
            Evolutions.evolve(
                costfn.evolution,
                costfn.device,
                workbasis,
                costfn.grid,
                ψ0;
                result=ψ,
                callback=callback,
            );
            Modular.measure(costfn.measurement, costfn.device, workbasis, ψ, T)
        )
    end

    function CostFunctions.grad!function(costfn::Energy; ϕ=nothing)
        nT = length(costfn.grid)
        nG = CtrlVQE.ngrades(costfn.device)
        nK = Modular.nobservables(costfn.measurement)
        isnothing(ϕ) && (ϕ = Array{eltype(costfn)}(undef, (nT, nG, nK)))

        workbasis = Evolutions.workbasis(costfn.evolution)
        ψ0 = Modular.prepare(costfn.reference, costfn.device, workbasis)
                                                        # Initial state, in the workbasis.
        T = last(costfn.grid)                           # Time when measurement occurs.s
        Ō = Modular.observables(costfn.measurement, costfn.device, workbasis, T)
                                                        # Observables, in the workbasis.

        # ADD IN A CALLBACK TO RECORD THE COMPLETELY EVOLVED STATE
        ψT = similar(ψ0)
        U = Devices.basisrotation(costfn.measurement.basis, workbasis, costfn.device)
        saveevolvedstate = (i, t, ψ) -> (
            i == lastindex(costfn.grid) || return;  # Only consider ψ at end of grid.
            ψT .= ψ;                                # Copy state into ψT.
            LAT.rotate!(U, ψT);                     # Rotate into measurement basis.
        )

        return (∇f, x) -> (
            Parameters.bind!(costfn.device, x);
            Evolutions.gradientsignals(
                costfn.evolution,
                costfn.device,
                workbasis,
                costfn.grid,
                ψ0,
                Ō;
                result=ϕ,
                callback=saveevolvedstate,
            );
            Devices.gradient(
                costfn.measurement, costfn.device, costfn.grid, ϕ, ψT;
                result=∇f,
            )
        )
    end
end