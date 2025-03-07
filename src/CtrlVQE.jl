module CtrlVQE
    import LocalCustoms: @local_    # Gives easy exports from local modules.

    # UTILITIES
    include("utils/Quples.jl")
        @local_ export Quples: Quple
    include("utils/LinearAlgebraTools.jl")
        import .LinearAlgebraTools as LAT
        export LAT
    include("utils/QubitProjections.jl")
        @local_ export QubitProjections

    # ENUMERATIONS
    include("enums/Parameters.jl")
        @local_ export Parameters
    include("enums/Validation.jl")
        @local_ export Validation: validate
        import .Validation: @withresult
        export @withresult
    include("enums/Prototypes.jl")
        @local_ export Prototypes: Prototype
    include("enums/Operators.jl")
        @local_ export Operators
    include("enums/Bases.jl")
        @local_ export Bases

    # CORE INTERFACE
    include("core/Integrations.jl")
        @local_ export Integrations
        @local_ export Integrations: nsteps, duration, stepsize, integrate
    include("core/Signals.jl")
        @local_ export Signals
        @local_ export Signals: valueat, partial
        @local_ export Signals: Constrained
    include("core/Devices.jl")
        @local_ export Devices
        @local_ export Devices: ndrives, ngrades, nlevels, nqubits, nstates, noperators
        @local_ export Devices: drivequbit, gradequbit
    include("core/Evolutions.jl")
        @local_ export Evolutions
        @local_ export Evolutions: workbasis, evolve, evolve!, gradientsignals
    include("core/CostFunctions.jl")
        @local_ export CostFunctions
        @local_ export CostFunctions: cost_function, grad_function, grad!function
        @local_ export CostFunctions: nobservables, trajectory_callback
    #= Note that only the generically useful names have been exported
        from each of these modules. =#

    # BASICS
    include("basics/Integrations/TrapezoidalIntegrations.jl")
        @local_ export TrapezoidalIntegrations: TrapezoidalIntegration, TemporalLattice

    include("basics/Signals/ConstantSignals.jl");
        @local_ export ConstantSignals: ConstantSignal, ComplexConstantSignal, Constant
    include("basics/Signals/WindowedSignals.jl")
        @local_ export WindowedSignals: WindowedSignal, Windowed
    include("basics/Signals/CompositeSignals.jl")
        @local_ export CompositeSignals: CompositeSignal, Composed

    include("basics/Devices/RealWindowedResonantTransmonDevices.jl")
        @local_ export RealWindowedResonantTransmonDevices: RWRTDevice
    include("basics/Devices/WindowedResonantTransmonDevices.jl")
        @local_ export WindowedResonantTransmonDevices: CWRTDevice
    include("basics/Devices/TransmonDevices.jl")
        @local_ export TransmonDevices: TransmonDevice

    include("basics/Evolutions/RotatingFrameEvolutions.jl")
        @local_ export RotatingFrameEvolutions: ROTATING_FRAME
    include("basics/Evolutions/QubitFrameEvolutions.jl")
        @local_ export QubitFrameEvolutions: QUBIT_FRAME

    include("basics/CostFunctions/DenseLeakageFunctions.jl")
        @local_ export DenseLeakageFunctions: DenseLeakage
    include("basics/CostFunctions/DenseObservableFunctions.jl")
        @local_ export DenseObservableFunctions: DenseObservable
    include("basics/CostFunctions/CompositeCostFunctions.jl")
        @local_ export CompositeCostFunctions: CompositeCostFunction
    include("basics/CostFunctions/WindowedResonantPenalties.jl")
        @local_ export WindowedResonantPenalties: WindowedResonantPenalty
    include("basics/CostFunctions/SignalStrengthPenalties.jl")
        @local_ export SignalStrengthPenalties: SignalStrengthPenalty
    include("basics/CostFunctions/AmplitudePenalties.jl")
        @local_ export AmplitudePenalties: AmplitudePenalty
    include("basics/CostFunctions/DetuningPenalties.jl")
        @local_ export DetuningPenalties: DetuningPenalty

    # MODULAR FRAMEWORK
    include("modulars/ModularFramework.jl")
        @local_ export ModularFramework

end
