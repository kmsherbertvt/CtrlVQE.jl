module ModularFramework
    import LocalCustoms: @local_    # Gives easy exports from local modules.

    include("core/__abstractinterface.jl")
        @local_ export ModularFramework: algebratype
        @local_ export ModularFramework: AlgebraType, DriftType, DriveType, LocalDrive
        @local_ export ModularFramework: ParameterMap
        @local_ export ModularFramework: sync!, map_values, map_gradients
        @local_ export ModularFramework: ReferenceType, MeasurementType
        @local_ export ModularFramework: prepare, measure, nobservables, observables

    include("core/LocalDevices.jl")
        @local_ export LocalDevices: LocalDevice
    include("core/Energies.jl")
        @local_ export Energies: Energy
    include("core/DrivePenalties.jl")
        @local_ export DrivePenalties: DrivePenalty

    ######################################################################################

    include("basics/TruncatedBosonicAlgebras.jl")
        @local_ export TruncatedBosonicAlgebras: TruncatedBosonicAlgebra
    include("basics/PauliAlgebras.jl")
        @local_ export PauliAlgebras: PauliAlgebra

    include("basics/TransmonDrifts.jl")
        @local_ export TransmonDrifts: TransmonDrift

    include("basics/DipoleDrives.jl")
        @local_ export DipoleDrives: DipoleDrive

    include("basics/DisjointMappers.jl")
        @local_ export DisjointMappers: DisjointMapper, DISJOINT
    include("basics/LinearMappers.jl")
        @local_ export LinearMappers: LinearMapper, addbasisvector!

    include("basics/KetReferences.jl")
        @local_ export KetReferences: KetReference
    include("basics/DenseReferences.jl")
        @local_ export DenseReferences: DenseReference

    include("basics/DenseMeasurements.jl")
        @local_ export DenseMeasurements: DenseMeasurement
    #= TODO: PauliMeasurement =#

    ######################################################################################

    include("Prototypes.jl")
        # This module just implements a bunch of `Devices.Prototype` methods.

end # module ModularFramework
