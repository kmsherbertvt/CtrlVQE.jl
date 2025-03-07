"""
    ModularFramework

Provides a framework for designing devices and energy functions in a less monolithic way.
Different behaviors are delegated to different types.

Behavior for devices are divided into:
- `AlgebraType`
- `DriftType`
- `DriveType`
- `ParameterMap`
These can be implemented by the user and then combined into a pre-existing concrete type,
    usually `LocalDevice`.

Behavior for energy functions are divided into:
- `ReferenceType`
- `MeasurementType`
These can be implemented by the user and then combined
    with an integration, device, and evolution
    using the pre-existing concrete type `Energy`.

Each type has a `validate` method defined, with the following signatures.
(See the function's docstring for details.)

    validate(::AlgebraType)
    validate(::DriftType; algebra::AlgebraType)
    validate(::DriveType; algebra::AlgebraType, grid::IntegrationType, t)
    validate(::ParameterMap; device::DeviceType)

    validate(::ReferenceType; device::DeviceType)
    validate(::MeasurementType; grid::IntegrationType, device::DeviceType, t)

"""
module ModularFramework
    import LocalCustoms: @local_    # Gives easy exports from local modules.

    # DEVICE MODULES
    include("core/__deviceinterface.jl")
        @local_ export ModularFramework: algebratype
        @local_ export ModularFramework: AlgebraType, DriftType, DriveType, LocalDrive
        @local_ export ModularFramework: ParameterMap
        @local_ export ModularFramework: sync!, map_values, map_gradients
    include("core/LocalDevices.jl")
        @local_ export LocalDevices: LocalDevice
    include("core/__devicevalidation.jl")

    # ENERGY MODULES
    include("core/__energyinterface.jl")
        @local_ export ModularFramework: ReferenceType, MeasurementType
        @local_ export ModularFramework: prepare, measure, observables
    include("core/Energies.jl")
        @local_ export Energies: Energy
    include("core/DrivePenalties.jl")
        @local_ export DrivePenalties: DrivePenalty
    include("core/__energyvalidation.jl")

    #= TODO:
    - import ...CtrlVQE in basics? If it were an issue, it would not have compiled!
      Perhaps we actually want to get RID of the .. in src/basics..!
      An in modulars/basics, use import CtrlVQE.ModularFramework..?
      Standardize throughout. If we can just *not* use relative imports, that'd be great.
    - I get that we can't really define Prototypes for these modular types until basic types are defined. But we CAN define a ModularFramework.Prototype function with an interface for each type. For that matter, why do we not have just one CtrlVQE.Prototype function? It can be an enumeration just like Validate..? We had considered it before, but not its parallelism with validation. Let's go ahead and do that...
    =#

    ######################################################################################
    #= BASIC IMPLEMENTATIONS =#

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
    #= PROTOTYPES =#

    include("Prototypes.jl")
        # This module just implements a bunch of `Devices.Prototype` methods.

end # module ModularFramework
