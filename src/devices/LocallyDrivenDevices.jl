"""
    LocallyDrivenDevices

A small extension to the `Devices` interface,
    for devices in which each drive term is local to a single qubit in the bare basis.

"""
module LocallyDrivenDevices
    import ..TempArrays: array
    const LABEL = Symbol(@__MODULE__)

    import ..Devices
    import ..Devices: Evolvable
    using ..Devices

    import ..LinearAlgebraTools
    import ..Bases, ..Operators

    using Memoization: @memoize
    using LinearAlgebra: I, Diagonal, Hermitian, Eigen, eigen

    # Find here all the functions you need to implement for your own `LocallyDrivenDevice`.
    include("__local__abstractinterface.jl")
    export LocallyDrivenDevice
    export drivequbit, gradequbit
    # NOTE: New devices must also implement the `Parameters` and `DeviceType` interfaces.

    # Find here some generic elements of the interface, already implemented.
    include("__local__concreteinterface.jl")
    export localdriveoperators, localdrivepropagators

    # Find here more efficient implementations for some core `DeviceType` functionality.
    include("__local__overrides.jl")
end