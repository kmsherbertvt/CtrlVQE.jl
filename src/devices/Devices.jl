"""
    Devices

*In silico* representation of quantum devices, in which quantum states evolve in time.

In this package,
    the "static" components (ie. qubit frequencies, couplings, etc.)
    and the "drive" components (ie. control signal, variational parameters, etc.)
    are *all* integrated into a single `DeviceType` object.
All you need to know how a quantum state `ψ` evolves up time `T` is in the device.

"""
module Devices
    import ..TempArrays: array
    const LABEL = Symbol(@__MODULE__)

    import ..LinearAlgebraTools
    import ..Bases, ..Operators
    import ..Parameters, ..Integrations

    using Memoization: @memoize
    using LinearAlgebra: I, Diagonal, Hermitian, Eigen, eigen

    const Evolvable = AbstractVecOrMat{<:Complex{<:AbstractFloat}}

    # Find here all the functions you need to implement for your own `DeviceType`.
    include("devicetype/__abstractinterface.jl")
    export DeviceType
    export ndrives, ngrades, nlevels, nqubits, noperators
    export localalgebra
    export qubithamiltonian, staticcoupling, driveoperator, gradeoperator
    export gradient
    # NOTE: New devices must also implement the `Parameters` interface.
    # NOTE: Consider implementing a `LocallyDrivenDevice`, with a couple extra requirements.

    Base.eltype(::DeviceType{F}) where {F} = F

    # Find here some generic elements of the interface, already implemented.
    include("devicetype/__concreteinterface.jl")
    export nstates
    export globalalgebra
    export globalize, dress, basisrotation

    #= Find here the core functionality of the `DeviceType`,
        interacting with each `OperatorType` as efficiently as possible. =#
    include("devicetype/__operator.jl")
    export operator, localqubitoperators

    include("devicetype/__propagation.jl")
    export propagator, localqubitpropagators, propagate!

    include("devicetype/__evolution.jl")
    export evolver, localqubitevolvers, evolve!

    include("devicetype/__brakets.jl")
    export expectation, braket

end # module Devices

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
    include("locallydrivendevice/__abstractinterface.jl")
    export LocallyDrivenDevice
    export drivequbit, gradequbit
    # NOTE: New devices must also implement the `Parameters` and `DeviceType` interfaces.

    # Find here some generic elements of the interface, already implemented.
    include("locallydrivendevice/__concreteinterface.jl")
    export localdriveoperators, localdrivepropagators

    # Find here more efficient implementations for some core `DeviceType` functionality.
    include("locallydrivendevice/__overrides.jl")
end