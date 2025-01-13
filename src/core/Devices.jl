"""
    Devices

*In silico* representation of quantum devices, in which quantum states evolve in time.

In this package,
    the "static" components (e.g., qubit frequencies, couplings, etc.)
    and the "drive" components (e.g., control signal, variational parameters, etc.)
    are *all* integrated into a single `DeviceType` object.
All you need to know how a quantum state `ψ` evolves up time `T` is in the device.

"""
module Devices
    const Evolvable = AbstractVecOrMat{<:Complex{<:AbstractFloat}}

    include("Devices/__abstractinterface.jl")
        export DeviceType
        export ndrives, ngrades, nlevels, nqubits, noperators
        export localalgebra
        export qubithamiltonian, staticcoupling, driveoperator, gradeoperator
        export gradient
        export Prototype
        # New devices must also implement the `Parameters` interface.
        # Consider implementing a `LocallyDrivenDevice`, with a couple extra requirements.
    include("Devices/__concreteinterface.jl")
        export nstates
        export globalalgebra
        export globalize, dress, basisrotation
        # Also implements Base.eltype
    include("Devices/__opfun__operator.jl")
        export operator, localqubitoperators
    include("Devices/__opfun__propagation.jl")
        export propagator, localqubitpropagators, propagate!
    include("Devices/__opfun__evolution.jl")
        export evolver, localqubitevolvers, evolve!
        # NOTE: `Devices.evolve!` and `Evolutions.evolve!` are distinct functions!
    include("Devices/__opfun__brakets.jl")
        export expectation, braket
    include("Devices/__validation.jl")

    #= The next few files implement a small extension to the `Devices` interface,
        for devices in which each drive term is local to a single qubit in the bare basis.

    `LocallyDrivenDevices` could plausibly be its own module,
        but the extended interface really ought to share a namespace
        with the rest of `Devices`. =#

    include("Devices/__local__abstractinterface.jl")
        export LocallyDrivenDevice
        export drivequbit, gradequbit
        # New local devices must also implement the `Parameters`, `DeviceType` interfaces.
    include("Devices/__local__concreteinterface.jl")
        export localdriveoperators, localdrivepropagators
    include("Devices/__local__overrides.jl")
    include("Devices/__local__validation.jl")

end # module Devices
