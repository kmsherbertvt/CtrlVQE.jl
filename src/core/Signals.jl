"""
    Signals

Time-dependent functions suitable for control signals with variational parameters.

The main motivation of this module
    is to provide a common interface for analytical gradients and optimization.

"""
module Signals
    include("Signals/__abstractinterface.jl")
        export SignalType
        export valueat, partial
        # Also extends Base.string
        # New signals must also implement the `Parameters` interface.
        # Consider implementing a `ParametricSignal`, which does most of the work for you.
    include("Signals/__concreteinterface.jl")
        export parametertype, returntype
        # Implements a (::SignalType)(...) signature
        # Implements certain method signatures from the abstract interface.
    include("Signals/__validation.jl")

    #= The next few files implement a small extension to the `Signals` interface,
        for signals implemented by a simple mutable struct.

    `ParametricSignals` could plausibly be its own module,
        but the extended interface really ought to share a namespace
        with the rest of `Signals`. =#

    include("Signals/__parametric__abstractinterface.jl")
        export ParametricSignal
        export parameters
        # New parametric signals must also implement the `Signals` interface.
        # BUT they do NOT need to implement the `Parameters` interface!
    include("Signals/__parametric__concreteinterface.jl")
        # Implements the `Parameters` interface for `ParametricSignals`.
    include("Signals/__parametric__validation.jl")

    #= The next file has the honor of introducing the only concrete type
        in the "core" of CtrlVQE.
    It's a clever bit of code that lets you freeze arbitrary parameters
        in a `ParametricSignal`,
        saving you from having to write a custom signal
        for every application under the sun.

    If there were a `ParametricSignals` module, this would be a part of it. =#

    include("Signals/__constrainedsignals.jl")
        export ConstrainedSignal
        export Constrained
end
