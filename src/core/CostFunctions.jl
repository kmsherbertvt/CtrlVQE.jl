"""
    CostFunctions

Detailed instructions for how to compute an energy from a set of variational parameters.

Note that the `CostFunctionType` defined here refers to a type, not a function.
But you can call `CostFunctions.cost_function(::CostFunctionType)` to get a function.
This is the thing you would feed into an optimizer.

"""
module CostFunctions
    include("CostFunctions/__abstractinterface.jl")
        export CostFunctionType
        export cost_function, grad!function
        # Also extends Base.length
        # Consider implementing an `EnergyFunction`, with a couple extra requirements.
    include("CostFunctions/__concreteinterface.jl")
        export grad_function
        # Also implements Base.eltype and a (::CostFunctionType)(...) signature
    include("CostFunctions/__validation.jl")

    #= The next few files implement a small extension to the `CostFunctions` interface,
        for cost functions representing an energy of a time-evolved state.

    `EnergyFunctions` could plausibly be its own module,
        but the extended interface really ought to share a namespace
        with the rest of `CostFunctions`. =#

    include("CostFunctions/__energy__abstractinterface.jl")
        export EnergyFunction
        export nobservables, trajectory_callback
        # Also extends the `CostFunctions` interface for a couple functions.
        # New energy functions must also implement the `CostFunctions` interface.
    include("CostFunctions/__energy__validation.jl")
end