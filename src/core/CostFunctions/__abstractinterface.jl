"""
    CostFunctionType{F}

Encapsulates a cost function, the thing you plug into an optimization algorithm.

# Implementation

Any concrete sub-type `CF` must implement the following methods:
- `cost_function(::CF)`:
        returns a `Function` which takes a parameter vector
            and returns the output of the cost function
        ie. a callable expression (x::Vector) -> f(x)
- `grad!function(::CF)`:
        returns a mutating `Function` which takes a gradient vector (to be mutated)
            and a parameter vector, and writes the gradient vector to the first argument.
        As a matter of habit, the resulting gradient vector should also be returned.
- `Base.length(::CF)`:
        the number of parameters this cost function takes

If your cost function involves calculating the expectation value of a time-evolved state,
    you should implement an `EnergyFunction` (even if it isn't strictly an energy).
This type has a couple extra requirements to allow energy trajectories over the evolution.

"""
abstract type CostFunctionType{F} end

"""
    cost_function(fn::CostFunctionType)::Function

Converts the struct `fn` into a literal function of a parameter vector.

The function accepts a parameter vector `x̄`
    which should have the type and length given by `eltype(fn)` and `length(fn)`.
The function returns the value of `fn` at the point `x̄`.

"""
function cost_function end

"""
    grad!function(fn::CostFunctionType)::Function

Constructs a (mutating) function to calculate the gradient of `fn` at a particular point.

The function accepts a gradient vector `∇f̄` (to be mutated) and a parameter vector `x̄`.
Both should have the type and length given by `eltype(fn)` and `length(fn)`.
After the function is called, `∇f̄` contains the gradient of `fn` at the point `x̄`.
As per the Julia guidelines on mutating functions, `∇f̄` itself should then be returned.

"""
function grad!function end

"""
    Base.length(fn::CostFunctionType)::Int

Gives the number of parameters for this cost function.

"""
function Base.length(fn::CostFunctionType)
    error("Not Implemented")
end