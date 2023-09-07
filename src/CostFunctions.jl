import LinearAlgebra: norm

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

"""
    CostFunctionType{F}

Super-type for "cost functions", to be plugged directly into optimization algorithms.

# Implementation

Any concrete sub-type `CF` must implement the following methods:
- `Base.length(::CF)`:
        the number of parameters this cost function takes
- `cost_function(::CF)`:
        returns a `Function` which takes a parameter vector
            and returns the output of the cost function
        ie. a callable expression (x::Vector) -> f(x)
- `grad_function(::CF)`:
        returns a mutating `Function` which takes a gradient vector (to be mutated)
            and a parameter vector, and writes the gradient vector to the first argument.
        As a matter of habit, the resulting gradient vector should also be returned.

"""
abstract type CostFunctionType{F} end

"""
    Base.length(fn::CostFunctionType)

Gives the number of parameters for this cost function.
"""
function Base.length(fn::CostFunctionType)
    error("Not Implemented")
    return 0
end

"""
    cost_function(fn::CostFunctionType)

Converts the struct `fn` into a literal `Function` of a parameter vector.

The function accepts a parameter vector `x̄`
    which should have the type and length given by `eltype(fn)` and `length(fn)`.
The function returns the value of `fn` at the point `x̄`.

"""
function cost_function(fn::CostFunctionType{F}) where {F}
    error("Not Implemented")
    return (x̄) -> zero(F)
end

"""
    grad_function(fn::CostFunctionType)

Constructs a (mutating) `Function` to calculate the gradient of `fn` at a particular point.

The function accepts a gradient vector `∇f̄` (to be mutated) and a parameter vector `x̄`.
Both should have the type and length given by `eltype(fn)` and `length(fn)`.
After the function is called, `∇f̄` contains the gradient of `fn` at the point `x̄`.

"""
function grad_function(fn::CostFunctionType{F}) where {F}
    error("Not Implemented")
    return (∇f̄, x̄) -> Vector{F}(undef, length(fn))
end



# IMPLEMENTED INTERFACE

"""
    Base.eltype(fn::CostFunctionType)

Gives the number type for parameters of this cost function.
"""
function Base.eltype(::CostFunctionType{F}) where {F}
    return F
end

"""
    (fn::CostFunctionType)(x̄::AbstractVector)

Evaluate the value of `fn` at the point `x̄`.

This is syntactic sugar for constructing a dedicated `cost_function` and calling it.
It will not normally take advantage of cached work variables,
    so avoid using it in high-performance code.

"""
function (fn::CostFunctionType)(x̄::AbstractVector)
    return cost_function(fn)(x̄)
end

"""
    grad_function_byvalue(fn::CostFunctionType)

Constructs a `Function` to calculate the gradient of `fn` at a particular point.

The function accepts a parameter vector `x̄`
    which should have the type and length given by `eltype(fn)` and `length(fn)`.
The function returns the gradient (a vector) of `fn` at the point `x̄`.

"""
function grad_function_byvalue(fn::CostFunctionType{F}) where {F}
    gd = grad_function(fn)
    ∇f̄ = Vector{F}(undef, length(fn))
    return (x̄) -> (gd(∇f̄, x̄); ∇f̄)
end





"""
    CompositeCostFunction(components::AbstractVector{CostFunctionType})

The sum of several cost-functions.

Use this eg. to combine an energy function with one or more penalty functions.

Note that all components must have matching `eltype` and `length`.

As a practical convenience, this struct maintains some dynamic attributes
    so that results for each component function can be inspected
    in the middle of an optimization.
I don't really consider them an official part of the interface,
    but feel free to use them. Although...they aren't technically "thread-safe". ^_^

"""
struct CompositeCostFunction{F} <: CostFunctionType{F}
    components::Vector{CostFunctionType{F}}

    L::Int                  # NUMBER OF PARAMETERS

    f_counter::Ref{Int}     # NUMBER OF TIMES ANY COST FUNCTION IS CALLED
    g_counter::Ref{Int}     # NUMBER OF TIMES ANY GRAD FUNCTION IS CALLED
    values::Vector{F}       # STORES VALUES OF EACH COMPONENT FROM THE LAST CALL
    gnorms::Vector{F}       # STORES NORMS OF EACH GRADIENT FROM THE LAST CALL

    function CompositeCostFunction(
        components::AbstractVector{CostFunctionType{F}},
    ) where {F}
        if length(components) == 0
            return new{F}(CostFunctionType{F}[], 0, Ref(0), Ref(0), F[], F[])
        end
        
        L = length(first(components))
        for component in components; @assert length(component) == L; end

        return new{F}(
            convert(Vector{CostFunctionType{F}}, components),
            L, Ref(0), Ref(0),
            Vector{F}(undef, L),
            Vector{F}(undef, L),
        )
    end
end

"""
    CompositeCostFunction(f̄::CostFunctionType{F}...)

Alternate constructor, letting each function be passed as its own argument.

"""
function CompositeCostFunction(components::CostFunctionType{F}...) where {F}
    return CompositeCostFunction(CostFunctionType{F}[component for component in components])
end

Base.length(fn::CompositeCostFunction) = fn.L

function cost_function(fn::CompositeCostFunction)
    fs = [cost_function(component) for component in fn.components]
    return (x̄) -> (
        fn.f_counter[] += 1;
        total = 0;
        for (i, f) in enumerate(fs);
            fx = f(x̄);          # EVALUATE EACH COMPONENT
            fn.values[i] = fx;
            total += fx;        # TALLY RESULTS FOR THE COMPONENT
        end;
        total
    )
end

function grad_function(fn::CompositeCostFunction{F}) where {F}
    g!s = [grad_function(component) for component in fn.components]
    ∇f̄_ = zeros(F, fn.L)        # USE THIS AS THE GRADIENT VECTOR FOR EACH COMPONENT CALL
    return (∇f̄, x̄) -> (
        fn.g_counter[] += 1;
        ∇f̄ .= 0;
        for (i, g!) in enumerate(g!s);
            g!(∇f̄_, x̄);         # EVALUATE EACH COMPONENT. WRITES GRADIENT TO ∇f̄_
            fn.gnorms[i] = norm(∇f̄_);
            ∇f̄ .+= ∇f̄_;         # TALLY RESULTS FOR THE COMPONENT
        end;
        ∇f̄;
    )
end
