import .CostFunctions: CostFunctionType

import ..CtrlVQE: CostFunctions

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
    return CostFunctions.cost_function(fn)(x̄)
end

"""
    grad_function(fn::CostFunctionType)

Constructs a `Function` to calculate the gradient of `fn` at a particular point.

The function accepts a parameter vector `x̄`
    which should have the type and length given by `eltype(fn)` and `length(fn)`.
The function returns the gradient (a vector) of `fn` at the point `x̄`.

"""
function grad_function(fn::CostFunctionType{F}; kwargs...) where {F}
    g! = CostFunctions.grad!function(fn; kwargs...)
    ∇f̄ = Vector{F}(undef, length(fn))
    return (x̄) -> (g!(∇f̄, x̄); ∇f̄)
end