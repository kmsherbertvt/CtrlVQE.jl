using Memoization: @memoize

"""
    array(::F, shape::Tuple, index=nothing)

Fetch a temporary array with type `F` and shape `shape`.

The `index` parameter is an additional unique key,
    allowing the module to cache mulitple arrays of the same type and shape.
You should pass `index=Symbol(@__MODULE__)` to prevent collisions across modules.
You may pass a tuple, eg. `index=(Symbol(@__MODULE__), :otherkey)`
    to prevent collisions within a module.

"""
@memoize function array(::F, shape::Tuple, index=nothing) where {F<:Number}
    # TODO (lo): Thread-safe and hands-off approach to this.
    return Array{F}(undef, shape)
end

"""
    array(F::Type{<:Number}, shape::Tuple, index=nothing)

Same as above but passing the type directly, rather than an instance of the type.

"""
function array(F::Type{<:Number}, shape::Tuple, index=nothing)
    return array(zero(F), shape, index)
end