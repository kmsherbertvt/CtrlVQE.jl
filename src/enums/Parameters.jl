"""
    Parameters

Standardized interface for interacting with variational parameters.

Most commonly encountered when implementing `Signals` and `Devices`.

"""
module Parameters
    """
        count(entity)

    The number of variational parameters in `entity`.

    # Implementation

    Must return a non-negative integer.

    """
    function count end

    """
        names(entity)

    An ordered list of human-readable names for each variational parameter in `entity`.

    # Implementation

    Must return a vector of strings.

    """
    function names end

    """
        values(entity)

    An ordered list of the numerical values for each variational parameter in `entity`.

    The resulting array should be treated as read-only,
        since whether it is or isn't "backed" by `entity` is not enforced.
    That means changes to the resulting array MAY or MAY NOT modfiy `entity`.
    So don't do it!

    # Implementation

    Must return a vector of some float type.

    If at all possible, avoid allocating a new array when calling this function.

    """
    function values end

    """
        bind!(entity, x̄::AbstractVector)

    Assigns new values for each variational parameter in `entity`.

    # Implementation

    This method should mutate `entity` such that, for example,
        the expression `bind!(entity, x̄); ȳ = values(entity); x̄ == ȳ` evaluates true.
    The method should return `entity`,
        in accordance with Julia's style guidelines for mutating functions.

    """
    function bind! end
end