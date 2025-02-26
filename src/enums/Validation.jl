"""
    Validation

Standardized interface for ensuring user-defined types are consistent with the interface.

"""
module Validation
    """
        validate(type)

    Check that `type` is consistent with the interface defined by its supertype.

    Each abstract type defined in the core interface implements its own `validate`,
        running a suite of interface-compliance and self-consistency checks.

    TODO: Add a note here about the optional `grid`, `device` kwargs,
        including the suggestion to use prototype

    """
    function validate end
end