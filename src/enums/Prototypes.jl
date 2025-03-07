"""
    Prototypes

Standardized interface for quickly constructing certain objects.

"""
module Prototypes
    """
        Prototype(::Type{T}; kwargs...)

    Construct a prototypical object of type `T`.

    Each type accepts a different collection of keyword arguments,
        which is largely up to the user defining the type,
        depending on how flexible they'd like their prototype function to be.
    However, certain keyword arguments must be accepted for certain abstract types,
        in order for standard tests and benchmarking to work.
    But even these should be provided sensible default values.

        Prototype(::Type{<:DeviceType}; n::Int, T::Real, kwargs...)
        Prototype(::Type{<:IntegrationType}; T::Real, r::Int, kwargs...)

    - `n` is the number of qubits.
    - `r` is the total number of time steps.
    - `T` is a default pulse duration (often but not always irrelevant in `DeviceTypes`).

    """
    function Prototype end

end