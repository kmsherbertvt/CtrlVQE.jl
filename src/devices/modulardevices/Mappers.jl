module Mappers
    """
        Mapper

    Enumeration type used to determine
        how each device parameter relates to the parameters for each channel.

    """
    abstract type Mapper end

    """
        DISJOINT

    Useful for when parameters for each channel are completely independent of one another.

    In this case, all the parameters for channel 1
        are followed by all the parameters for channel 2, and so on.

    """
    struct DisjointMapper <: Mapper end
    DISJOINT = DisjointMapper()

    """
        LinearMapper(A::Array{F,3})

    Useful for when each channel is decomposed into the same basis,
        and each device parameter scales a characteristic pulse pattern over all channels.

    This is the mapper of choice for "Modal Adaptive", "Natural", and "Normal" pulses.

    The parameter tensor `A` is indexed `A[j,k,i]`, with the following indices:
    - j indexes y (channel parameters)
    - k indexes x (device parameters)
    - i indexes the channel

    This choice of indexing maximizes efficiency of the operation ``y_{ij} = A_{ijk} x_k``,
        where ``y_i`` is the parameter vector for channel i,
        and ``x`` is the device parameter vector.

    Another operation of interest is extending `A` in-place
        to accommodate an additional parameter, perhaps selected adaptively.
    This would involve permuting the matrix so that `k` is indexed last,
        pushing a vectorized form of the new A[:,k,:], and then unpermuting.

    TODO: When you do this and it works, include an example here.

    """
    struct LinearMapper{F} <: Mapper A::Array{F,3} end
end