module LinearMappers
    import ..ModularFramework as Modular
    import ..ModularFramework: ParameterMap

    import CtrlVQE: Parameters

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
    struct LinearMapper{F} <: ParameterMap
        A::Array{F,3}
    end

    function Parameters.names(pmap::LinearMapper, device)
        return ["x$k" for k in eachindex(device.x)]
    end

    function Modular.sync!(pmap::LinearMapper, device)
        # ENSURE x IS THE RIGHT SIZEs
        resize!(device.x, size(pmap, 1))

        # ENSURE x HAS THE RIGHT VALUES
        #= TODO: This operation isn't super well-defined.
        I suppose we could do some kind of least-squares fit?
        Presumably better to just...not.

        Um well shouldn't we be able to at least PROJECT drive parameters onto the basis vectors defined by A?
        We have to assume A is orthogonal but that's okay.
        It clearly isn't guaranteed to be entirely correct,
            but it is easy to do, well-defined, and a useful operation.
        =#
    end

    function Modular.map_values(pmap::LinearMapper, device, i::Int, result)
        return mul!(result, pmap.A[:,:,i], device.x)
    end

    function Modular.map_gradients(pmap::LinearMapper, device, i::Int, result)
        result .= transpose(pmap.A[:,:,i])
        return result
    end

    #= TODO: Go ahead and add a special function here to add on a column to A.
    And export it in `ModularFramework.jl` =#
end