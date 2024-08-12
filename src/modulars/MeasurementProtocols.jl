module MeasurementProtocols
    import ..Algebras

    import ...TempArrays: array
    const LABEL = Symbol(@__MODULE__)

    import ...LinearAlgebraTools
    import ...Integrations, ...Devices
    import ...Bases, ...Operators

    abstract type MeasurementProtocolType{A<:Algebras.AlgebraType} end

    Algebras.algebratype(::MeasurementProtocolType{A}) where {A} = A

    """
        measure(measurer, device, basis, ψ, t)

    Perform the measurement protocol on the state ψ provided in the given basis.

    In case the measurement protocol is time-dependent
        (as in, for example, measurement occuring in a device frame),
        the time is also provided.

    """
    function measure end


    """
        observables(measurer)

    Identify the number of Hermitian observables needed for this measurement.

    For example, to measure the normalized energy,
        separate observables are needed for both energy and normalization,
        and the results are combined in a non-linear way to produce the final outcome.

    """
    function nobservables end

    """
        observables(measurer, device, basis, t; result=nothing)

    Prepare dense matrices for each Hermitian observable needed for this measurement,
        as represented in the requested basis.

    In case the measurement protocol is time-dependent
        (as in, for example, measurement occuring in a device frame),
        the time is also provided.

    The result is indexed such that `result[:,:,k]` is the kth matrix,
        the same format required by the parameter Ō in `Evolutions.gradientsignal`.
    The size of the third dimension must be equal to `nobservables(measurer)`.

    The array is stored in `result` if provided.
    If `result` is not provided, the array is of type `Complex{eltype(device)}`.

    """
    function observables end

    """
        gradient(measurer, device, ϕ, grid; result=nothing)

    Compute the gradient vector of parameters in `device`,
        given the gradient signals `ϕ` evaluated on a time `grid`.

    The argument `ϕ` is a 3d array; `ϕ[:,j,k]` contains the jth gradient signal
        ``ϕ_j(t)`` evaluated at each point in `grid` for observable `k`.
    This is the same format returned by `Evolutions.gradientsignal`.

    This function is mainly expected to delegate to `Devices.gradient`,
        which expects the 2d array `ϕ[:,:,k]`.

    The array is stored in `result` if provided.
    If `result` is not provided, the array is of type `Complex{eltype(device)}`.

    """
    function gradient end





    """
        BareMeasurement(algebra, observable, basis, frame)

    Reperesents a bare measurement of an observable in a given basis and frame,
        without any logical projection prior to frame rotation.

    """
    struct BareMeasurement{A} <: MeasurementProtocolType{A}
        observable::Matrix
        basis::Bases.BasisType
        frame::Operators.StaticOperator
    end

    function measure(
        measurer::BareMeasurement,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        ψ::AbstractVector,
        t::Real,
    )
        # COPY THE STATE SO WE CAN ROTATE IT
        ψ_ = array(eltype(ψ), size(ψ), LABEL)
        ψ_ .= ψ

        # ROTATE THE STATE INTO THE MEASUREMENT BASIS
        U = Devices.basisrotation(measurer.basis, basis, device)
        LinearAlgebraTools.rotate!(U, ψ_)

        # APPLY THE FRAME ROTATION
        Devices.evolve!(measurer.frame, device, measurer.basis, -t, ψ_)

        # TAKE THE EXPECTATION VALUE
        return real(LinearAlgebraTools.expectation(measurer.observable, ψ_))
    end

    nobservables(measurement::BareMeasurement) = 1

    function observables(
        measurer::BareMeasurement,
        device::Devices.DeviceType,
        basis::Bases.BasisType,
        t::Real;
        result=nothing
    )
        N = Devices.nstates(device)
        isnothing(result) && (result = Array{Complex{eltype(device)}}(undef, N, N, 1))

        # REPRESENT O IN THE MEASUREMENT BASIS
        result[:,:,1] .= measurer.observable

        # ROTATE INTO THE REQUESTED BASIS
        U = Devices.basisrotation(basis, measurer.basis, device)
        LinearAlgebraTools.rotate!(U, @view(result[:,:,1]))

        # APPLY THE FRAME ROTATION
        Devices.evolve!(measurer.frame, device, basis, t, @view(result[:,:,1]))

        return result
    end

    function gradient(
        measurer::BareMeasurement,
        device::Devices.DeviceType,
        grid::Integrations.IntegrationType,
        ϕ::AbstractArray;
        result=nothing
    )
        return Devices.gradient(device, grid, @view(ϕ[:,:,1]); result=result)
    end





    #= TODO:
        - ProjectedMeasurement (projection prior to frame rotation,
            appropriate if frame rotation is done classically).
        - Normalization
        - NormalizedMeasurement (solves E and N both, and puts them together)
        - NormalizedProjectedMeasurement

    =#

end