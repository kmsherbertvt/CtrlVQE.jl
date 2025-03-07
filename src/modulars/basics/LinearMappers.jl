module LinearMappers
    import CtrlVQE.ModularFramework as Modular
    import CtrlVQE.ModularFramework: ParameterMap

    import CtrlVQE: Parameters

    import LinearAlgebra: mul!

    """
        LinearMapper(encoding::Vector{F}, size::Vector{Int})
        LinearMapper(A::Array{F,3})

    Associate each device parameter with the weight of a basis vector for each drive.

    The implemetation of this type is designed
        around the assumption that each drive has the same number of parameters.

    The linear map `A` is a 3d array such that `y[i] = A[:,:,i] * x`,
        where `y[i]` is the parameter vector for drive `i`
        and `x` is the parameter vector for the whole device.

    `A` is represented internally as a data vector `encoding` and shape vector `size`,
        so that it can be resized in-place.

    ```jldoctests
    julia> using CtrlVQE.ModularFramework;

    julia> pmap = LinearMapper(ones(2,4,3));

    julia> device = Devices.Prototype(LocalDevice{Float64}, 3; pmap=pmap);

    julia> validate(pmap; device=device);

    julia> map_values(pmap, device, 1)
    2-element Vector{Float64}:
     0.0
     0.0

    julia> Parameters.count(device)
    4

    ```

    """
    struct LinearMapper{F} <: ParameterMap
        encoding::Vector{F}
        size::Vector{Int}
    end

    function LinearMapper(A::AbstractArray{F,3}) where {F}
        return LinearMapper(collect(vec(A)), collect(size(A)))
    end

    function Parameters.names(pmap::LinearMapper, device)
        return ["x$k" for k in eachindex(device.x)]
    end

    function Modular.sync!(pmap::LinearMapper, device)
        # TEMP: REMEMBER ORIGINAL SIZE
        L = length(device.x)

        # ENSURE x IS THE RIGHT SIZE
        resize!(device.x, pmap.size[2])

        # TEMP: PUT REASONABLE VALUES IN FOR THE NEW ELEMENTS
        device.x[1+L:end] .= 0

        # ENSURE x HAS THE RIGHT VALUES
        #= TODO: Tacitly assume encoding is orthogonal,
            and compute projection of drives onto the space spanned by the encoding
            (implicitly assuming each basis vector is orthogonal). =#

        return device
    end

    function Modular.map_values(pmap::LinearMapper, device, i::Int; result=nothing)
        L = Parameters.count(device.drives[i])
        isnothing(result) && (result = Array{eltype(device)}(undef, L))
        A = reshape(pmap.encoding, Tuple(pmap.size))
        return mul!(result, @view(A[:,:,i]), device.x)
    end

    function Modular.map_gradients(pmap::LinearMapper, device, i::Int; result=nothing)
        LT = Parameters.count(device)
        Li = Parameters.count(device.drives[i])
        isnothing(result) && (result = Array{eltype(device)}(undef, LT, Li))
        A = reshape(pmap.encoding, Tuple(pmap.size))
        result .= transpose(@view(A[:,:,i]))
        return result
    end

    """
        addbasisvector!(pmap, a)

    Update a `LinearMapper` to add in a new device parameter.

    You'll pretty much always need to `sync!` a device after calling this.

    # Parameters
    - `pmap`: the `LinearMapper` object to mutate.
    - `a::AbstractMatrix`: a[:,i] contains the mapping
        from the new device parameter to the parameters of drive `i`.

    """
    function addbasisvector!(pmap::LinearMapper{F}, a::AbstractMatrix{F}) where {F}
        A = reshape(pmap.encoding, Tuple(pmap.size))

        @assert size(a,1) == pmap.size[1]   # EACH COLUMN MAPS PARAMETERS OF A DRIVE
        @assert size(a,2) == pmap.size[3]   # THERE'S A COLUMN FOR EACH DRIVE

        # ACCOUNT FOR ONE NEW DEVICE PARAMETER
        pmap.size[2] += 1

        # COPY THE EXISTING ENCODING TO A NEW, LARGER ARRAY
        A_ = Array{F}(undef, Tuple(pmap.size))
        for k in axes(A,2)
            A_[:,k,:] .= @view(A[:,k,:])
        end

        # COPY THE NEW COLUMNS IN
        A_[:,end,:] .= a

        # UPDATE THE ENCODING IN-PLACE
        resize!(pmap.encoding, length(A_))
        pmap.encoding .= vec(A_)

        return pmap
    end
end