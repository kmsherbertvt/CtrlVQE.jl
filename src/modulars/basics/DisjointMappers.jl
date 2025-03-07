module DisjointMappers
    import ..ModularFramework as Modular
    import ..ModularFramework: ParameterMap

    import CtrlVQE: Parameters

    """
        DISJOINT

    Useful for when parameters for each drive are completely independent of one another.

    In this case, all the parameters for drive 1
        are followed by all the parameters for drive 2, and so on.

    ```jldoctests
    julia> using CtrlVQE.ModularFramework;

    julia> pmap = DISJOINT;

    julia> device = Devices.Prototype(LocalDevice{Float64}, 3; pmap=pmap);

    julia> validate(pmap; device=device);

    julia> map_values(pmap, device, 1)
    2-element Vector{Float64}:
     0.0
     0.0

    julia> Parameters.count(device)
    6

    ```

    """
    struct DisjointMapper <: ParameterMap end
    DISJOINT = DisjointMapper()

    function Parameters.names(::DisjointMapper, device)
        return vcat((
            ["[#$i]$name" for name in Parameters.names(drive)]
                for (i, drive) in enumerate(device.drives)
        )...)
    end

    function Modular.sync!(::DisjointMapper, device)
        # ENSURE x IS THE RIGHT SIZE
        resize!(device.x, sum(Parameters.count, device.drives, init=0))

        # ENSURE x HAS THE RIGHT VALUES
        Δ = 0
        for (i, drive) in enumerate(device.drives)
            L = Parameters.count(drive)
            device.x[Δ+1:Δ+L] .= Parameters.values(drive)
            Δ += L
        end

        return device
    end

    function Modular.map_values(::DisjointMapper, device, i::Int; result=nothing)
        L = Parameters.count(device.drives[i])
        isnothing(result) && (result = Array{eltype(device)}(undef, L))
        offset = sum(Parameters.count, @view(device.drives[1:i-1]), init=0)
        result .= @view(device.x[offset+1:offset+L])
        return result
    end

    function Modular.map_gradients(::DisjointMapper, device, i::Int; result=nothing)
        LT = Parameters.count(device)
        Li = Parameters.count(device.drives[i])
        isnothing(result) && (result = Array{eltype(device)}(undef, LT, Li))
        offset = sum(Parameters.count, @view(device.drives[1:i-1]), init=0)
        result .= 0
        for j in axes(result,2)
            result[offset+j, j] = 1
        end
        return result
    end
end