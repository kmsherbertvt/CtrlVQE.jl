module ModularDevices
    import ...Parameters, ...Devices, ...LocallyDrivenDevices
    import ..Algebras

    import ...TempArrays: array
    const LABEL = Symbol(@__MODULE__)

    import ...Integrations

    using LinearAlgebra: mul!

    import ..Algebras: AlgebraType
    import ..StaticHamiltonians: StaticHamiltonianType
    import ..Channels: LocalChannel
    import ..Mappers: Mapper, DisjointMapper, LinearMapper

    struct ModularDevice{
        F,
        A <: AlgebraType,
        H <: StaticHamiltonianType{A},
        C <: LocalChannel{A},
        M <: Mapper,
    } <: LocallyDrivenDevices.LocallyDrivenDevice{F}
        algebra::A
        static::H
        channels::Vector{C}
        mapper::M
        x::Vector{F}
    end

    function ModularDevice(
        static::StaticHamiltonianType{A},
        channels::Vector{<:LocalChannel{A}},
        mapper::DisjointMapper,
        F::Type{<:AbstractFloat},
    ) where {A}
        L = sum(Parameters.count, channels)
        return ModularDevice(A(), static, channels, mapper, zeros(F,L))
    end

    function ModularDevice(
        static::StaticHamiltonianType{A},
        channels::Vector{<:LocalChannel{A}},
        mapper::LinearMapper,
        F::Type{<:AbstractFloat},
    ) where {A}
        L = size(mapper.A, 2)
        return ModularDevice(A(), static, channels, mapper, zeros(F,L))
    end

    Algebras.algebratype(::ModularDevice{F,A,H,C,M}) where {F,A,H,C,M} = A

    """
        update_channels(device::ModularDevice)

    Update all parameters in each channel to match the current device parameters.

    """
    function sync_channels!(device::ModularDevice)
        L = maximum(Parameters.count, device.channels)
        y_ = array(eltype(device), (L,), LABEL)
        for (i, channel) in enumerate(device.channels)
            y = @view(y_[1:Parameters.count(channel)])
            map_values!(device, i, y)
            Parameters.bind!(channel, y)
        end
    end

    """
        sync_parameters!(device::ModularDevice)

    Update the parameter vector to match current state of all channels.

    Note that this may resize the parameter vector,
        which may warrant clearing cached arrays.

    """
    function sync_parameters!(device::CtrlVQE.Modulars.ModularDevice)
        L = sum(CtrlVQE.Parameters.count, device.channels)
        resize!(device.x, L)
        offset = 0
        for channel in device.channels
            Li = CtrlVQE.Parameters.count(channel)
            device.x[1+offset:Li+offset] .= CtrlVQE.Parameters.values(channel)
            offset += Li
        end
        return device
    end

    """
        map_values!(device::ModularDevice, i::Int, y::AbstractVector)

    Compute parameters in channel i as a function of all device parameters.

    Results are written to the vector `y` and returned.

    """
    function map_values! end

    function map_values!(
        device::ModularDevice{F,A,H,C,<:DisjointMapper},
        i::Int,
        y::AbstractVector{F},
    ) where {F,A,H,C}
        offset = sum(Parameters.count, @view(device.channels[1:i-1]), init=0)
        L = Parameters.count(device.channels[i])
        y .= @view(device.x[offset+1:offset+L])
        return y
    end

    function map_values!(
        device::ModularDevice{F,A,H,C,<:LinearMapper},
        i::Int,
        y::AbstractVector{F},
    ) where {F,A,H,C}
        mul!(y, device.mapper.A[:,:,i], device.x)
    end

    """
        map_gradients!(device::ModularDevice, i::Int, g::AbstractMatrix)

    Compute gradients for parameters in channel i with respect to each device parameter.

    Results are written to `g`, such that `g[k,j]` is ``∂_{x_k} y_j``.

    """
    function map_gradients! end

    function map_gradients!(
        device::ModularDevice{F,A,H,C,<:DisjointMapper},
        i::Int,
        g::AbstractMatrix{F},
    ) where {F,A,H,C}
        offset = sum(Parameters.count, @view(device.channels[1:i-1]), init=0)
        g .= 0
        for j in axes(g,2)
            g[offset+j, j] = 1
        end
        return g
    end

    function map_gradients!(
        device::ModularDevice{F,A,H,C,<:LinearMapper},
        i::Int,
        g::AbstractMatrix{F},
    ) where {F,A,H,C}
        g .= transpose(device.mapper.A[:,:,i])
        return g
    end


    ######################################################################################
    #= `Parameters` interface =#

    Parameters.count(device::ModularDevice) = length(device.x)
    Parameters.values(device::ModularDevice) = device.x

    function Parameters.names(
        device::ModularDevice{F,A,H,C,<:LinearMapper},
    ) where {F,A,H,C}
        return ["x$k" for k in eachindex(device.x)]
    end

    function Parameters.names(
        device::ModularDevice{F,A,H,C,<:DisjointMapper},
    ) where {F,A,H,C}
        return vcat(
            ["[#$i]$name" for name in Parameters.names(channel)]
                for (i, channel) in enumerate(device.channels)
        )
    end

    function Parameters.bind!(device::ModularDevice, x::AbstractVector)
        device.x .= x
        sync_channels!(device)
    end

    ######################################################################################
    # Counting methods

    Devices.nqubits(device::ModularDevice) = Devices.nqubits(device.algebra)
    Devices.nlevels(device::ModularDevice) = Devices.nlevels(device.algebra)
    Devices.ndrives(device::ModularDevice) = length(device.channels)
    Devices.ngrades(device::ModularDevice) = Devices.ndrives(device) * 2
    Devices.noperators(device::ModularDevice) = Devices.noperators(device.algebra)

    ######################################################################################
    # Algebra methods

    function Devices.localalgebra(device::ModularDevice; result=nothing)
        isnothing(result) && return Devices._localalgebra(device)
        return Devices.localalgebra(device.algebra; result=result)
    end

    ######################################################################################
    # Operator methods

    function Devices.qubithamiltonian(
        device::ModularDevice,
        ā,
        q::Int;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        return Devices.qubithamiltonian(device.static, ā, q; result=result)
    end

    function Devices.staticcoupling(
        device::ModularDevice,
        ā;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        return Devices.staticcoupling(device.static, ā; result=result)
    end

    function Devices.driveoperator(
        device::ModularDevice,
        ā,
        i::Int,
        t::Real;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        return Devices.driveoperator(device.channels[i], ā, t; result=result)
    end

    function Devices.gradeoperator(
        device::ModularDevice,
        ā,
        j::Int,
        t::Real;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        i = ((j-1) >> 1) + 1
        return Devices.gradeoperator(device.channels[i], ā, j, t; result=result)
    end

    ######################################################################################
    # Gradient methods

    function Devices.gradient(
        device::ModularDevice,
        grid::Integrations.IntegrationType,
        ϕ̄::AbstractMatrix;
        result=nothing,
    )
        isnothing(result) && (result = similar(device.x))

        # TEMP ARRAY TO HOLD GRADIENTS FOR EACH CHANNEL (one at a time)
        L = maximum(Parameters.count, device.channels)
        ∂y_ = array(eltype(device), (L,), LABEL)
        g_ = array(eltype(device), (length(device.x), L), LABEL)

        result .= 0
        for (i, channel) in enumerate(device.channels)
            ∂y = @view(∂y_[1:Parameters.count(channel)])
            Devices.gradient(channel, grid, ϕ̄[:,2i-1:2i]; result=∂y)

            g = @view(g_[:,1:Parameters.count(channel)])
            map_gradients!(device, i, g)

            # ADD MATRIX PRODUCT g * ∂y TO RESULT
            mul!(result, g, ∂y, 1, 1)
        end
        return result
    end

    ######################################################################################
    # `LocallyDrivenDevice` interface

    function LocallyDrivenDevices.drivequbit(device::ModularDevice, i::Int)
        return LocallyDrivenDevices.drivequbit(device.channels[i])
    end

    function LocallyDrivenDevices.gradequbit(device::ModularDevice, j::Int)
        i = ((j-1) >> 1) + 1
        return LocallyDrivenDevices.drivequbit(device.channels[i])
    end
end