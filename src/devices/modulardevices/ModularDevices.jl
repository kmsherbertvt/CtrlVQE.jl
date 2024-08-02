module ModularDevices
    import ..TempArrays: array
    const LABEL = Symbol(@__MODULE__)

    import ..Parameters, ..Devices, ..LocallyDrivenDevices

    using LinearAlgebra: mul!

    #=

    The methods we need to implement for a ModularDevice:

    Parameters
    - values, count, names, bind!
    Devices
    - ndrives, ngrades, nlevels, nqubits, noperators
    - localalgebra
    - qubithamiltonian, staticcoupling, driveoperator, gradeoperator
    - gradient
    LocallyDrivenDevices
    - drivequbit, gradequbit

    The conceit of ModularDevices is that each of these
        is delegated to more specialized abstraction:

    AlgebraType - handles localalgebra, nlevels, noperators
    StaticHamiltonianType{A} - handles qubithamiltonian, staticcoupling
    ChannelType{A} - handles driveoperator, gradeoperator;
        a LocalChannel type handles drivequbit

    =#
    ######################################################################################

    include("Algebras.jl");             import .Algebras: AlgebraType
    include("StaticHamiltonians.jl");   import .StaticHamiltonians: StaticHamiltonianType
    include("Channels.jl");             import .Channels: ChannelType, LocalChannel
    include("Mappers.jl");              import .Mappers: Mapper
    import .Mappers: DisjointMapper, LinearMapper

    ######################################################################################

    struct ModularDevice{
        F,
        A <: AlgebraType,
        H <: StaticHamiltonianType{A},
        C <: LocalChannel{A},
        M <: Mapper,
    } <: Devices.LocallyDrivenDevice{F}
        algebra::A
        static::H
        channels::Vector{C}
        mapper::M
        x::Vector{F}
    end

    """
        maxchannelcount(device::ModularDevice)

    Identify the largest number of parameters found in any channel.

    Useful for allocating temp arrays when mapping between device/channel parameters.

    """
    function maxchannelcount(device::ModularDevice)
        return maximum(Parameter.count, device.channels)
    end

    """
        update_channels(device::ModularDevice)

    Update all parameters in each channel to match the current device parameters.

    """
    function syncchannels!(device::ModularDevice)
        y_ = array(eltype(device), maxchannelcount(device), LABEL)
        for (i, channel) in enumerate(device.channels)
            y = @view(y_[1:Parameters.count(channel)])
            map_values!(device.map, i, y)
            Parameters.bind!(channel, y)
        end
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
        L = Parameters.count(device.channel[i])
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

    Parameters.count(::ModularDevice) = length(device.x)
    Parameters.values(device::ModularDevice) = deepcopy(device.x)

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

    function Parameters.bind!(device::ModularDevice{F}, x::AbstractVector)
        device.x .= x
        update_channels!(device)
    end

    ######################################################################################
    # Counting methods

    Devices.nqubits(device::ModularDevice) = Devices.nqubits(device.static)
    Devices.nlevels(device::ModularDevice) = Devices.nlevels(device.algebra)
    Devices.ndrives(device::ModularDevice) = length(device.channels)
    Devices.ngrades(device::ModularDevice) = Devices.ndrives(device) * 2
    Devices.noperators(device::ModularDevice) = Devices.noperators(device.algebra)

    ######################################################################################
    # Algebra methods

    function Devices.localalgebra(device::ModularDevice; result=nothing)
        isnothing(result) && return _localalgebra(device)
        āq = Devices.localalgebra(device.algebra)
        for q in 1:nqubits(device)
            result[:,:,:,q] .= āq
        end
        return result
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
        ā::MatrixList;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        return Devices.staticcoupling(device.static, ā; result=result)
    end

    function Devices.driveoperator(
        device::ModularDevice,
        ā::MatrixList,
        i::Int,
        t::Real;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        return Devices.driveoperator(device.channel[i], ā, t; result=result)
    end

    function Devices.gradeoperator(
        device::ModularDevice,
        ā::MatrixList,
        j::Int,
        t::Real;
        result=nothing,
    )
        C = Complex{eltype(device)}
        isnothing(result) && (result = Array{C}(undef, size(ā)[1:2]))
        i = ((j-1) >> 1) + 1
        return Devices.gradeoperator(device.channel[i], ā, j, t; result=result)
    end

    ######################################################################################
    # Gradient methods

    function Devices.gradient(
        device::ModularDevice,
        grid::Integrations.IntegrationType,
        ϕ̄::AbstractMatrix;
        result=nothing,
    )
        # TEMP ARRAY TO HOLD GRADIENTS FOR EACH CHANNEL (one at a time)
        ∂y_ = array(eltype(device), maxchannelcount(device), LABEL)
        g_ = array(eltype(device), (length(device.x), maxchannelcount(device)), LABEL)

        result .= 0
        for (i, channel) in enumerate(device.channels)
            ∂y = @view(∂y_[1:Parameters.count(channel)])
            Devices.gradient(channel, grid, ϕ̄[2i-1:2i]; result=∂y)

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
        return LocallyDrivenDevices.drivequbit(device.channel[i])
    end

    function LocallyDrivenDevices.gradequbit(device::ModularDevice, j::Int)
        i = ((j-1) >> 1) + 1
        return LocallyDrivenDevices.drivequbit(device.channel[i])
    end

end