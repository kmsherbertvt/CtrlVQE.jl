module Bounds
    import ...CostFunctions

    import ...TempArrays: array
    const LABEL = Symbol(@__MODULE__)

    import ...Integrations, ...Signals

    import ..ModularDevices: ModularDevice, map_gradients!
    import ..Channels: QubitChannel

    # NOTE: Implicitly use smooth bounding function.
    smoothwall(u) = exp(u - 1/u)
    smoothgrad(u) = exp(u - 1/u) * (1 + 1/u^2)

    function overflow_cost(
        signal::Signals.SignalType{F,FΩ},
        grid::Integrations.IntegrationType{F},
        A::F,
        σ::F,
    ) where {F,FΩ}
        Φ(t, Ω) = (
            u = (abs(Ω) - A) / σ;
            u ≤ 0 ? zero(u) : u
        )

        t̄ = Integrations.lattice(fn.grid)                       # CACHED, THEREFORE FREE
        Ω̄ = array(FΩ, (length(t̄),), (LABEL, :overflow))         # TO FILL, FOR EACH DRIVE

        Ω̄ = Signals.valueat(signal, t̄; result=Ω̄)
        return Integrations.integrate(grid, Φ, Ω̄)
    end

    function overflow_grad!(
        result,         # A vector with an element for each parameter in signal.
        signal::Signals.SignalType{F,FΩ},
        grid::Integrations.IntegrationType{F},
        A::F,
        σ::F,
    ) where {F,FΩ}
        Φ(t, Ω, ∂) = (
            u = (abs(Ω) - A) / σ;
            u ≤ 0 ? zero(u) : real(conj(Ω)*∂) / (abs(Ω)*fn.σ)
        )

        t̄ = Integrations.lattice(fn.grid)                       # CACHED, THEREFORE FREE
        Ω̄ = array(FΩ, (length(t̄),), (LABEL, :overflow))         # TO FILL, FOR EACH DRIVE
        ∂̄ = array(FΩ, (length(t̄),), (LABEL, :gradient))         # TO FILL, FOR EACH DRIVE

        Ω̄ = Signals.valueat(signal, t̄; result=Ω̄)
        for k in 1:Parameters.count(signal)
            ∂̄ = Signals.partial(k, signal, t̄; result=∂̄);
            result[k] = Integrations.integrate(grid, Φ, Ω̄, ∂̄)
        end
        return result
    end


    abstract type BoundaryType end

    function get_signal end
    function get_slice end

    struct AmplitudeBound <: BoundaryType end
    AMPLITUDE = AmplitudeBound()

    struct FrequencyBound <: BoundaryType end
    FREQUENCY = FrequencyBound()





    get_signal(::AmplitudeBound, channel::QubitChannel) = channel.Ω
    function get_slice(::AmplitudeBound, channel::QubitChannel)
        return 1:Parameters.count(channel.Ω)
    end

    get_signal(::FrequencyBound, channel) = channel.ν
    function get_slice(::FrequencyBound, channel::QubitChannel)
        return (1+Parameters.count(channel.Ω)):Parameters.count(channel)
    end




    """
        Bound(boundarytype, device, grid, A, σ, λ)

    Smooth bounds on an integral over each drive signal in a device.

    Each pulse is integrated separately; any "area" beyond ΩMAX is penalized.

    NOTE: When cost and gradient functions are constructed
        by `CostFunctions.cost_function` and `CostFunctions.grad_function_inplace`,
        work arrays are allocated based on the number of parameters
        currently in each channel.
    Therefore, if a device is "adapted" to include more parameters,
        these functions must be regenerated by calling the factory function again.
    But you should not have to reconstruct the `Bound` object itself.

    # Parameters
    - `boundarytype`: the boundary type, determining which signal is penalized
    - `device`: the device
    - `grid`: how to integrate over time
    - `A`: Maximum allowable absolute signal on a device.
    - `λ`: Penalty strength.
    - `σ`: Penalty effective width: smaller means steeper.

        Bound(boundarytype, device, grid; A=1, σ=A, λ=1)

    Convenience constructor when each channel has the same bounds,
        where A, σ, and λ are passed as kwarg scalars.

    A typical construction, using the standard choice σ=ΩMAX,
        may look like this:

    ```
    ΩMAX = 2π * 0.02 # GHz
    bound = Bound(AMPLITUDE, device, grid; A=ΩMAX)
    ```


    """
    struct Bound{F,T<:BoundaryType} <: CostFunctions.CostFunctionType{F}
        boundarytype::T
        device::ModularDevice      # Must be eltype F
        grid::Integrations.IntegrationType{F}
        A::Vector{F}            # MAXIMUM PERMISSIBLE AMPLITUDE (for each channel)
        σ::Vector{F}            # STEEPNESS OF BOUND (for each channel)
        λ::Vector{F}            # STRENGTH OF BOUND (for each channel)

        function Bound(
            boundarytype::BoundaryType,
            device::ModularDevice,
            grid::Integrations.IntegrationType{F},
            A::AbstractVector{<:Real},
            σ::AbstractVector{<:Real},
            λ::AbstractVector{<:Real},
        ) where {F}
            @assert eltype(device) == F
            @assert length(A) == length(σ) == length(λ) == length(device.channels)

            return new{F,T}(
                boundarytype,
                device,
                grid,
                convert(Vector{F}, A),
                convert(Vector{F}, σ),
                convert(Vector{F}, λ),
            )
        end

    end

    function Bound(
        boundarytype::BoundaryType,
        device::ModularDevice,
        grid::Integrations.IntegrationType{F};
        A=1,
        σ=A,
        λ=1,
    ) where {F}
        nD = length(device.channels)
        return Bound(boundarytype, device, grid, fill(A, nD), fill(σ, nD), fill(λ, nD))
    end




    Base.length(fn::Bound) = Parameters.count(fn.device)

    function CostFunctions.cost_function(fn::Bound{F}) where {F}
        x̄0 = Vector{eltype(device)}(undef, Parameter.count(device))
        return (x̄) -> (
            x̄0 .= Parameters.values(fn.device);
            Parameters.bind!(fn.device, x̄);
            total = zero(F);
            for channel in fn.device.channels;
                signal = get_signal(fn.boundarytype, channel);
                J = overflow_cost(signal, fn.grid, fn.A, fn.σ);
                total += fn.λ * smoothwall(J);
            end;
            Parameters.bind!(fn.device, x̄0);
            total
        )
    end

    function CostFunctions.grad_function_inplace(fn::Bound{F}) where {F}
        x̄0 = Vector{eltype(device)}(undef, Parameter.count(device))

        # TEMP ARRAY TO HOLD GRADIENTS FOR EACH CHANNEL (one at a time)
        L = maximum(Parameters.count, fn.device.channels)
        ∂y_ = Array{eltype(device)}(undef, L)
        g_ = Array{eltype(device)}(undef, length(device.x), L)

        return (∇f̄, x̄) -> (
            x̄0 .= Parameters.values(fn.device);
            Parameters.bind!(fn.device, x̄);
            ∇f̄ .= 0;
            for (i, channel) in enumerate(fn.device.channels);
                signal = get_signal(fn.boundarytype, channel);
                slice = get_slice(fn.boundarytype, channel);

                # Effect of signal parameters on overflow.
                ∂y = @view(@view(∂y_[1:Parameters.count(signal)])[slice]);
                overflow_grad!(∂y, signal, fn.grid, fn.A, fn.σ);

                # Smooth out the cusp between some overflow and none.
                J = overflow_cost(signal, fn.grid, fn.A, fn.σ);
                ∂y .*= fn.λ * smoothgrad(J);

                # Connect signal parameters to the device.
                g = @view(g_[:,1:Parameters.count(channel)]);
                map_gradients!(device, i, g);

                # ADD MATRIX PRODUCT g * ∂y TO RESULT
                mul!(∇f̄, @view(g[:,slice]), ∂y, 1, 1);
            end;
            Parameters.bind!(fn.device, x̄0);
            ∇f̄
        )
    end
end

