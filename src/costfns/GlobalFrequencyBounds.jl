import ..CostFunctions
export GlobalFrequencyBound

import ..Parameters, ..Integrations, ..Devices, ..Signals

# NOTE: Implicitly use smooth bounding function.
wall(u) = exp(u - 1/u)
grad(u) = exp(u - 1/u) * (1 + 1/u^2)

"""
    GlobalAmplitudeBound(device, grid, ΩMAX, λ, σ)

Smooth bounds on an integral over each drive frequency in a device.

TODO: Well okay right now the integration is redundant.
    This function just discards parameters for each drive signal,
        and if there are any leftover, they are assumed to be the drive frequencies,
        and they are penalized if they are too far away from the resonance frequencies.
    Some day we will not necessarily use constant drive frequencies,
        and this type will look a lot more like `GlobalFrequencyBound`.


# Parameters
- `device`: the device
- `grid`: how to integrate over time
- `ΩMAX`: Maximum allowable amplitude on a device.
- `λ`: Penalty strength.
- `σ`: Penalty effective width: smaller means steeper.

"""
struct GlobalFrequencyBound{F,FΩ} <: CostFunctions.CostFunctionType{F}
    device::Devices.DeviceType{F,FΩ}
    grid::Integrations.IntegrationType{F}
    ΔMAX::F             # MAXIMUM PERMISSIBLE DETUNING
    λ::F                # STRENGTH OF BOUND
    σ::F                # STEEPNESS OF BOUND

    function GlobalFrequencyBound(
        device::Devices.DeviceType{DF,FΩ},
        grid::Integrations.IntegrationType{IF},
        ΔMAX::Real,
        λ::Real,
        σ::Real,
    ) where {DF,FΩ,IF}
        F = promote_type(Float16, DF, real(FΩ), IF, eltype(ΔMAX), eltype(λ), eltype(σ))
        return new{F,FΩ}(device, grid, ΔMAX, λ, σ)
    end
end

Base.length(fn::GlobalFrequencyBound) = Parameters.count(fn.device)

function CostFunctions.cost_function(fn::GlobalFrequencyBound{F,FΩ}) where {F,FΩ}
    return (x̄) -> (
        Parameters.bind(fn.device, x̄);
        total = zero(F);
        for i in 1:Devices.ndrives(fn.device);
            q = Devices.drivequbit(fn.device, i);
            Δ = Devices.detuningfrequency(fn.device, i, q);
            u = (abs(Δ) - fn.ΔMAX) / fn.σ;
            total += u ≤ 0 ? zero(u) : fn.λ * wall(u);
        end;
        total
    )
end

function CostFunctions.grad_function_inplace(fn::GlobalFrequencyBound{F,FΩ}) where {F,FΩ}
    # TRY TO INFER WHICH PARAMETERS REFER TO DRIVE FREQUENCIES
    #= TODO (mid): This is way too restrictive.
        We presently assume every drive frequency has exactly 0 or 1 parameters,
            and that they follow after all signal parameters.
        The correct solution will, I think, look almost identical to GlobalAmplitudeBounds
            after frequencies become signals themselves.
    =#
    nD = Devices.ndrives(fn.device)
    offset = 0
    for i in 1:nD
        signal = Devices.drivesignal(fn.device, i)
        offset += Parameters.count(signal)
    end

    offset == length(fn) && return (∇f̄, x̄) -> (∇f̄ .= 0; ∇f̄) # NO FREQUENCY PARAMETERS
    offset + nD == length(fn) || error("Ill-defined number of frequency parameters.")

    # AT THIS POINT (for now) ASSUME x[offset+i] == ith frequency
    return (∇f̄, x̄) -> (
        Parameters.bind(fn.device, x̄);
        ∇f̄ .= 0;
        for i in 1:nD;
            q = Devices.drivequbit(fn.device, i);
            Δ = Devices.detuningfrequency(fn.device, i, q);
            u = (abs(Δ) - fn.ΔMAX) / fn.σ;
            ∇f̄[offset+i] += u ≤ 0 ? zero(u) : fn.λ * grad(u) * sign(Δ) / fn.σ;
        end;
        ∇f̄
    )
end