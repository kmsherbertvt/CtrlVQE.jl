import ..CostFunctions
export GlobalAmplitudeBound

import ..Parameters, ..Integrations, ..Devices, ..Signals

# NOTE: Implicitly use smooth bounding function.
wall(u) = exp(u - 1/u)
grad(u) = exp(u - 1/u) * (1 + 1/u^2)

"""
    GlobalAmplitudeBound(device, grid, ΩMAX, λ, σ)

Smooth bounds on an integral over each drive signal in a device.

Each pulse is integrated separately; any "area" beyond ΩMAX is penalized.

# Parameters
- `device`: the device
- `grid`: how to integrate over time
- `ΩMAX`: Maximum allowable amplitude on a device.
- `λ`: Penalty strength.
- `σ`: Penalty effective width: smaller means steeper.

"""
struct GlobalAmplitudeBound{F,FΩ} <: CostFunctions.CostFunctionType{F}
    device::Devices.DeviceType{F,FΩ}
    grid::Integrations.IntegrationType{F}
    ΩMAX::F             # MAXIMUM PERMISSIBLE AMPLITUDE
    λ::F                # STRENGTH OF BOUND
    σ::F                # STEEPNESS OF BOUND

    function GlobalAmplitudeBound(
        device::Devices.DeviceType{DF,FΩ},
        grid::Integrations.IntegrationType{IF},
        ΩMAX::Real,
        λ::Real,
        σ::Real,
    ) where {DF,FΩ,IF}
        F = promote_type(Float16, DF, real(FΩ), IF, eltype(ΩMAX), eltype(λ), eltype(σ))
        return new{F,FΩ}(device, grid, ΩMAX, λ, σ)
    end
end

Base.length(fn::GlobalAmplitudeBound) = Parameters.count(fn.device)

function CostFunctions.cost_function(fn::GlobalAmplitudeBound{F,FΩ}) where {F,FΩ}
    t̄ = Integrations.lattice(fn.grid)                       # CACHED, THEREFORE FREE
    Ω̄ = Vector{FΩ}(undef, length(t̄))                        # TO FILL, FOR EACH DRIVE

    Φ(t, Ω) = (
        u = (abs(Ω) - fn.ΩMAX) / fn.σ;
        u ≤ 0 ? zero(u) : fn.λ * wall(u)
    )

    return (x̄) -> (
        Parameters.bind(fn.device, x̄);
        total = zero(F);
        for i in 1:Devices.ndrives(fn.device);
            signal = Devices.drivesignal(fn.device, i);
            Ω̄ = Signals.valueat(signal, t̄; result=Ω̄);
            total += Integrations.integrate(fn.grid, Φ, Ω̄)
        end;
        total
    )
end

function CostFunctions.grad_function_inplace(fn::GlobalAmplitudeBound{F,FΩ}) where {F,FΩ}
    t̄ = Integrations.lattice(fn.grid)                       # CACHED, THEREFORE FREE
    Ω̄ = Vector{FΩ}(undef, length(t̄))                        # TO FILL, FOR EACH DRIVE
    ∂̄ = Vector{FΩ}(undef, length(t̄))                        # TO FILL, FOR EACH PARAMETER

    Φ(t, Ω, ∂) = (
        u = (abs(Ω) - fn.ΩMAX) / fn.σ;
        u ≤ 0 ? zero(u) : fn.λ * grad(u) * real(conj(Ω)*∂) / (abs(Ω)*fn.σ)
    )

    return (∇f̄, x̄) -> (
        Parameters.bind(fn.device, x̄);
        ∇f̄ .= 0;
        offset = 0;
        for i in 1:Devices.ndrives(fn.device);
            signal = Devices.drivesignal(fn.device, i);
            Ω̄ = Signals.valueat(signal, t̄; result=Ω̄);
            L = Parameters.count(signal);
            for k in 1:L
                ∂̄ = Signals.partial(k, signal, t̄; result=∂̄);
                ∇f̄[offset+k] = Integrations.integrate(fn.grid, Φ, Ω̄, ∂̄);
            end;
            offset += L;
        end;
        ∇f̄
    )
end