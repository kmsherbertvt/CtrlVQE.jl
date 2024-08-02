module Channels
    import ..Parameters, ..Devices, ..LocallyDrivenDevices
    import ..Algebras: AlgebraType
    import ..Algebras: TruncatedBosonicAlgebra

    import ...TempArrays: array
    const LABEL = Symbol(@__MODULE__)

    import ...LinearAlgebraTools
    import ...Signals
    import ...Signals: SignalType

    """
        ChannelType{F,A}

    Component of `ModularDevices` delegated the following methods:
    - `Devices.driveoperator`
    - `Devices.gradeoperator`
    - `Devices.gradient`
    They also need to implement the whole `Parameters` interface.
    - `Devices.staticcoupling`

    While `driveoperator`, `gradeoperator`, and `gradient` have optional `result` kwargs,
        subtypes of `ChannelType` should consider them mandatory.
    Further, `driveoperator` should omit the drive index `i` as a parameter,
        and only the parity of `j` should be used in `gradeoperator`.
    The `ModularDevice` interface handles the rest.

    """
    abstract type ChannelType{A<:AlgebraType} end

    """
        LocalChannel{F,A}

    Extension of `ChannelType` which is additionally delegated
        `Devices.drivequbit` from `LocallyDrivenDevices`.

    While `Devices.drivequbit` normally accepts a drive index `i`,
        subtypes of `ChannelType` should omit this.
    The `ModularDevice` interface handles the rest.


    """
    abstract type LocalChannel{A} <: ChannelType{A} end

    ######################################################################################

    function gradient_for_Ω!(
        result,
        Ω::SignalType{F,Complex{F}},        # Complex signal.
        grid::Integrations.IntegrationType,
        ϕα::AbstractVector,
        ϕβ::AbstractVector,
    ) where {F}
        t̄ = Integrations.lattice(grid)
        ∂̄ = array(Complex{F}, size(t̄), LABEL)
        Φ = (t, ∂, ϕα, ϕβ) -> (real(∂)*ϕα + imag(∂)*ϕβ)

        for k in 1:Parameters.count(Ω)
            ∂̄ = Signals.partial(k, Ω, t̄; result=∂̄)
            result[k] = Integrations.integrate(grid, Φ, ∂̄, ϕα, ϕβ)
        end

        return result
    end

    function gradient_for_ν!(
        result,
        ν::SignalType{F,F},
        Ω::SignalType{F,Complex{F}},        # Complex signal
        grid::Integrations.IntegrationType,
        ϕα::AbstractVector,
        ϕβ::AbstractVector,
    )
        # CALCULATE GRADIENT FOR FREQUENCY PARAMETERS
        t̄ = Integrations.lattice(grid)
        ∂̄ = array(F, size(t̄), LABEL)
        Φ = (t, ∂, Ω, ϕα, ϕβ) -> ∂ * (t * (real(Ω)*ϕβ - imag(Ω)*ϕα))

        Ω̄ = array(Complex{F}, size(t̄), LABEL)
        Ω̄ = Signals.valueat(Ω, t̄; result=Ω̄)

        for k in 1:Parameters.count(ν)
            ∂̄ = Signals.partial(k, ν, t̄; result=∂̄)
            result[k] = Integrations.integrate(grid, Φ, ∂̄, Ω̄, ϕα, ϕβ)
        end

        return result
    end

    # TODO: There are versions for real channel Ω, and for polar channel.
    #= TODO: There's also a no-RWA version
        but its interface would be the same as polar
        so I think it needs to be categorically separated. =#

    ######################################################################################

    struct QubitChannel{F} <: LocalChannel{TruncatedBosonicAlgebra{F}}
        q::Int
        Ω::SignalType{F,Complex{F}}
        ν::SignalType{F,F}
    end

    function Parameters.count(channel::QubitChannel)
        return sum(
            Parameters.count(channel.Ω),
            Parameters.count(channel.ν),
        )
    end

    function Parameters.values(channel::QubitChannel)
        return [
            Parameters.values(channel.Ω);
            Parameters.values(channel.ν);
        ]
    end

    function Parameters.names(channel::QubitChannel)
        return [
            ["[Ω]$name" for name in Parameters.names(channel.Ω)];
            ["[ν]$name" for name in Parameters.names(channel.ν)];
        ]
    end

    function Parameters.bind!(channel::QubitChannel{F}, x::AbstractVector{F}) where {F}
        L = Parameters.count(channel.Ω)
        xΩ = @view(x[1:L])
        xν = @view(x[1+L:end])

        Parameters.bind!(channel.ν, xΩ)
        Parameters.bind!(channel.ν, xν)
    end

    function LocallyDrivenDevices.drivequbit(channel::QubitChannel)
        return channel.q
    end

    function Devices.driveoperator(channel::QubitChannel, ā, t::Real; result)
        a = @view(ā[:,:,1,channel.q])
        ν = Signals.valueat(channel.ν, t)
        Ω = Signals.valueat(channel.Ω, t)
        e = exp(im * ν * t)

        result .= 0
        result .+= (Ω * e)  .* a
        result .+= (Ω * e)' .* a'
        return result
    end

    function Devices.gradeoperator(
        channel::QubitChannel,
        ā,
        j::Int,
        t::Real;
        result=nothing,
    )
        a = @view(ā[:,:,1,channel.q])
        ν = Signals.valueat(channel.ν, t)
        e = exp(im * ν * t)

        phase = Bool(j & 1) ? 1 : im    # Odd j -> "α" gradient operator; even j  -> "β"

        result .= 0
        result .+= (phase * e)  .* a
        result .+= (phase * e)' .* a'
        return result
    end

    function Devices.gradient(
        channel::QubitChannel,
        grid::Integrations.IntegrationType,
        ϕ̄::AbstractMatrix;
        result,
    )
        ϕα = @view(ϕ̄[:,1])
        ϕβ = @view(ϕ̄[:,2])

        L = Parameters.count(channel.Ω)
        gΩ = @view(result[1:L])
        gν = @view(result[1+L:end])

        gradient_for_Ω!(gΩ, channel.Ω, grid, ϕα, ϕβ)
        gradient_for_ν!(gν, channel.ν, channel.Ω, grid, ϕα, ϕβ)

        return result
    end

    # ######################################################################################

    # struct RealChannel{F} <: LocalChannel{TruncatedBosonicAlgebra{F}}
    #     q::Int
    #     Ω::SignalType{F,F}
    #     ν::SignalType{F}
    # end
    # LocallyDrivenDevices.drivequbit(channel::RealChannel) = channel.q
    # # RealChannel:    Ω [exp(iνt) a + h.t]    Ω and ν are real signals

    # ######################################################################################

    # struct PolarChannel{F} <: LocalChannel{TruncatedBosonicAlgebra{F}}
    #     q::Int
    #     Ω::SignalType{F,F}
    #     ϕ::SignalType{F,F}
    #     ν::SignalType{F}
    # end
    # LocallyDrivenDevices.drivequbit(channel::PolarChannel) = channel.q
    # # PolarChannel:   Ω exp[i(νt+ϕ)] + h.t.   Ω, ϕ, ν are real signals

    # ######################################################################################

    # struct NoRWAChannel{F} <: LocalChannel{TruncatedBosonicAlgebra{F}}
    #     q::Int
    #     Ω::SignalType{F,F}
    #     ϕ::SignalType{F,F}
    #     ν::SignalType{F}
    # end
    # LocallyDrivenDevices.drivequbit(channel::NoRWAChannel) = channel.q
    # # NoRWAChannel:   Ω sin(νt+ϕ) (a + a')    Ω, ϕ, ν are real signals

end