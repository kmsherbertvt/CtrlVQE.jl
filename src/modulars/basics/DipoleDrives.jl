module DipoleDrives
    import ..ModularFramework: LocalDrive
    import ..ModularFramework: TruncatedBosonicAlgebra

    import CtrlVQE: Parameters
    import CtrlVQE: Integrations, Signals, Devices, CostFunctions

    import TemporaryArrays: @temparray

    """
    TODO
    """
    struct DipoleDrive{
        A <: TruncatedBosonicAlgebra,
        F,                  # Float type
        SΩ <: Signals.SignalType{F,Complex{F}},
        SΔ <: Signals.SignalType{F,F},
    } <: LocalDrive{A}
        q::Int
        ω::F
        Ω::SΩ
        Δ::SΔ

        function DipoleDrive{A}(
            q::Int,
            ω::F,
            Ω::SΩ,
            Δ::SΔ,
        ) where {
            F,
            A<:TruncatedBosonicAlgebra,
            SΩ<:Signals.SignalType{F,Complex{F}},
            SΔ<:Signals.SignalType{F,F},
        }
            return new{A,F,SΩ,SΔ}(q, ω, Ω, Δ)
        end
    end

    ######################################################################################
    #= PARAMETER INTERFACE =#

    function Parameters.count(drive::DipoleDrive)
        return sum((
            Parameters.count(drive.Ω),
            Parameters.count(drive.Δ),
        ))
    end

    function Parameters.values(drive::DipoleDrive)
        return [
            Parameters.values(drive.Ω);
            Parameters.values(drive.Δ);
        ]
    end

    function Parameters.names(drive::DipoleDrive)
        return [
            ["[Ω]$name" for name in Parameters.names(drive.Ω)];
            ["[Δ]$name" for name in Parameters.names(drive.Δ)];
        ]
    end

    function Parameters.bind!(drive::DipoleDrive, x::AbstractVector)
        L = Parameters.count(drive.Ω)
        xΩ = @view(x[1:L])
        xΔ = @view(x[1+L:end])

        Parameters.bind!(drive.Ω, xΩ)
        Parameters.bind!(drive.Δ, xΔ)
    end

    ######################################################################################

    Devices.ngrades(drive::Type{<:DipoleDrive}) = 2
    Devices.drivequbit(drive::DipoleDrive) = drive.q

    function Devices.driveoperator(drive::DipoleDrive, ā, t::Real; result)
        a = @view(ā[:,:,1,drive.q])
        Ω = Signals.valueat(drive.Ω, t)
        ν = Signals.valueat(drive.Δ, t) + drive.ω
        z = Ω * cis(ν * t)

        # COMPUTE OPERATOR
        result .= 0
        result .+= z  .* a
        result .+= z' .* a'
        return result
    end

    function Devices.gradeoperator(
        drive::DipoleDrive,
        ā,
        j::Int,
        t::Real;
        result=nothing,
    )
        a = @view(ā[:,:,1,drive.q])
        ν = Signals.valueat(drive.Δ, t) + drive.ω
        e = cis(ν * t)
        phase = Bool(j & 1) ? 1 : im    # Odd j -> real gradeop; even j  -> imag

        # COMPUTE OPERATOR
        result .= 0
        result .+= (phase * e ) .* a
        result .+= (phase'* e') .* a'
        return result
    end

    function Devices.gradient(
        drive::DipoleDrive{A,F},
        grid::Integrations.IntegrationType,
        ϕ::AbstractMatrix;
        result,
    ) where {A,F}
        # ALIAS RELEVANT GRADIENT SIGNALS
        ϕα = @view(ϕ[:,1])
        ϕβ = @view(ϕ[:,2])

        # ALLOCATE ARRAYS FOR TIME INTEGRATION
        Ω = @temparray(Complex{F}, length(grid), :gradient, :amplitude)
        Signals.valueat(drive.Ω, grid; result=Ω)
        ∂ = @temparray(Complex{F}, length(grid), :gradient, :partial)

        # DEFINE INTEGRANDS FOR TIME INTEGRATION
        ΦΩ = (t, ∂, ϕα, ϕβ) -> (real(∂)*ϕα + imag(∂)*ϕβ)
        ΦΔ = (t, ∂, Ω, ϕα, ϕβ) -> ∂ * (t * (real(Ω)*ϕβ - imag(Ω)*ϕα))

        #= AMPLITUDE GRADIENTS =#
        L = Parameters.count(drive.Ω)
        for k in 1:L
            Signals.partial(k, drive.Ω, grid; result=∂)
            result[k] = Integrations.integrate(grid, ΦΩ, ∂, ϕα, ϕβ)
        end

        #= FREQUENCY GRADIENTS =#
        for k in 1+L:length(result)
            Signals.partial(k, drive.Δ, grid; result=∂)
            result[k] = Integrations.integrate(grid, ΦΔ, ∂, Ω, ϕα, ϕβ)
        end

        return result
    end

    ######################################################################################

    # struct DipoleAmplitudePenalty{F} <: CostFunctions.CostFunctionType{F}

    # end

    #= TODO: Simple wrapper around amplitude parameters. Gradient is zero for others.

    But, hey, as long as frequency signal has no parameters, the wrapper is redundant.

    And we COULD re-spec DrivePenalty to use another ParameterMap type,
        which further maps drive parameters to the actual penalty function.
    Then, here, we need only define an AMPLITUDE and FREQUENCY parameter mapper,
        which implement map_values and map_gradient.
    Different signatures than the device one, a bit weird... =#

end