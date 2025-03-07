module DipoleDrives
    import ..ModularFramework: LocalDrive
    import ..ModularFramework: TruncatedBosonicAlgebra

    import CtrlVQE: Parameters
    import CtrlVQE: Integrations, Signals, Devices, CostFunctions

    import TemporaryArrays: @temparray

    """
        DipoleDrive(q, ω, Ω, Δ)

    A drive term representing interaction of a bosonic mode with an electric dipole,
        in the rotating wave approximation.

    `` \\hat V = Ω(t) e^{i [Δ(t)+ω] t} \\hat a_q + {\\rm h.t.} ``

    # Parameters
    - `q`: the integer index identifying which qubit this drive applies to
    - `ω`: the resonance frequency of the qubit
    - `Ω`: a `SignalType` representing the complex drive strength over time
    - `Δ`: a `SignalType` representing the detuning over time

    ```jldoctests
    julia> grid = TemporalLattice(20.0, 400);

    julia> Ω = Constant(2.0+1.0im);

    julia> Δ = Constant(0.0);

    julia> using CtrlVQE.ModularFramework;

    julia> A = TruncatedBosonicAlgebra{3,2};

    julia> drive = DipoleDrive{A}(1, 4.82, Ω, Δ);

    julia> validate(drive; algebra=A(), grid=grid, t=10.0);

    julia> ā0 = Devices.localalgebra(A());

    julia> Devices.driveoperator(drive, ā0, 10.0)
    3×3 Matrix{ComplexF64}:
            0.0+0.0im      -0.0693931-2.23499im         0.0+0.0im
     -0.0693931+2.23499im         0.0+0.0im      -0.0981367-3.16075im
            0.0+0.0im      -0.0981367+3.16075im         0.0+0.0im

    ```

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

    function Devices.driveoperator(
        drive::DipoleDrive{A,F},
        ā,
        t::Real;
        result=nothing,
    ) where {A,F}
        isnothing(result) && (result = Array{Complex{F}}(undef, size(ā)[1:2]))
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
        drive::DipoleDrive{A,F},
        ā,
        j::Int,
        t::Real;
        result=nothing,
    ) where {A,F}
        isnothing(result) && (result = Array{Complex{F}}(undef, size(ā)[1:2]))
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
        result=nothing,
    ) where {A,F}
        isnothing(result) && (result = Array{F}(undef, Parameters.count(drive)))
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