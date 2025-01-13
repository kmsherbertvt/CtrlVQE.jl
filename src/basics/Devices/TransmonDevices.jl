module TransmonDevices
    export TransmonDevice

    import ..CtrlVQE: LAT
    import ..CtrlVQE: Parameters
    import ..CtrlVQE: Integrations, Signals, Devices

    import ..CtrlVQE.Quples: Quple
    import ..CtrlVQE.Signals: SignalType
    import ..CtrlVQE: Constant, Constrained

    import TemporaryArrays: @temparray

    import LinearAlgebra: mul!

    """
        TransmonDevice{m}(ω, δ, g, quples, q, Ω, Δ)

    A transmon device where control signals and drive frequencies are signals.

    Drives are approximated with RWA.

    # Type Parameters
    - `F`: (inferred from arguments) the float type of this device
    - `m`: the integer number of levels per transmon

    # Parameters
    - `ω`: an (abstract) vector of qubit resonance frequencies
    - `δ`: an (abstract) vector of qubit anharmonicities
    - `g`: an (abstract) vector of qubit coupling strengths
    - `quples`: an (abstract) vector of quples identifying each coupling
    - `q`: the qubits corresponding to each drive.
    - `Ω`: an (abstract) vector of control signals for each drive. May be real or complex.
    - `Δ`: an (abstract) vector of detunings for each drive frequency. Must be real.

    """
    struct TransmonDevice{
        F,
        m,
        SΩ<:SignalType,
        SΔ<:SignalType,
    } <: Devices.LocallyDrivenDevice{F}
        # QUBIT LISTS
        ω::Vector{F}
        δ::Vector{F}
        # COUPLING LISTS
        g::Vector{F}
        quples::Vector{Quple}
        # SIGNAL LISTS
        q::Vector{Int}
        Ω::Vector{SΩ}
        Δ::Vector{SΔ}

        function TransmonDevice{m}(
            ω::AbstractVector{<:Real},
            δ::AbstractVector{<:Real},
            g::AbstractVector{<:Real},
            quples::AbstractVector{Quple},
            q::AbstractVector{<:Int},
            Ω::AbstractVector{<:SignalType{F,R}},
            Δ::AbstractVector{<:SignalType{F,F}},
        ) where {m,F,R}
            # VALIDATE PARALLEL LISTS ARE CONSISTENT SIZE
            @assert length(ω) == length(δ) ≥ 1 # NUMBER OF QUBITS
            @assert length(g) == length(quples)             # NUMBER OF COUPLINGS
            @assert length(q) == length(Ω) == length(Δ)     # NUMBER OF SIGNALS

            # VALIDATE QUBIT INDICES
            for (p,q) in quples
                @assert 1 <= p <= length(ω)
                @assert 1 <= q <= length(ω)
            end
            for q_ in q
                @assert 1 <= q_ <= length(ω)
            end

            # VALIDATE THAT THE HILBERT SPACE HAS SOME VOLUME...
            @assert m ≥ 2

            # INFER TYPE
            @assert real(R) == F
            SΩ = eltype(Ω)
            SΔ = eltype(Δ)

            # STANDARDIZE TYPING
            return new{F,m,SΩ,SΔ}(
                convert(Array{F}, ω),
                convert(Array{F}, δ),
                convert(Array{F}, g),
                quples,
                convert(Array{Int}, q),
                convert(Array{SΩ}, Ω),
                convert(Array{SΔ}, Δ),
            )
        end
    end

    """
        TransmonDevice{m}(ω, δ, g, quples; kwargs...)

    Convenience constructor allowing a more semantic approach to inputting signals.

    # Keyword Arguments
    - `q`: may be a an (abstract) vector of `Int`.
            Defaults to one drive for each qubit (i.e. `1:length(ω)`)
    - `Ω`: may be a `SignalType` or an (abstract) vector of `SignalTypes`.
            If a single `SignalType` is provided, it is duplicated for each drive.
    - `Δ`: may be a `Bool` or a `SignalType` or an (abstract) vector of `SignalTypes`.
            If a single `SignalType` is provided, it is duplicated for each drive.
            If `Δ=true`, drive frequencies are constant signals initialized on resonance.
            If `Δ=false` (the default), drive frequencies are *frozen* on resonance.

    """
    function TransmonDevice{m}(
        ω::AbstractVector{<:Real},
        δ::AbstractVector{<:Real},
        g::AbstractVector{<:Real},
        quples::AbstractVector{Quple};
        q=1:length(ω),
        Ω=Constant(zero(ComplexF64)),
        Δ=false,
    ) where {m}
        Ω_ = Ω isa Vector ? Ω : [deepcopy(Ω) for _ in q]
        F = isempty(Ω_) ? promote_eltype(ω,δ,g) : Signals.parametertype(first(Ω_))

        Δ_ = Δ isa Vector ? Δ :
            Δ == true ? [Constant(zero(F)) for _ in q] :
            Δ == false ? [Constrained(Constant(zero(F)), :A) for _ in q] :
            [deepcopy(Δ) for _ in q]

        return TransmonDevice{m}(ω, δ, g, quples, q, Ω_, Δ_)
    end

    #= `Parameters` INTERFACE =#

    function Parameters.count(device::TransmonDevice)
        LΩ = sum(Parameters.count, device.Ω)
        LΔ = sum(Parameters.count, device.Δ)
        return LΩ + LΔ
    end

    function Parameters.names(device::TransmonDevice)
        allnames = String[]
        for (i,signal) in enumerate(device.Ω)
            q = Devices.drivequbit(device, i)
            append!(allnames, "Ω$i(q$q):$name" for name in Parameters.names(signal))
        end
        for (i,signal) in enumerate(device.Δ)
            q = Devices.drivequbit(device, i)
            append!(allnames, "Δ$i(q$q):$name" for name in Parameters.names(signal))
        end
        return allnames
    end

    function Parameters.values(device::TransmonDevice)
        return vcat(
            (Parameters.values(signal) for signal in device.Ω)...,
            (Parameters.values(signal) for signal in device.Δ)...,
        )
    end

    function Parameters.bind!(
        device::TransmonDevice{F,m},
        x::AbstractVector{F},
    ) where {F,m}
        Δ = 0
        for signal in device.Ω
            L = Parameters.count(signal)
            Parameters.bind!(signal, @view(x[Δ+1:Δ+L]))
            Δ += L
        end
        for signal in device.Δ
            L = Parameters.count(signal)
            Parameters.bind!(signal, @view(x[Δ+1:Δ+L]))
            Δ += L
        end
        return device
    end

    #= `LocallyDrivenDevice` INTERFACE =#
    Devices.drivequbit(device::TransmonDevice, i::Int) = device.q[i]
    Devices.gradequbit(device::TransmonDevice, j::Int) = device.q[((j-1) >> 1) + 1]

    #= `DeviceType` INTERFACE =#
    Devices.ndrives(device::TransmonDevice) = length(device.q)
    Devices.ngrades(device::TransmonDevice) = 2 * length(device.q)
    Devices.nlevels(device::TransmonDevice{F,m}) where {F,m} = m
    Devices.nqubits(device::TransmonDevice) = length(device.ω)
    Devices.noperators(device::TransmonDevice) = 1

    function Devices.localalgebra(device::TransmonDevice; result=nothing)
        isnothing(result) && return Devices._localalgebra(device)

        m = Devices.nlevels(device)
        n = Devices.nqubits(device)

        a = @temparray(eltype(device), (m,m), :localalgebra)
        a .= 0
        for i in 1:m-1
            a[i,i+1] = √i
        end

        for q in 1:n
            result[:,:,1,q] .= a
        end
        return result
    end

    function Devices.qubithamiltonian(device::TransmonDevice, ā, q::Int; result=nothing)
        a = @view(ā[:,:,1,q])
        result === nothing && (result = Matrix{Complex{eltype(device)}}(undef, size(a)))

        # PREP AN IDENTITY MATRIX
        Im = @temparray(Bool, size(a), :qubithamiltonian)
        LAT.basisvectors!(Im)

        # CONSTRUCT ω a'a - δ/2 a'a'aa
        result .= 0
        result .-= (device.δ[q]/2) .* Im    #       - δ/2    I
        result = LAT.rotate!(a', result)    #       - δ/2   a'a
        result .+=  device.ω[q]    .* Im    # ω     - δ/2   a'a
        result = LAT.rotate!(a', result)    # ω a'a - δ/2 a'a'aa
        return result
    end

    function Devices.staticcoupling(device::TransmonDevice, ā; result=nothing)
        d = size(ā,1)
        result === nothing && (result = Matrix{Complex{eltype(device)}}(undef, d,d))

        # PREP A MATRIX FOR p'q
        aTa = @temparray(eltype(ā), size(result), :staticcoupling)

        result .= 0
        for pq in eachindex(device.g)
            g = device.g[pq]
            p, q = device.quples[pq]

            aTa = mul!(aTa, (@view(ā[:,:,1,p]))', @view(ā[:,:,1,q])) # p'q
            result .+= g .* aTa             # g ( p'q )
            result .+= g .* aTa'            # g ( p'q + q'p )
        end
        return result
    end

    function Devices.driveoperator(
        device::TransmonDevice, ā, i::Int, t::Real;
        result=nothing,
    )
        q = Devices.drivequbit(device, i)
        a = @view(ā[:,:,1,q])
        result === nothing && (result = Matrix{Complex{eltype(device)}}(undef, size(a)))

        # COMPUTE RWA-MODULATED DRIVE STRENGTH
        Ω = Signals.valueat(device.Ω[i], t)
        ν = Signals.valueat(device.Δ[i], t) + device.ω[q]
        z = Ω * cis(ν * t)

        # COMPUTE OPERATOR
        result .= 0
        result .+= z  .* a
        result .+= z' .* a'
        return result
    end

    function Devices.gradeoperator(
        device::TransmonDevice, ā, j::Int, t::Real;
        result=nothing,
    )
        i = ((j-1) >> 1) + 1
        q = Devices.gradequbit(device, j)
        a = @view(ā[:,:,1,q])
        result === nothing && (result = Matrix{Complex{eltype(device)}}(undef, size(a)))

        # COMPUTE RWA-MODULATION AND OPERATOR MULTIPLEXING
        ν = Signals.valueat(device.Δ[i], t) + device.ω[q]
        e = cis(ν * t)
        phase = Bool(j & 1) ? 1 : im    # Odd j -> real gradeop; even j  -> imag

        # COMPUTE OPERATOR
        result .= 0
        result .+= (phase * e ) .* a
        result .+= (phase'* e') .* a'

        #= NOTE: The biggest difference between real and complex amplitudes is that,
            with real amplitudes,
            the drive operators ``za+z'a'`` are their own gradient operators,
            while with complex amplitudes,
            you need to worry about complementary operators ``i(za-z'a')``. =#

        return result
    end

    function Devices.gradient(
        device::TransmonDevice{F},
        grid::Integrations.IntegrationType,
        ϕ::AbstractMatrix;
        result=nothing,
    ) where {F}
        isnothing(result) && (result=Vector{F}(undef, Parameters.count(device)))

        # ALLOCATE ARRAYS FOR TIME INTEGRATION
        Ω = @temparray(Complex{F}, length(grid), :gradient, :amplitude)
        ∂ = @temparray(Complex{F}, length(grid), :gradient, :partial)

        # DEFINE INTEGRANDS FOR TIME INTEGRATION
        ΦΩ = (t, ∂, ϕα, ϕβ) -> (real(∂)*ϕα + imag(∂)*ϕβ)
        ΦΔ = (t, ∂, Ω, ϕα, ϕβ) -> ∂ * (t * (real(Ω)*ϕβ - imag(Ω)*ϕα))

        # OFFSETS FOR GRADIENT INDICES
        ΔΩ = 0
        ΔΔ = sum(Parameters.count, device.Ω)

        for i in 1:Devices.ndrives(device)
            j = 2i - 1
            ϕα = @view(ϕ[:,j])
            ϕβ = @view(ϕ[:,j+1])

            signal_Ω = device.Ω[i]
            Signals.valueat(signal_Ω, grid; result=Ω)
            signal_Δ = device.Δ[i]

            #= AMPLITUDE GRADIENTS =#
            L = Parameters.count(signal_Ω)
            for k in 1:L
                Signals.partial(k, signal_Ω, grid; result=∂)
                result[ΔΩ+k] = Integrations.integrate(grid, ΦΩ, ∂, ϕα, ϕβ)
            end
            ΔΩ += L

            #= FREQUENCY GRADIENTS =#
            L = Parameters.count(signal_Δ)
            for k in 1:L
                Signals.partial(k, signal_Δ, grid; result=∂)
                result[ΔΔ+k] = Integrations.integrate(grid, ΦΔ, ∂, Ω, ϕα, ϕβ)
            end
            ΔΔ += L
        end
        return result
    end

    """
        Prototype(::Type{TransmonDevice{F,m}}, n::Int; kwargs...)

    A prototypical `TransmonDevice` with the following decisions:
    - All anharmonicities are constant.
    - Couplings are linear.
    - Each coupling strength equals the difference in resonance frequencies
        of the coupled qubits.
    - By default, all resonance frequencies are equally spaced
        (so, coupling strengths are constant)
        but this can be controlled through kwargs.
    - Drives match those of the defaults when using the kwarg constructor.

    Default parameters are vaguely reminiscent of IBM devices circa 2021,
        although the default behavior of linearly-spaced resonance frequencies
        is not realistic and should be avoided outside of testing/benchmarking.

    # Keyword Arguments
    - `ω0=4.82`: resonance frequency of first qubit.
    - `Δω=0.02`: the spacing in resonance frequencies between adjacent qubits.
        When passed as a float (including the default),
            resonance frequencies are linearly spaced.
        Instead, you can pass this as an explicit vector with `n-1` elements.
    - `δ0=0.30`: the constant anharmonicity.
    - `T=10.0`: the pulse duration,
        but this has no effect since the default signals are constant.

    """
    function Devices.Prototype(
        ::Type{TransmonDevice{F,m}}, n::Int;
        ω0=2π*4.82, Δω=2π*0.02, δ0=2π*0.30, T=10.0,
    ) where {F,m}
        !(Δω isa AbstractVector) && (Δω = fill(F(Δω), n-1))

        ω = fill(F(ω0), n); ω .+= [0; cumsum(Δω)]
        δ = fill(F(δ0), n)
        g = deepcopy(Δω)
        quples = [Quple(q,q+1) for q in 1:n-1]
        return TransmonDevice{m}(ω, δ, g, quples)
    end

end