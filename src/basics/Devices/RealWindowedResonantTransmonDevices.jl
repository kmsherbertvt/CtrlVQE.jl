module RealWindowedResonantTransmonDevices
    export RealWindowedResonantTransmonDevice

    import CtrlVQE: LAT
    import CtrlVQE: Parameters, Prototypes
    import CtrlVQE: Integrations, Devices

    import CtrlVQE.Quples: Quple

    import TemporaryArrays: @temparray

    import LinearAlgebra: mul!

    """
        RealWindowedResonantTransmonDevice{m}(ω, δ, g, quples, T, Ω)

    A minimalist transmon device with real-valued constant windows driven on resonance.

    Windows are always equally spaced.
    Drives are approximated with RWA.

    # Type Parameters
    - `F`: (inferred from arguments) the float type of this device
    - `m`: the integer number of levels per transmon

    # Parameters
    - `ω`: an (abstract) vector of qubit resonance frequencies
    - `δ`: an (abstract) vector of qubit anharmonicities
    - `g`: an (abstract) vector of qubit coupling strengths
    - `quples`: an (abstract) vector of quples identifying each coupling
    - `T`: the total pulse duration applied on this device
    - `Ω`: a matrix of complex amplitudes.
        `Ω[w,q]` is the amplitude applied on qubit `q` in the time-window indexed by `w`.

    """
    struct RealWindowedResonantTransmonDevice{F,m} <: Devices.LocallyDrivenDevice{F}
        # QUBIT LISTS
        ω::Vector{F}
        δ::Vector{F}
        # COUPLING LISTS
        g::Vector{F}
        quples::Vector{Quple}
        # DRIVE LISTS
        T::F
        Ω::Matrix{F}

        function RealWindowedResonantTransmonDevice{m}(
            ω::AbstractVector{<:Real},
            δ::AbstractVector{<:Real},
            g::AbstractVector{<:Real},
            quples::AbstractVector{Quple},
            T::Real,
            Ω::AbstractMatrix{<:AbstractFloat},
        ) where {m}
            # VALIDATE PARALLEL LISTS ARE CONSISTENT SIZE
            @assert length(ω) == length(δ) == size(Ω,2) ≥ 1 # NUMBER OF QUBITS
            @assert length(g) == length(quples)             # NUMBER OF COUPLINGS

            # VALIDATE QUBIT INDICES
            for (p,q) in quples
                @assert 1 <= p <= length(ω)
                @assert 1 <= q <= length(ω)
            end

            # VALIDATE THAT THE HILBERT SPACE HAS SOME VOLUME...
            @assert m ≥ 2

            # INFER TYPE
            F = promote_type(eltype(ω), eltype(δ), eltype(g), typeof(T), eltype(Ω))

            # STANDARDIZE TYPING
            return new{F,m}(
                convert(Array{F}, ω),
                convert(Array{F}, δ),
                convert(Array{F}, g),
                quples,
                convert(F, T),
                convert(Array{F}, Ω),
            )
        end
    end

    """
        RealWindowedResonantTransmonDevice{m}(ω, δ, g, quples, T, W::Int)

    Convenience constructor where `Ω` is initialized to zero, with `W` time windows.

    """
    function RealWindowedResonantTransmonDevice{m}(
        ω::AbstractVector{<:Real},
        δ::AbstractVector{<:Real},
        g::AbstractVector{<:Real},
        quples::AbstractVector{Quple},
        T::Real,
        W::Int,
    ) where {m}
        F = promote_type(eltype(ω), eltype(δ), eltype(g), typeof(T), Float16)
        Ω = zeros(F, W, length(ω))
        return RealWindowedResonantTransmonDevice{m}(ω, δ, g, quples, T, Ω)
    end

    #= ALIAS TYPE FOR SHORTER SIGNATURES =#
    const RWRTDevice = RealWindowedResonantTransmonDevice

    #= `Parameters` INTERFACE =#

    Parameters.count(device::RWRTDevice) = length(device.Ω)
    Parameters.names(device::RWRTDevice) = [
        "Ω$q:α$w" for w in axes(device.Ω,1) for q in axes(device.Ω,2)
    ]
    Parameters.values(device::RWRTDevice) = vec(device.Ω)

    function Parameters.bind!(
        device::RWRTDevice{F,m},
        x::AbstractVector{F},
    ) where {F,m}
        vec(device.Ω) .= x
        return device
    end

    #= `LocallyDrivenDevice` INTERFACE =#
    Devices.drivequbit(device::RWRTDevice, i::Int) = i
    Devices.gradequbit(device::RWRTDevice, j::Int) = j

    #= `DeviceType` INTERFACE =#
    Devices.ndrives(device::RWRTDevice) = length(device.ω)
    Devices.ngrades(device::RWRTDevice) = length(device.ω)
    Devices.nlevels(device::RWRTDevice{F,m}) where {F,m} = m
    Devices.nqubits(device::RWRTDevice) = length(device.ω)
    Devices.noperators(device::RWRTDevice) = 1

    function Devices.localalgebra(device::RWRTDevice; result=nothing)
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

    function Devices.qubithamiltonian(device::RWRTDevice, ā, q::Int; result=nothing)
        a = @view(ā[:,:,1,q])
        result === nothing && (result = Matrix{Complex{eltype(device)}}(undef, size(a)))

        # PREP AN IDENTITY MATRIX
        Im = @temparray(Bool, size(a), :qubithamiltonian)
        LAT.basisvectors(size(a,1); result=Im)

        # CONSTRUCT ω a'a - δ/2 a'a'aa
        result .= 0
        result .-= (device.δ[q]/2) .* Im    #       - δ/2    I
        result = LAT.rotate!(a', result)    #       - δ/2   a'a
        result .+=  device.ω[q]    .* Im    # ω     - δ/2   a'a
        result = LAT.rotate!(a', result)    # ω a'a - δ/2 a'a'aa
        return result
    end

    function Devices.staticcoupling(device::RWRTDevice, ā; result=nothing)
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

    function Devices.driveoperator(device::RWRTDevice, ā, i::Int, t::Real;result=nothing)
        q = Devices.drivequbit(device, i)
        a = @view(ā[:,:,1,q])
        result === nothing && (result = Matrix{Complex{eltype(device)}}(undef, size(a)))

        # COMPUTE RWA-MODULATED DRIVE STRENGTH
        w = __windowindex(device, t)
        z = device.Ω[w,q] * cis(device.ω[q] * t)

        # COMPUTE OPERATOR
        result .= 0
        result .+= z  .* a
        result .+= z' .* a'
        return result
    end

    function Devices.gradeoperator(
        device::RWRTDevice, ā, j::Int, t::Real;
        result=nothing,
    )
        q = Devices.gradequbit(device, j)
        a = @view(ā[:,:,1,q])
        result === nothing && (result = Matrix{Complex{eltype(device)}}(undef, size(a)))

        # COMPUTE RWA-MODULATION AND OPERATOR MULTIPLEXING
        e = cis(device.ω[q] * t)

        # COMPUTE OPERATOR
        result .= 0
        result .+= e  .* a
        result .+= e' .* a'

        #= NOTE: The biggest difference between real and complex amplitudes is that,
            with real amplitudes,
            the drive operators ``za+z'a'`` are their own gradient operators,
            while with complex amplitudes,
            you need to worry about complementary operators ``i(za-z'a')``. =#

        return result
    end

    function Devices.gradient(
        device::RWRTDevice{F},
        grid::Integrations.IntegrationType,
        ϕ::AbstractMatrix;
        result=nothing,
    ) where {F}
        isnothing(result) && (result=Vector{F}(undef, Parameters.count(device)))
        ∇f = reshape(result, size(device.Ω))

        # MANUALLY INTEGRATE OVER TIME TO EXPLOIT WINDOWED STRUCTURE
        ∇f .= 0
        for i in eachindex(grid)
            τ = Integrations.stepat(grid, i)
            t = Integrations.timeat(grid, i)
            w = __windowindex(device, t)
            ∇f[w,:] .+= τ .* @view(ϕ[1+i,:])
        end
        #= NOTE: It may be appropriate to implement a more generic `gradient`
            to accommodate integration types with a more complicated `integrate`.
        =#
        return result
    end

    """
        __windowindex(device, t)

    Infer the column index of `device.Ω` i which time `t` falls.

    """
    function __windowindex(device::RWRTDevice, t::Real)
        W = size(device.Ω,1)                        # TOTAL WINDOW COUNT
        w = 1 + round(Int, (t ÷ (device.T / W)))    # WINDOW INDEX FOR TIME t
        w > W && (w = W)                            # HANDLE t ≥ T
        return w
    end

    """
        Prototype(::Type{RealWindowedResonantTransmonDevice{F,m}}, n::Int; kwargs...)

    A prototypical `RealWindowedResonantTransmonDevice` with the following decisions:
    - All anharmonicities are constant.
    - Couplings are linear.
    - Each coupling strength equals the difference in resonance frequencies
        of the coupled qubits.
    - By default, all resonance frequencies are equally spaced
        (so, coupling strengths are constant)
        but this can be controlled through kwargs.

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
    - `T=10.0`: total pulse duration.
    - `W=1`: number of window segments.

    ```jldoctests
    julia> grid = TemporalLattice(20.0, 400);

    julia> device = Prototype(RWRTDevice{Float64,3}; n=2);

    julia> validate(device; grid=grid, t=10.0);

    julia> nlevels(device)
    3
    julia> nqubits(device)
    2
    ```

    """
    function Prototypes.Prototype(
        ::Type{RealWindowedResonantTransmonDevice{F,m}};
        n=2, T=10.0,
        ω0=2π*4.82, Δω=2π*0.02, δ0=2π*0.30, W=1,
    ) where {F,m}
        !(Δω isa AbstractVector) && (Δω = fill(F(Δω), n-1))

        ω = fill(F(ω0), n); ω .+= [0; cumsum(Δω)]
        δ = fill(F(δ0), n)
        g = deepcopy(Δω)
        quples = [Quple(q,q+1) for q in 1:n-1]
        return RealWindowedResonantTransmonDevice{m}(ω, δ, g, quples, T, W)
    end

end