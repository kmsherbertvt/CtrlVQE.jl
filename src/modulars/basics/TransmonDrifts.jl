module TransmonDrifts
    import ..ModularFramework: DriftType
    import ..ModularFramework: TruncatedBosonicAlgebra

    import CtrlVQE: Quple
    import CtrlVQE: LAT
    import CtrlVQE: Devices

    import TemporaryArrays: @temparray

    import LinearAlgebra: mul!

    """
        TransmonDrift(ω, δ, g, quples)

    A static Hamiltonian for architectures of `n` transmons with fixed-couplings.

    ``\\hat H_0 = \\sum_q ω_q \\hat a^\\dagger_q \\hat a_q
    - \\sum_q δ_q \\hat a^\\dagger_q \\hat a^\\dagger_q \\hat a_q \\hat a_q
    + \\sum_{⟨pq⟩} g_{pq} (\\hat a^\\dagger_p \\hat a_q + \\hat a^\\dagger_q \\hat a_p)``

    # Parameters
    - `ω`: a vector of `n` resonance frequencies
    - `δ`: a vector of `n` anharmonicities
    - `g`: a vector of coupling strengths
    - `quples`: a vector of `Quple`
        identifying the qubit pairs associated with each `g[i]`

    ```jldoctests
    julia> using CtrlVQE.ModularFramework;

    julia> A = TruncatedBosonicAlgebra{3,2};

    julia> drift = TransmonDrift{A}([4.82, 4.84], [0.30, 0.30], [0.02], [Quple(1,2)]);

    julia> validate(drift; algebra=A());

    julia> ā0 = Devices.localalgebra(A());

    julia> Devices.qubithamiltonian(drift, ā0, 1)
    3×3 Matrix{ComplexF64}:
     0.0+0.0im   0.0+0.0im   0.0+0.0im
     0.0+0.0im  4.82+0.0im   0.0+0.0im
     0.0+0.0im   0.0+0.0im  9.34+0.0im

    julia> Devices.qubithamiltonian(drift, ā0, 2)
    3×3 Matrix{ComplexF64}:
     0.0+0.0im   0.0+0.0im   0.0+0.0im
     0.0+0.0im  4.84+0.0im   0.0+0.0im
     0.0+0.0im   0.0+0.0im  9.38+0.0im

    ```

    """
    struct TransmonDrift{
        A <: TruncatedBosonicAlgebra,
        F,                  # Float type
    } <: DriftType{A}
        ω::Vector{F}        # Must be of length n
        δ::Vector{F}        # Must be of length n
        g::Vector{F}
        quples::Vector{Quple}

        function TransmonDrift{A}(
            ω::AbstractVector{<:Real},
            δ::AbstractVector{<:Real},
            g::AbstractVector{<:Real},
            quples::AbstractVector{Quple},
        ) where {A<:TruncatedBosonicAlgebra}
            F = promote_type(eltype(ω), eltype(δ), eltype(g))
            n = Devices.nqubits(A)

            @assert n == length(ω) == length(δ)
            @assert length(g) == length(quples)

            return new{A,F}(
                convert(Array{F}, ω),
                convert(Array{F}, δ),
                convert(Array{F}, g),
                quples,
            )
        end
    end

    function Devices.qubithamiltonian(
        drift::TransmonDrift{A,F},
        ā,
        q::Int;
        result=nothing,
    ) where {A,F}
        isnothing(result) && (result = Array{Complex{F}}(undef, size(ā)[1:2]))
        a = @view(ā[:,:,1,q])

        # PREP AN IDENTITY MATRIX
        Im = @temparray(Bool, size(a), :qubithamiltonian)
        LAT.basisvectors(size(a,1); result=Im)

        result .= 0
        result .-= (drift.δ[q]/2)  .* Im        #       - δ/2    I
        result = LAT.rotate!(a', result)        #       - δ/2   a'a
        result .+= (drift.ω[q]) .* Im           # ω     - δ/2   a'a
        result = LAT.rotate!(a', result)        # ω a'a - δ/2 a'a'aa
        return result
    end

    function Devices.staticcoupling(
        drift::TransmonDrift{A,F},
        ā;
        result=nothing,
    ) where {A,F}
        isnothing(result) && (result = Array{Complex{F}}(undef, size(ā)[1:2]))
        aTa = @temparray(eltype(result), size(result), :staticcoupling)

        result .= 0
        for pq in eachindex(drift.quples)
            p, q = drift.quples[pq]

            aTa = mul!(aTa, (@view(ā[:,:,1,p]))', @view(ā[:,:,1,q]))
            result .+= drift.g[pq] .* aTa
            result .+= drift.g[pq] .* aTa'
        end
        return result
    end
end