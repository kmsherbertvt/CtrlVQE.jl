module TransmonDrifts
    import ..ModularFramework: DriftType
    import ..ModularFramework: TruncatedBosonicAlgebra

    import CtrlVQE: Quple
    import CtrlVQE: LAT
    import CtrlVQE: Devices

    import TemporaryArrays: @temparray

    import LinearAlgebra: mul!

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

    function Devices.qubithamiltonian(drift::TransmonDrift, ā, q::Int; result)
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
        drift::TransmonDrift,
        ā;
        result,
    )
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