module StaticHamiltonians
    import ...Devices
    import ..Algebras

    import ...TempArrays: array
    const LABEL = Symbol(@__MODULE__)

    import ...LinearAlgebraTools
    import ...Quples: Quple

    using LinearAlgebra: I, mul!

    import ..Algebras: AlgebraType
    import ..Algebras: TruncatedBosonicAlgebra

    """
        StaticHamiltonianType{A}

    Component of `ModularDevices` delegated the following methods:
    - `Devices.qubithamiltonian`
    - `Devices.staticcoupling`

    While `qubithamiltonian` and `staticcoupling` have optional `result` kwargs,
        subtypes of `StaticHamiltonianType` should consider it mandatory.
    The `ModularDevice` interface handles the rest.

    """
    abstract type StaticHamiltonianType{A<:AlgebraType} end

    Algebras.algebratype(::StaticHamiltonianType{A}) where {A} = A

    ######################################################################################

    struct TransmonHamiltonian{
        F,                  # Float type
        A <: TruncatedBosonicAlgebra,
    } <: StaticHamiltonianType{A}
        ω::Vector{F}        # Must be of length n
        δ::Vector{F}        # Must be of length n
        g::Vector{F}
        quples::Vector{Quple}

        function TransmonHamiltonian{A}(
            ω::AbstractVector{<:Real},
            δ::AbstractVector{<:Real},
            g::AbstractVector{<:Real},
            quples::AbstractVector{Quple},
        ) where {A<:TruncatedBosonicAlgebra}
            F = promote_type(eltype(ω), eltype(δ), eltype(g))
            n = Devices.nqubits(A)

            @assert n == length(ω) == length(δ)
            @assert length(g) == length(quples)

            return new{F,A}(
                convert(Array{F}, ω),
                convert(Array{F}, δ),
                convert(Array{F}, g),
                quples,
            )
        end
    end

    function Devices.qubithamiltonian(static::TransmonHamiltonian, ā, q::Int; result)
        a = @view(ā[:,:,1,q])
        Im = Matrix(I, size(a))     # UNAVOIDABLE ALLOCATION?

        result .= 0
        result .-= (static.δ[q]/2)  .* Im                       #       - δ/2    I
        result = LinearAlgebraTools.rotate!(a', result)         #       - δ/2   a'a
        result .+= (static.ω[q]) .* Im                          # ω     - δ/2   a'a
        result = LinearAlgebraTools.rotate!(a', result)         # ω a'a - δ/2 a'a'aa
        return result
    end

    function Devices.staticcoupling(
        static::TransmonHamiltonian,
        ā;
        result,
    )
        aTa = array(eltype(result), size(result), LABEL)

        result .= 0
        for pq in eachindex(static.quples)
            p, q = static.quples[pq]

            aTa = mul!(aTa, (@view(ā[:,:,1,p]))', @view(ā[:,:,1,q]))
            result .+= static.g[pq] .* aTa
            result .+= static.g[pq] .* aTa'
        end
        return result
    end

end