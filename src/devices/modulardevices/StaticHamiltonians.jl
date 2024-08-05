module StaticHamiltonians
    import ..Devices
    import ..Algebras: AlgebraType
    import ..Algebras: TruncatedBosonicAlgebra

    import ...TempArrays: array
    const LABEL = Symbol(@__MODULE__)

    import ...LinearAlgebraTools
    import ...Quples: Quple

    using LinearAlgebra: I, mul!

    """
        StaticHamiltonianType{A}

    Component of `ModularDevices` delegated the following methods:
    - `Devices.nqubits`
    - `Devices.qubithamiltonian`
    - `Devices.staticcoupling`

    While `qubithamiltonian` and `staticcoupling` have optional `result` kwargs,
        subtypes of `StaticHamiltonianType` should consider it mandatory.
    The `ModularDevice` interface handles the rest.

    """
    abstract type StaticHamiltonianType{A<:AlgebraType} end

    ######################################################################################

    struct TransmonHamiltonian{m,F} <: StaticHamiltonianType{TruncatedBosonicAlgebra{m,F}}
        ω::Vector{F}
        δ::Vector{F}
        g::Vector{F}
        quples::Vector{Quple}

        # TODO: Inner constructor to accept abstract vectors and proof length-consistency.
    end

    Devices.nqubits(static::TransmonHamiltonian) = length(static.ω)
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
        static::TransmonHamiltonian{m,F},
        ā;
        result,
    ) where {m,F}
        aTa = array(F, size(result), LABEL)

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