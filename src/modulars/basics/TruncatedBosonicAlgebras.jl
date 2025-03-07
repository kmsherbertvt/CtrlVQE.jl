module TruncatedBosonicAlgebras
    import ..ModularFramework: AlgebraType

    import CtrlVQE: Devices

    """
        TruncatedBosonicAlgebra{m,n}()

    An algebra of `n` distinguishable bosonic modes, each represented with `m` levels.

    This algebra is useful for transmon architectures.

    ```jldoctests
    julia> using CtrlVQE.ModularFramework;

    julia> A = TruncatedBosonicAlgebra{3,2};

    julia> validate(A());

    julia> nlevels(A)
    3
    julia> nqubits(A)
    2
    julia> noperators(A)
    1

    ```

    """
    struct TruncatedBosonicAlgebra{m,n} <: AlgebraType{m,n} end

    Devices.noperators(::Type{<:TruncatedBosonicAlgebra}) = 1

    function Devices.localalgebra(
        ::TruncatedBosonicAlgebra{m,n};
        result=nothing,
    ) where {m,n}
        isnothing(result) && (result = Array{Float64}(undef, m, m, 1, n))
        result .= 0
        for q in 1:n
            for i in 1:m-1
                result[i,i+1,1,q] = √i
            end
        end
        return result
    end
end