module TruncatedBosonicAlgebras
    import ..ModularFramework: AlgebraType

    import CtrlVQE: Devices

    """
    TODO
    """
    struct TruncatedBosonicAlgebra{m,n} <: AlgebraType{m,n} end

    Devices.noperators(::Type{<:TruncatedBosonicAlgebra}) = 1

    function Devices.localalgebra(::TruncatedBosonicAlgebra{m,n}; result) where {m,n}
        result .= 0
        for q in 1:n
            for i in 1:m-1
                result[i,i+1,1,q] = √i
            end
        end
        return result
    end
end