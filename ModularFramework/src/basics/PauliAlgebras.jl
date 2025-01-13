module PauliAlgebras
    import ..ModularFramework: AlgebraType

    import CtrlVQE: Devices

    struct PauliAlgebra{n} <: AlgebraType{2,n} end

    Devices.noperators(::Type{<:PauliAlgebra}) = 3

    function Devices.localalgebra(::PauliAlgebra{n}; result) where {n}
        result .= 0
        for q in 1:n
            # X
            result[1,2,1,q] = 1
            result[2,1,1,q] = 1
            # Y
            result[1,2,2,q] = -im
            result[2,1,2,q] = im
            # Z
            result[1,1,3,q] = 1
            result[2,2,3,q] = -1
        end
        return result
    end
end