module Algebras
    import ..Devices

    """
        AlgebraType

    Component of `ModularDevices` delegated the following methods:
    - `Devices.nlevels`
    - `Devices.noperators`
    - `Devices.localalgebra`

    While `Devices.localalgebra` normally accepts a `result` kwarg and returns a 4darray,
        subtypes of `AlgebraType` should return a new 3darray,
        where [:,:,σ] is the matrix for the σ'th algebraic operator,
        acting on an arbitrary qubit.
    The `ModularDevice` interface handles the rest.

    """
    abstract type AlgebraType end

    ######################################################################################

    struct TruncatedBosonicAlgebra{m,F} <: AlgebraType end

    Devices.nlevels(::TruncatedBosonicAlgebra{m,F}) where {m,F} = m
    Devices.noperators(::TruncatedBosonicAlgebra) = 1
    function Devices.localalgebra(::TruncatedBosonicAlgebra{m,F}) where {m,F}
        āq = zeros(F, (m, m, 1))
        for i in 1:m-1
            āq[i,i+1,1] = √i
        end
        return āq
    end

    ######################################################################################

    struct PauliAlgebra <: AlgebraType end

    Devices.nlevels(::PauliAlgebra) = 2
    Devices.noperators(::PauliAlgebra) = 1
    function Devices.localalgebra(algebra::PauliAlgebra)
        isnothing(result) && (result = Array{Complex{Bool}}(undef, 2, 2, 3))
        āq = zeros(F, (2, 2, 3))
        # X
        āq[1,2,1] = 1
        āq[2,1,1] = 1
        # Y
        āq[1,2,2] = -im
        āq[2,1,2] = im
        # Z
        āq[1,1,3] = 1
        āq[2,2,3] = -1
        return āq
    end

end