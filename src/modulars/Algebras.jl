module Algebras
    import ...Devices

    """
        algebratype(object)

    Fetch the algebra type backing this object,
        defining the physical Hilbert space in which quantum operations occur.

    This function is implemented by most types appearing in `ModularDevices`.

    """
    function algebratype end

    """
        AlgebraType{m,n}

    # Type Parameters
    - `m`: the number of levels in each qubit
    - `n`: the number of qubits

    These parameters serve to define the physical Hilbert space of a system,
        both locally and globally.
    For now we require all local qubits to have the same number of levels,
        though they kind in principle have different algebraic structure
        (so, you might have an algebra with both spin and boson qubits,
        by embedding the spin qubits in a larger-dimensional space, maybe...).
    This decision is based mostly on the fact that working with 4d arrays
        is much better than working with vectors of 3d arrays.
    To generalize our devices, we'd need to modify the interface somewhat,
        I think mainly in how `result` is allocated.
    Maybe just maybe we can delegate the array allocation to the AlgebraType..?
    Lol, this is too complicated to think about.

    Component of `ModularDevices` delegated the following methods:
    - `Devices.nlevels`
    - `Devices.nqubits`
    - `Devices.noperators`
    - `Devices.localalgebra`

    The first two are handled by the type parameters of `AlgebraType` itself,
        so subtypes need only implement `noperators` and `localalgebra`.

    While `Devices.localalgebra` has an optional `result` kwarg,
        subtypes of `AlgebraType` should consider it mandatory.
    The `ModularDevice` interface handles the rest.

    """
    abstract type AlgebraType{m,n} end

    Devices.nlevels(::Type{<:AlgebraType{m,n}}) where {m,n} = m
    Devices.nqubits(::Type{<:AlgebraType{m,n}}) where {m,n} = n

    algebratype(algebra::AlgebraType) = typeof(algebra)

    Devices.nlevels(::AlgebraType{m,n}) where {m,n} = m
    Devices.nqubits(::AlgebraType{m,n}) where {m,n} = n

    ######################################################################################

    struct TruncatedBosonicAlgebra{m,n} <: AlgebraType{m,n} end

    Devices.noperators(::TruncatedBosonicAlgebra) = 1

    function Devices.localalgebra(::TruncatedBosonicAlgebra{m,n}; result) where {m,n}
        result .= 0
        for q in 1:n
            for i in 1:m-1
                result[i,i+1,1,q] = √i
            end
        end
        return result
    end

    ######################################################################################

    struct PauliAlgebra{n} <: AlgebraType{2,n} end

    Devices.noperators(::PauliAlgebra) = 3

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