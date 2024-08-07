module PreparationProtocols
    import ..Algebras

    import ...Bases, ...Devices

    import ...LinearAlgebraTools

    abstract type PreparationProtocolType{A<:Algebras.AlgebraType} end

    Algebras.algebratype(::PreparationProtocolType{A}) where {A} = A

    """
        initialstate(preparer, device, basis; result=nothing)

    Prepare the state and represent it as a statevector ψ in the given basis.

    The array is stored in `result` if provided.
    If `result` is not provided, the array is of type `Complex{eltype(device)}`.

    """
    function initialstate end


    """
        KetPreparation(algebra, ket, basis)

    Represents a basis state of the given basis.

    Note the result of `initialize` will only be a basis state
        if the ket basis and the initialization basis match.

    Index 1 of ket is the *most significant* bit when determining *which* basis state.
    """
    struct KetPreparation{A} <: PreparationProtocolType{A}
        ket::Vector{Int}
        basis::Bases.BasisType

        function KetPreparation{A}(
            ket::AbstractVector{<:Integer},
            basis::Bases.BasisType,
        ) where {A<:Algebras.AlgebraType}
            m = Devices.nlevels(A)
            @assert all(0 .≤ ket .< m)
            return new{A}(ket, basis)
        end
    end

    function initialstate(
        preparer::KetPreparation,
        device::Devices.DeviceType,
        basis::Bases.BasisType;
        result=nothing
    )
        N = Devices.nstates(device)
        isnothing(result) && (result = Array{Complex{eltype(device)}}(undef, N))

        # REPRESENT ψ IN ITS NATIVE BASIS
        i = 0
        m = Devices.nlevels(device)
        for q in eachindex(preparer.ket)
            i *= m
            i += preparer.ket[q]
        end
        result .= 0; result[1+i] = 1

        # ROTATE INTO THE REQUESTED BASIS
        U = Devices.basisrotation(basis, preparer.basis, device)
        LinearAlgebraTools.rotate!(U, result)

        return result
    end



end