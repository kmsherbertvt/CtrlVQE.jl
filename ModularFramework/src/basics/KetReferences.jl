module KetReferences
    import ..ModularFramework as Modular
    import ..ModularFramework: ReferenceType

    import CtrlVQE.LinearAlgebraTools as LAT
    import CtrlVQE: Bases
    import CtrlVQE: Devices

    """
        KetReference(ket, basis)

    Represents a basis state of the given basis.

    Note the result of `initialize` will only be a basis state
        if the ket basis and the initialization basis match.

    Index 1 of ket is the *most significant* bit when determining *which* basis state.

    """
    struct KetReference{B<:Bases.BasisType} <: ReferenceType
        ket::Vector{Int}
        basis::B
    end

    function Modular.prepare(
        reference::KetReference,
        device::Devices.DeviceType,
        basis::Bases.BasisType;
        result=nothing,
    )
        m = Devices.nlevels(device)
        n = Devices.nstates(device)
        N = Devices.nstates(device)
        @assert n == length(reference.ket)
        @assert m ≥ maximum(reference.ket)
        isnothing(result) && (result=Array{Complex{eltype(device)}}(undef, N))

        # REPRESENT ψ IN ITS NATIVE BASIS
        i = 0
        for q in eachindex(reference.ket)
            i *= m
            i += reference.ket[q]
        end
        result .= 0; result[1+i] = 1

        # ROTATE INTO THE REQUESTED BASIS
        U = Devices.basisrotation(basis, reference.basis, device)
        LAT.rotate!(U, result)

        return result
    end
end