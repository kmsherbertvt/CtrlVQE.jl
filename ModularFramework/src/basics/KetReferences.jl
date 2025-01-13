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

    #= TODO: We don't really need `m` here.
        We just use `nlevels` from the device.
    Indeed, we don't WANT it,
        since it prevents a perfectly good ket from being used on devices with larger m.
    =#

    function Modular.prepare(
        reference::KetReference,
        device::Devices.DeviceType,
        basis::Bases.BasisType;
        result=nothing,
    )
        N = Devices.nstates(device)
        isnothing(result) && (result=Array{Complex{eltype(device)}}(undef, N))

        # REPRESENT ψ IN ITS NATIVE BASIS
        m = Devices.nlevels(device)
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