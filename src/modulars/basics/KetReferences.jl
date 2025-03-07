module KetReferences
    import ..ModularFramework as Modular
    import ..ModularFramework: ReferenceType

    import CtrlVQE.LinearAlgebraTools as LAT
    import CtrlVQE: Bases
    import CtrlVQE: Devices

    """
        KetReference(ket, basis)

    Represents a basis state of the given basis.

    # Parameters
    - `ket`: a vector of integers, representing the ket.
        For example, `[0,1,1]` represents state `|011⟩`, the fourth basis state.
        Note that index 1 of the ket is the *most significant* bit.
    - `basis`: the `BasisType` identifying the basis for which this state is a ket state.

    ```jldoctests
    julia> using CtrlVQE.ModularFramework;

    julia> reference = KetReference([0,1], Bases.BARE);

    julia> device = Devices.Prototype(LocalDevice{Float64}, 2);

    julia> validate(reference; device=device);

    julia> prepare(reference, device, Bases.BARE)
    4-element Vector{ComplexF64}:
     0.0 + 0.0im
     1.0 + 0.0im
     0.0 + 0.0im
     0.0 + 0.0im

    ```

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
        n = Devices.nqubits(device)
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