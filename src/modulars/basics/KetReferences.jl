module KetReferences
    import CtrlVQE.ModularFramework as Modular
    import CtrlVQE.ModularFramework: ReferenceType

    import CtrlVQE.LinearAlgebraTools as LAT
    import CtrlVQE: Devices

    """
        KetReference(basis::B, ket)

    Represents a basis state of the given basis `B`.

    # Parameters
    - `basis::Bases.BasisType`: the basis this ket is defined in.
    - `ket`: a vector of integers representing the ket.
        For example, `[0,1,1]` represents state `|011⟩`, the fourth basis state.
        Note that index 1 of the ket is the *most significant* bit.

    ```jldoctests
    julia> using CtrlVQE.ModularFramework;

    julia> reference = KetReference(BARE, [0,1]);

    julia> device = Prototype(LocalDevice{Float64}; n=2);

    julia> validate(reference; device=device);

    julia> prepare(reference, device)
    4-element Vector{ComplexF64}:
     0.0 + 0.0im
     1.0 + 0.0im
     0.0 + 0.0im
     0.0 + 0.0im

    ```

    """
    struct KetReference{B} <: ReferenceType{B}
        ket::Vector{Int}
    end

    function KetReference(::B, ket::AbstractVector{Int}) where {B}
        return KetReference{B}(collect(ket))
    end

    function Modular.prepare(
        reference::KetReference,
        device::Devices.DeviceType;
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

        return result
    end
end