module DenseReferences
    import CtrlVQE.ModularFramework as Modular
    import CtrlVQE.ModularFramework: ReferenceType

    import CtrlVQE.LinearAlgebraTools as LAT
    import CtrlVQE.QubitProjections: isometrize
    import CtrlVQE: Bases
    import CtrlVQE: Devices

    """
        DenseReference(statevector, basis; m, n)

    Represents an arbitrary statevector of the given basis.

    # Parameters
    - `statevector`: a dense statevector.
    - `basis`: the `BasisType` identifying the basis `statevector` is written in.

    The kwargs `m` and `n` specify the number of levels and number of qubits
        for which the statevector is defined.
    The reference state may be prepared on a device with the same number of qubits,
        and at least as many levels.
    Typically, only one of `m` and `n` are provided,
        and the other is inferred from the length of `statevector`.
    If neither is provided, `m` defaults to 2.
    If both are provided, an error will be thrown
        if they are not consistent with the length of `statevector`.

    ```jldoctests
    julia> using CtrlVQE.ModularFramework;

    julia> reference = DenseReference([0,1,0,0], Bases.BARE);

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
    struct DenseReference{F<:Number,B<:Bases.BasisType} <: ReferenceType
        statevector::Vector{F}
        basis::B
        m::Int
        n::Int

        function DenseReference(statevector, basis, m, n)
            N = length(statevector)
            @assert N == m^n
            return new{eltype(statevector),typeof(basis)}(statevector, basis, m, n)
        end
    end

    function DenseReference(statevector, basis; m=nothing, n=nothing)
        N = length(statevector)
        isnothing(m) && isnothing(n) && (m=2)
        isnothing(m) && (m = round(Int, N^(1/n)))
        isnothing(n) && (n = round(Int, log(m,N)))
        return DenseReference(statevector, basis, m, n)
    end

    function Modular.prepare(
        reference::DenseReference,
        device::Devices.DeviceType,
        basis::Bases.BasisType;
        result=nothing,
    )
        N = Devices.nstates(device)
        isnothing(result) && (result=Array{Complex{eltype(device)}}(undef, N))

        # REPRESENT ψ IN ITS NATIVE BASIS, WITH THE CORRECT `m`
        m = Devices.nlevels(device)
        n = Devices.nqubits(device)
        @assert m >= reference.m
        @assert n == reference.n
        isometrize(reference.statevector, n, m; result=result)

        # ROTATE INTO THE REQUESTED BASIS
        U = Devices.basisrotation(basis, reference.basis, device)
        LAT.rotate!(U, result)

        return result
    end
end