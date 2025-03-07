module Prototypes
    using CtrlVQE.ModularFramework

    import CtrlVQE
    import CtrlVQE: Quple
    import CtrlVQE: Devices
    import CtrlVQE: Constrained, Constant

    """
        Prototype(TransmonDrift; kwargs...)

    Construct a prototypical, linearly-coupled transmon static Hamiltonian.

    # Keyword Arguments
    - `n::Int`: the number of qubits
    - `A`: the algebra type
    - `F`: the float type
    - `Δω`: a vector of resonance spacings between adjacent qubits,
        or a fixed spacing if provided as a scalar.
        Note that fixed spacing (which is the default)
            tends to result in degeneracies and ill-defined dressed bases.
    - `ω0`: the resonance frequency of the first qubit.
    - `δ0`: the anharmonicity (applied to all qubits).
    In this prototype, all qubits have the same anharmonicity,
        and coupling constants are equal to the resonance frequency spacing.

    """
    function CtrlVQE.Prototypes.Prototype(
        ::Type{TransmonDrift};
        n=2,
        A=TruncatedBosonicAlgebra{2,n},
        F=Float64,
        Δω=F(2π*0.02), ω0=F(2π*4.82), δ0=F(2π*0.30),
    )
        n = Devices.nqubits(A)
        Δω isa AbstractVector || (Δω = fill(Δω, n-1))

        ω = fill(ω0, n); ω .+= [0; cumsum(Δω)]
        δ = fill(δ0, n)
        g = deepcopy(Δω)
        quples = [Quple(q,q+1) for q in 1:n-1]
        return TransmonDrift{A}(ω, δ, g, quples)
    end

    """
        Prototype(DipoleDrive; kwargs...)

    Construct a prototypical dipole drive term,
        with a single free complex amplitude (two parameters) and a single fixed detuning.

    # Keyword Arguments
    - `q::Int`: the targeted qubit index
    - `A`: the algebra type
    - `F`: the float type
    - `ω`: the resonance frequency of the targeted qubit

    """
    function CtrlVQE.Prototypes.Prototype(
        ::Type{DipoleDrive};
        q=1,
        A=TruncatedBosonicAlgebra{2,n},
        F=Float64,
        ω=F(2π*4.82),
    )
        Ω = Constant(zero(Complex{F}))
        Δ = Constrained(Constant(zero(F)), :A)
        return DipoleDrive{A}(q, ω, Ω, Δ)
    end

    """
        Prototype(LocalDevice{F}; kwargs...)

    Construct a prototypical modular device.

    # Keyword Arguments
    - `n::Int`: number of qubits
    - `T::Real`: totally redundant, just required for interface
    The remaining kwargs coincide exactly with the standard constructor:
    - `F`: the number type, defaults to `Float64`.
    - `algebra::AlgebraType`: defaults to a `TruncatedBosonicAlgebra` with `m=2`.
    - `drift::DriftType`: defaults to a prototypical `TransmonDrift`.
    - `drives::Vector{DriveType}`: defaults to a prototypical `DipoleDrive` on each qubit.
    - `pmap::ParameterMap`: defaults to a `DISJOINT` mapper.

    """
    function CtrlVQE.Prototypes.Prototype(
        ::Type{D};
        n=2,
        T=10.0,
        F=Float64,
        algebra=nothing,
        drift=nothing,
        drives=nothing,
        pmap=nothing,
    ) where {D<:LocalDevice}
        isnothing(algebra) && (algebra = TruncatedBosonicAlgebra{2,n}())
        A = algebratype(algebra)
        if isnothing(drift)
            if A <: TruncatedBosonicAlgebra
                drift = CtrlVQE.Prototypes.Prototype(TransmonDrift; n=n, A=A, F=F)
            else
                error("No default drift for $A")
            end
        end
        if isnothing(drives)
            if drift isa TransmonDrift
                ω = drift.ω
                drives = [
                CtrlVQE.Prototypes.Prototype(DipoleDrive; q=q, A=A, ω=ω[q])
                    for q in 1:n
                ]
            else
                error("No default drive for $(typeof(drift))")
            end
        end
        if isnothing(pmap)
            pmap = DISJOINT
        end

        return LocalDevice(F, algebra, drift, drives, pmap)
    end
end
