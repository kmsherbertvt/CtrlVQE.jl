module Prototypes
    using ..ModularFramework

    import CtrlVQE: Quple
    import CtrlVQE: Devices
    import CtrlVQE: Constrained, Constant

    """
        Prototype(::Type{LocalDevice{
            F,
            TruncatedBosonicAlgebra,
            TransmonDrift,
            DipoleDrive,
            DisjointMapper,
        }}, n::Int; kwargs...)

    A prototypical transmon device with the following decisions:
    - All anharmonicities are constant.
    - Couplings are linear.
    - Each coupling strength equals the difference in resonance frequencies
        of the coupled qubits.
    - By default, all resonance frequencies are equally spaced
        (so, coupling strengths are constant)
        but this can be controlled through kwargs.
    - Drives have constant complex amplitudes, applied on resonance.

    Default parameters are vaguely reminiscent of IBM devices circa 2021,
        although the default behavior of linearly-spaced resonance frequencies
        is not realistic and should be avoided outside of testing/benchmarking.

    NOTE: The `n` in the concrete algebra type is ignored.

    # Keyword Arguments
    - `ω0=4.82`: resonance frequency of first qubit.
    - `Δω=0.02`: the spacing in resonance frequencies between adjacent qubits.
        When passed as a float (including the default),
            resonance frequencies are linearly spaced.
        Instead, you can pass this as an explicit vector with `n-1` elements.
    - `δ0=0.30`: the constant anharmonicity.
    - `T=10.0`: the pulse duration,
        but this has no effect since the default signals are constant.

    """
    function Devices.Prototype(
        ::Type{<:LocalDevice{F,A_,H,V,P}},
        n::Int;
        ω0=2π*4.82, Δω=2π*0.02, δ0=2π*0.30, T=10.0,
    ) where {
        F,
        m, n_,
        A_ <: TruncatedBosonicAlgebra{m,n_},
        H <: TransmonDrift,
        V <: DipoleDrive,
        P <: DisjointMapper,
    }
        !(Δω isa AbstractVector) && (Δω = fill(F(Δω), n-1))

        algebra = TruncatedBosonicAlgebra{m,n}()
        A = algebratype(algebra)

        ω = fill(F(ω0), n); ω .+= [0; cumsum(Δω)]
        δ = fill(F(δ0), n)
        g = deepcopy(Δω)
        quples = [Quple(q,q+1) for q in 1:n-1]
        drift = TransmonDrift{A}(ω, δ, g, quples)

        drives = [
            DipoleDrive{A}(
                q, ω[q],
                Constant(zero(Complex{F})),
                Constrained(Constant(zero(F)), :A),
            ) for q in 1:n
        ]

        #= TODO:

        We'll want to allow a LinearMapper here.
        We can always construct a LinearMapper equivalent to the disjoint...
        But let's get this thing to compile first, eh?
        I don't think it's going to let me abuse type parameters quite so cavalierly.

        =#

        return LocalDevice(F, algebra, drift, drives, DISJOINT)
    end
end