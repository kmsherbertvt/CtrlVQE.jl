module ConstantSignals
    export ConstantSignal, ComplexConstantSignal, Constant

    import ..CtrlVQE: Signals

    """
        ConstantSignal(A::AbstractFloat)

    The constant real signal ``Ω(t) = A``.

    """
    mutable struct ConstantSignal{F} <: Signals.ParametricSignal{F,F}
        A::F    # CONSTANT VALUE
    end

    Signals.parameters(::ConstantSignal) = (:A,)
    Signals.valueat(signal::ConstantSignal, t::Real) = signal.A
    Signals.partial(i::Int, ::ConstantSignal{F}, t::Real) where {F} = one(F)

    Base.string(::ConstantSignal, names::AbstractVector{String}) = only(names)

    """
        ComplexConstantSignal(A::F, B::F) where {F<:AbstractFloat}

    The constant complex signal ``Ω(t) = A + iB``.

    """
    mutable struct ComplexConstantSignal{F} <: Signals.ParametricSignal{F,Complex{F}}
        A::F    # REAL PART
        B::F    # IMAGINARY PART
    end

    Signals.parameters(::ComplexConstantSignal) = (:A,:B)
    Signals.valueat(signal::ComplexConstantSignal, t::Real) = Complex(signal.A, signal.B)
    function Signals.partial(i::Int, ::ComplexConstantSignal{F}, t::Real) where {F}
        return i == 1 ? Complex(one(F),0) : Complex(0,one(F))
    end

    function Base.string(::ComplexConstantSignal, names::AbstractVector{String})
        A, B = names
        return "$A + i $B"
    end

    """
        Constant(A)
        Constant(A, B)

    Convenience constructors for a constant signal.

    The single argument form constructs a `ConstantSignal` when given a real number,
        or a `ComplexConstantSignal` when given a complex number.
    The two-argument form constructs a `ComplexConstantSignal`,
        taking `A` as the real part and `B` as the imaginary part.

    ```jldoctests
    julia> grid = TemporalLattice(20.0, 400);

    julia> realonly = Constant(2.0);

    julia> validate(realonly; grid=grid, t=10.0, rms=1e-6);

    julia> valueat(realonly, 10.0)
    2.0
    julia> Parameters.names(realonly)
    1-element Vector{String}:
     "A"

    julia> complex = Constant(2.0, 1.0);

    julia> validate(complex; grid=grid, t=10.0, rms=1e-6);

    julia> valueat(complex, 10.0)
    2.0 + 1.0im
    julia> Parameters.names(complex)
    2-element Vector{String}:
     "A"
     "B"

    julia> alsocomplex = Constant(2.0 + 1.0im);

    julia> validate(alsocomplex; grid=grid, t=10.0, rms=1e-6);

    julia> typeof(alsocomplex) == typeof(complex)
    true
    julia> valueat(alsocomplex, 10.0) == valueat(complex, 10.0)
    true
    julia> Parameters.names(alsocomplex) == Parameters.names(complex)
    true

    julia> imagfrozen = Constrained(complex, :B);

    julia> validate(imagfrozen; grid=grid, t=10.0, rms=1e-6);

    julia> valueat(imagfrozen, 10.0) == valueat(complex, 10.0)
    true
    julia> Parameters.names(imagfrozen) == Parameters.names(realonly)
    true
    ```

    """
    function Constant end
    Constant(A::Real) = ConstantSignal(A)
    Constant(C::Complex) = ComplexConstantSignal(real(C), imag(C))
    Constant(A,B) = ComplexConstantSignal(A,B)



    #= TODO: I removed PolarComplexConstant from Basics.
    It certainly deserves to be in SignalsKit. =#
end