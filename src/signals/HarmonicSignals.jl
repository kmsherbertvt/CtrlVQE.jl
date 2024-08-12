module HarmonicSignals
    import ..Signals
    export ComplexHarmonic

    import ..ParametricSignals: ParametricSignal, parameters

    mutable struct ComplexHarmonic{F} <: ParametricSignal{F,Complex{F}}
        A::F    # AMPLITUDE - REAL PART
        B::F    # AMPLITUDE - IMAG PART
        n::Int  # HARMONIC NUMBER
        T::F    # FUNDAMENTAL PERIOD (aka T such that ν=n/2T)
    end

    # """ Hard-code the fact that T is not meant to be a variational parameter.

    # (Alternatively, we could use Constrained(my_harmonic, :T),
    #     but this seems simpler.)
    # """
    # function ParametricSignals.parameters(::ComplexHarmonic)
    #     return [:A, :B]
    # end

    # TODO: Hard-coding didn't work. Why not?!

    function Signals.valueat(signal::ComplexHarmonic{F}, t::Real) where {F}
        A = signal.A; B = signal.B; n = signal.n; T = signal.T
        ν = F(2π * n / 2T)
        return Complex(A, B) * sin(ν*t)
    end

    function Signals.partial(i::Int, signal::ComplexHarmonic{F}, t::Real) where {F}
        field = parameters(signal)[i]
        A = signal.A; B = signal.B; n = signal.n; T = signal.T
        ν = F(2π * n / 2T)
        sine = sin(ν*t)
        return (field == :A ?   Complex(sine, zero(F))
            :   field == :B ?   Complex(zero(F), sine)
            :                   error("Not Implemented"))
    end

    function Base.string(signal::ComplexHarmonic, names::AbstractVector{String})
        A, B, T = names
        n = signal.n
        return "($A+i$B) sin($n⋅π⋅t/$T)"
    end
end





#= TODO: Didn't we learn that a native CompositeHarmonic is much better than a Composite{Harmonic}? Maybe it doesn't matter now that we have the mapping layer, but I suspect without thinking very hard that you should write the CompositeComplexHarmonic type, based loosely on CtrlJobs' ModalHarmonic type. =#