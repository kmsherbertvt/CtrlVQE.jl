import ...Signals

mutable struct Interval{F} <: Signals.ParametricSignal{F,F}
    A::F
    s1::F
    s2::F
end

function (signal::Interval{F})(t::Real) where {F}
    return signal.s1 ≤ t < signal.s2 ? signal.A : zero(F)
end

function Signals.partial(i::Int, signal::Interval, t::Real)
    # field = Signals.parameters(signal)[i]
    # return field == :A ? partial_A(signal, t) : error("Not Implemented")
    return i == 1 ? partial_A(signal, t) : error("Not Implemented")
end

function partial_A(signal::Interval{F}, t::Real) where {F}
    return signal.s1 ≤ t < signal.s2 ? one(F) : zero(F)
end

function Base.string(::Interval, names::AbstractVector{String})
    A, s1, s2 = names
    return "$A | t∊[$s1,$s2)"
end






mutable struct ComplexInterval{F} <: Signals.ParametricSignal{F,Complex{F}}
    A::F
    B::F
    s1::F
    s2::F
end

function (signal::ComplexInterval{F})(t::Real) where {F}
    return signal.s1 ≤ t < signal.s2 ? Complex(signal.A, signal.B) : zero(Complex{F})
end

function Signals.partial(i::Int, signal::ComplexInterval{F}, t::Real) where {F}
    # field = Signals.parameters(signal)[i]
    # return (field == :A ?   partial_A(signal, t)
    #     :   field == :B ?   partial_B(signal, t)
    #     :                   error("Not Implemented"))
    return (i == 1 ?    partial_A(signal, t)
        :   i == 2 ?    partial_B(signal, t)
        :               error("Not Implemented"))
end

function partial_A(signal::ComplexInterval{F}, t::Real) where {F}
    return Complex(signal.s1 ≤ t < signal.s2 ? one(F) : zero(F), zero(F))
end

function partial_B(signal::ComplexInterval{F}, t::Real) where {F}
    return Complex(zero(F), signal.s1 ≤ t < signal.s2 ? one(F) : zero(F))
end

function Base.string(::ComplexInterval, names::AbstractVector{String})
    A, s1, s2 = names
    return "$A + i $B | t∊[$s1,$s2)"
end

