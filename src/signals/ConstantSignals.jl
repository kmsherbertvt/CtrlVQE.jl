import ...Signals

mutable struct Constant{F} <: Signals.ParametricSignal{F,F}
    A::F    # CONSTANT VALUE
end

(signal::Constant)(t::Real) = signal.A
Signals.partial(i::Int, ::Constant{F}, t::Real) where {F} = one(F)

Base.string(::Constant, names::AbstractVector{String}) = names[1]





mutable struct ComplexConstant{F} <: Signals.ParametricSignal{F,Complex{F}}
    A::F    # REAL PART
    B::F    # IMAGINARY PART
end

(signal::ComplexConstant)(t::Real) = Complex(signal.A, signal.B)
function Signals.partial(i::Int, ::ComplexConstant{F}, t::Real) where {F}
    return i == 1 ? Complex(one(F),0) : Complex(0,one(F))
end

function Base.string(::ComplexConstant, names::AbstractVector{String})
    A, B = names
    return "$A + i $B"
end
