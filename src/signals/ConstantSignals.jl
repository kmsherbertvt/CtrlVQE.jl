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


mutable struct PolarComplexConstant{F} <: Signals.ParametricSignal{F,Complex{F}}
    r::F    # MODULUS
    ϕ::F    # PHASE
end

(signal::PolarComplexConstant)(t::Real) = signal.r * cis(signal.ϕ)
function Signals.partial(i::Int, signal::PolarComplexConstant{F}, t::Real) where {F}
    return i == 1 ? cis(signal.ϕ) : (im * signal(t))
end

function Base.string(::PolarComplexConstant, names::AbstractVector{String})
    r, ϕ = names
    return "$r exp(i $ϕ)"
end