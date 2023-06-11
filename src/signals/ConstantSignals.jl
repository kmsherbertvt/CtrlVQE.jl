# TODO (hi): add exports
import ...Signals

"""
    Constant(A::F) where {F<:AbstractFloat}

The constant real signal ``Ω(t) = A``.

"""
mutable struct Constant{F} <: Signals.ParametricSignal{F,F}
    A::F    # CONSTANT VALUE
end

(signal::Constant)(t::Real) = signal.A
Signals.partial(i::Int, ::Constant{F}, t::Real) where {F} = one(F)

Base.string(::Constant, names::AbstractVector{String}) = names[1]





"""
    ComplexConstant(A::F, B::F) where {F<:AbstractFloat}

The constant complex signal ``Ω(t) = A + iB``.

"""
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





"""
    PolarComplexConstant(r::F, ϕ::F) where {F<:AbstractFloat}

The constant complex signal ``Ω(t) = r \\exp(iϕ)``.

"""
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