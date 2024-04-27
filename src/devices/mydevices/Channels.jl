abstract type ChannelType{F,A<:AlgebraType} end
abstract type LocalChannel{F,A} <: ChannelType{F,A} end
    # TODO: For `driveoperator` and `gradeoperator`, sans index
    # TODO: Implements Parameters interface.

#= TODO:

QubitChannel:   Ω exp(iνt) a + h.t.     Ω is a complex signal, ν is a real signal
RealChannel:    Ω [exp(iνt) a + h.t]    Ω and ν are real signals
PolarChannel:   Ω exp[i(νt+ϕ)] + h.t.   Ω, ϕ, ν are real signals
NoRWAChannel:   Ω sin(νt+ϕ) (a + a')    Ω, ϕ, ν are real signals

=#