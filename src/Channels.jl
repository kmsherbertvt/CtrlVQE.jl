import ..Signals

"""
NOTE: Implements `Parameter` interface.
"""
abstract type AbstractChannel end


##########################################################################################
#=                                      CHANNEL TYPES
=#


#= FIXED CHANNEL =#

struct FixedChannel{S<:AbstractSignal,F<:AbstractFloat} <: AbstractChannel
    q::Int
    Ω::S
    ν::F
end

Parameter.count(channel::FixedChannel) = Parameter.count(channel.Ω)
Parameter.names(channel::FixedChannel) = ["$Ω:$p" for p in Parameter.names(channel.Ω)]
Parameter.types(channel::FixedChannel) = Parameter.types(channel.Ω)
Parameter.bind(channel::FixedChannel, x̄::AbstractVector) = Parameter.bind(channel.Ω, x̄)

#= TUNABLE CHANNEL =#

mutable struct TunableChannel{S<:AbstractSignal,F<:AbstractFloat} <: AbstractChannel
    q::Int
    Ω::S
    ν::F
end

Parameter.count(channel::TunableChannel) = Parameter.count(channel.Ω) + 1

function Parameter.names(channel::TunableChannel)
    names = ["$Ω:$name" for name in Parameter.names(channel.Ω)]
    push!(names, "ν")
    return names
end

function Parameter.types(channel::TunableChannel{F,S}) where {F,S}
    types = Parameter.types(channel.Ω)
    push!(types, F)
    return types
end

function Parameter.bind(channel::TunableChannel, x̄::AbstractVector)
    Parameter.bind(channel.Ω, x̄[begin:end-1])
    channel.ν = x̄[end]
end