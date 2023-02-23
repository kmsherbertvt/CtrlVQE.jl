module CtrlVQE

module Parameter
    function count(::Any)::Int end
    function names(::Any)::AbstractVector{String} end
    function types(::Any)::AbstractVector{DataType} end
    function bind(::Any, x̄::AbstractVector)::Nothing end
end

module Basis
    abstract type AbstractBasis end
    struct Dressed <: Basis end
    abstract type LocalBasis <: Basis end
    struct Occupation <: LocalBasis end
    struct Coordinate <: LocalBasis end
    struct Momentum   <: LocalBasis end
end

module Locality
    abstract type AbstractLocality end
    struct Local <: AbstractLocality end
    struct Mixed <: AbstractLocality end
end

module Temporality
    abstract type AbstractTemporality end
    struct Static <: AbstractTemporality end
    struct Driven <: AbstractTemporality end
end

module LinearAlgebraTools
    include("LinearAlgebraTools.jl")
end

module Signals
    include("Signals.jl")
end

module Channels
    include("Channels.jl")
end

module Devices
    include("Devices.jl")
end



#= SPECIFIC DEVICES =#

module TransmonDevices
    include("devices/TransmonDevice.jl")
end

#= TODO: Convenience constructor for transmon device with n qubits all at m levels.
            Use constant δ, g, and increment ω at reasonable fixed spacing.
=#

#= TODO: Convenience constructor for a channel with square pulses.
=#

end
