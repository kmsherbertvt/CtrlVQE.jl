module CtrlVQE

module Parameter
    function count(::Any)::Int end
    function names(::Any)::AbstractVector{String} end
    function types(::Any)::AbstractVector{DataType} end
    function bind(::Any, x̄::AbstractVector)::Nothing end
end

module Bases
    abstract type BasisType end
    struct Dressed <: BasisType end
    abstract type LocalBasis <: BasisType end
    struct Occupation <: LocalBasis end
    struct Coordinate <: LocalBasis end
    struct Momentum   <: LocalBasis end
end

module Operators
    abstract type OperatorType end
    abstract type StaticOperator end
    struct Qubit        <: StaticOperator end
    struct Coupling     <: StaticOperator end
    struct Channel      <: OperatorType end
    struct Gradient     <: OperatorType end
    struct Uncoupled    <: StaticOperator end
    struct Static       <: StaticOperator end
    struct Drive        <: OperatorType end
    struct Hamiltonian  <: OperatorType end
end

module LinearAlgebraTools
    include("LinearAlgebraTools.jl")
end

module Signals
    include("Signals.jl")
end

module Devices
    include("Devices.jl")
end

module Evolutions
    include("Evolutions.jl")
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

end # module CtrlVQE
