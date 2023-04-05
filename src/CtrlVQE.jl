module CtrlVQE

module Parameters
    function count(::Any)::Int end
    function names(::Any)::AbstractVector{String} end
    function values(::Any)::AbstractVector end
    function bind(::Any, x̄::AbstractVector)::Nothing end
end

module Bases
    abstract type BasisType end
    struct Dressed      <: BasisType end;       const DRESSED = Dressed()
    abstract type LocalBasis <: BasisType end
    struct Occupation   <: LocalBasis end;      const OCCUPATION = Occupation()
    struct Coordinate   <: LocalBasis end;      const COORDINATE = Coordinate()
    struct Momentum     <: LocalBasis end;      const MOMENTUM   = Momentum()
end

module Operators
    abstract type OperatorType end
    abstract type StaticOperator <: OperatorType end
    struct Qubit        <: StaticOperator end;  const QUBIT       = Qubit()
    struct Coupling     <: StaticOperator end;  const COUPLING    = Coupling()
    struct Channel      <: OperatorType end;    const CHANNEL     = Channel()
    struct Gradient     <: OperatorType end;    const GRADIENT    = Gradient()
    struct Uncoupled    <: StaticOperator end;  const UNCOUPLED   = Uncoupled()
    struct Static       <: StaticOperator end;  const STATIC      = Static()
    struct Drive        <: OperatorType end;    const DRIVE       = Drive()
    struct Hamiltonian  <: OperatorType end;    const HAMILTONIAN = Hamiltonian()
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
