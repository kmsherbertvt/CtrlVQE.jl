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

    module TransmonDevices
        include("devices/TransmonDevice.jl")
    end
    import .TransmonDevices: TransmonDevice
end

module Evolutions
    include("Evolutions.jl")
end


#= RECIPES =#

function SystematicTransmonDevice(m, n, pulsetemplate)
    ω̄ = 2π * collect(4.8 .+ (.02 * (1:n)))
    δ̄ = fill(2π * 0.30, n)
    ḡ = fill(2π * 0.02, n-1)
    quples = [Devices.Quple(q,q+1) for q in 1:n-1]
    q̄ = 1:n
    ν̄ = copy(ω̄)
    Ω̄ = [deepcopy(pulsetemplate) for _ in 1:n]
    return Devices.TransmonDevice(ω̄, δ̄, ḡ, quples, q̄, ν̄, Ω̄, m)
end
SystematicTransmonDevice(n, pulsetemplate) = SystematicTransmonDevice(2, n, pulsetemplate)

function FullyTrotterizedSignal(F, T, r)
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(T,r)
    return Signals.Composite([
        Signals.Constrained(
            Signals.Interval(zero(F), t-τ/2, t+τ/2),
            :s1, :s2,
        ) for t in t̄
    ]...)
end
FullyTrotterizedSignal(T, r) = FullyTrotterizedSignal(Float64, T, r)

function WindowedSquarePulse(F, T, W)
    t̄ = collect(range(0, T, W+1))
    t̄[end] += 10*eps(eltype(T))     # CLOSE THE RIGHT BOUNDARY
    return Signals.Composite([
        Signals.Constrained(
            Signals.Interval(zero(F), t̄[i], t̄[i+1]),
            :s1, :s2,
        ) for i in 1:W
    ]...)
end
WindowedSquarePulse(T, W) = WindowedSquarePulse(Float64, T, W)

end # module CtrlVQE
