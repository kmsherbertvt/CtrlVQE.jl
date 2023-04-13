module CtrlVQE

module TempArrays
    using Memoization: @memoize

    @memoize function array(::F, shape::Tuple, index=nothing) where {F<:Number}
        # NOTE: `index` gives a means of making distinct arrays of the same type and shape
        #= NOTE: Standard practice is to pass a Symbol(<modulename>) as index,
            to ensure no unwanted collisions. =#
        # TODO (lo): Thread-safe and hands-of approach to this.
        return Array{F}(undef, shape)
    end

    function array(F::Type{<:Number}, shape::Tuple, index=nothing)
        return array(zero(F), shape, index)
    end
end

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

    struct Qubit <: StaticOperator
        q::Int
    end

    struct Coupling <: StaticOperator end
    const COUPLING = Coupling()

    struct Uncoupled <: StaticOperator end
    const UNCOUPLED = Uncoupled()

    struct Static <: StaticOperator end
    const STATIC = Static()

    struct Channel{R<:Real} <: OperatorType
        i::Int
        t::R
    end

    struct Drive{R<:Real} <: OperatorType
        t::R
    end

    struct Hamiltonian{R<:Real} <: OperatorType
        t::R
    end

    struct Gradient{R<:Real} <: OperatorType
        j::Int
        t::R
    end
end

module LinearAlgebraTools
    include("LinearAlgebraTools.jl")
end

module Signals
    include("Signals.jl")

    module ConstantSignals; include("signals/ConstantSignals.jl"); end
    import .ConstantSignals: Constant, ComplexConstant

    module IntervalSignals; include("signals/IntervalSignals.jl"); end
    import .IntervalSignals: Interval, ComplexInterval

    module StepFunctionSignals; include("signals/StepFunctionSignals.jl"); end
    import .StepFunctionSignals: StepFunction

end

module Devices
    include("Devices.jl")

    module TransmonDevices; include("devices/TransmonDevices.jl"); end
    import .TransmonDevices: TransmonDevice
end

module Evolutions
    include("Evolutions.jl")
end

module QubitOperators
    include("QubitOperators.jl")
end

module CostFunctions
    include("CostFunctions.jl")

    #= ENERGY FUNCTIONS =#
    module BareEnergy; include("costfns/BareEnergy.jl"); end
    module ProjectedEnergy; include("costfns/ProjectedEnergy.jl"); end
    module NormalizedEnergy; include("costfns/NormalizedEnergy.jl"); end

    #= PENALTY FUNCTIONS =#
    module SoftBounds; include("costfns/SoftBounds.jl"); end
    module HardBounds; include("costfns/HardBounds.jl"); end

    #= TODO (mid): Global RMS penalty on selected parameters. =#
    #= TODO (mid): Global RMS penalty on diff of selected parameters. =#

    #= TODO (lo): Some way to pre-constrain x̄ BEFORE energy function
        Eg. activator function, see tensorflow tutorial for inspiration. =#

end


#= RECIPES =#

function SystematicTransmonDevice(F, m, n, pulsetemplate)
    ω0 = F(2π * 4.80)
    Δω = F(2π * 0.02)
    δ0 = F(2π * 0.30)
    g0 = F(2π * 0.02)

    ω̄ = collect(ω0 .+ (Δω * (1:n)))
    δ̄ = fill(δ0, n)
    ḡ = fill(g0, n-1)
    quples = [Devices.Quple(q,q+1) for q in 1:n-1]
    q̄ = 1:n
    ν̄ = copy(ω̄)
    Ω̄ = [deepcopy(pulsetemplate) for _ in 1:n]
    return Devices.TransmonDevice(ω̄, δ̄, ḡ, quples, q̄, ν̄, Ω̄, m)
end

function SystematicTransmonDevice(m, n, pulsetemplate)
    return SystematicTransmonDevice(Float64, m, n, pulsetemplate)
end

function SystematicTransmonDevice(n, pulsetemplate)
    return SystematicTransmonDevice(2, n, pulsetemplate)
end

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
