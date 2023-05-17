module CtrlVQE

module TempArrays
    using Memoization: @memoize

    @memoize function array(::F, shape::Tuple, index=nothing) where {F<:Number}
        # NOTE: `index` gives a means of making distinct arrays of the same type and shape
        #= NOTE: Standard practice is to pass a Symbol(<modulename>) as index,
            to ensure no unwanted collisions. =#
        # TODO (lo): Thread-safe and hands-off approach to this.
        return Array{F}(undef, shape)
    end

    function array(F::Type{<:Number}, shape::Tuple, index=nothing)
        return array(zero(F), shape, index)
    end
end

module Parameters
    function count(::Any)::Int; error("Not Implemented"); end
    function names(::Any)::Vector{String}; error("Not Implemented"); end
    function values(::Any)::AbstractVector{<:Number}; error("Not Implemented"); end
    function bind(::Any, x̄::AbstractVector)::Nothing; error("Not Implemented"); end
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

    struct Identity <: StaticOperator end
    const IDENTITY = Identity()

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
import .Operators: IDENTITY, COUPLING, UNCOUPLED, STATIC

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
import .Devices: nqubits, nstates, nlevels, ndrives, ngrades

module Evolutions
    include("Evolutions.jl")
end
import .Evolutions: trapezoidaltimegrid, evolve, evolve!, gradientsignals, Rotate

module QubitOperators
    include("QubitOperators.jl")
end

module CostFunctions
    include("CostFunctions.jl")

    #= ENERGY FUNCTIONS =#
    module EnergyFunctions
        import ..CostFunctions: AbstractCostFunction
        abstract type AbstractEnergyFunction <: AbstractCostFunction end
        function evaluate(::AbstractEnergyFunction, ψ::AbstractVector)
            return error("Not Implemented")
        end
        function evaluate(::AbstractEnergyFunction, ψ::AbstractVector, t::Real)
            return error("Not Implemented")
        end

        module BareEnergy; include("costfns/BareEnergy.jl"); end
        module ProjectedEnergy; include("costfns/ProjectedEnergy.jl"); end
        module Normalization; include("costfns/Normalization.jl"); end
        module NormalizedEnergy; include("costfns/NormalizedEnergy.jl"); end
    end
    import .EnergyFunctions: evaluate
    import .EnergyFunctions: BareEnergy, ProjectedEnergy, Normalization, NormalizedEnergy

    #= PENALTY FUNCTIONS =#
    module SoftBounds; include("costfns/SoftBounds.jl"); end
    module HardBounds; include("costfns/HardBounds.jl"); end
    module SmoothBounds; include("costfns/SmoothBounds.jl"); end

    #= TODO (mid): Local bounds using smoothing function exp(-x⁻¹) =#
    #= TODO (mid): Global RMS penalty on selected parameters. =#
    #= TODO (mid): Global RMS penalty on diff of selected parameters. =#

    #= TODO (lo): Some way to pre-constrain x̄ BEFORE energy function
        Eg. activator function, see tensorflow tutorial for inspiration. =#

end
import .CostFunctions: CompositeCostFunction, CompositeGradientFunction
import .CostFunctions: evaluate, BareEnergy, ProjectedEnergy, Normalization
import .CostFunctions: HardBounds, SoftBounds, SmoothBounds

#= RECIPES =#

function SystematicTransmonDevice(F, m, n, pulses)
    # INTERPRET SCALAR `pulses` AS A TEMPLATE TO BE COPIED
    Ω̄ = (pulses isa Signals.AbstractSignal) ? [deepcopy(pulses) for _ in 1:n] : pulses

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
    return Devices.TransmonDevice(ω̄, δ̄, ḡ, quples, q̄, ν̄, Ω̄, m)
end

function SystematicTransmonDevice(m, n, pulses)
    return SystematicTransmonDevice(Float64, m, n, pulses)
end

function SystematicTransmonDevice(n, pulses)
    return SystematicTransmonDevice(2, n, pulses)
end

function FullyTrotterizedSignal(::Type{Complex{F}}, T, r) where {F}
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(T,r)
    starttimes = t̄ .- (τ/2)
    return Signals.Windowed(
        [Signals.ComplexConstant(zero(F), zero(F)) for t in starttimes],
        starttimes,
    )
end

function FullyTrotterizedSignal(::Type{F}, T, r) where {F<:AbstractFloat}
    τ, τ̄, t̄ = Evolutions.trapezoidaltimegrid(T,r)
    starttimes = t̄ .- (τ/2)
    return Signals.Windowed(
        [Signals.Constant(zero(F)) for t in starttimes],
        starttimes,
    )
end

FullyTrotterizedSignal(T, r) = FullyTrotterizedSignal(Float64, T, r)

function WindowedSquarePulse(::Type{Complex{F}}, T, W) where {F}
    starttimes = range(zero(T), T, W+1)[1:end-1]
    return Signals.Windowed(
        [Signals.ComplexConstant(zero(F), zero(F)) for t in starttimes],
        starttimes,
    )
end

function WindowedSquarePulse(::Type{F}, T, W) where {F<:AbstractFloat}
    starttimes = range(zero(T), T, W+1)[1:end-1]
    return Signals.Windowed(
        [Signals.Constant(zero(F)) for t in starttimes],
        starttimes,
    )
end

WindowedSquarePulse(T, W) = WindowedSquarePulse(Float64, T, W)

end # module CtrlVQE
