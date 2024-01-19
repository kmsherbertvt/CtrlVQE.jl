module CtrlVQE

"""
    Parameters

Standardized interface for interacting with variational parameters.

Most commonly implemented by `Signals` and `Devices`.

"""
module Parameters
    """
        count(entity)

    The number of variational parameters in `entity`.

    # Implementation

    Must return a non-negative integer.

    """
    function count end

    """
        names(entity)

    An ordered list of human-readable names for each variational parameter in `entity`.

    # Implementation

    Must return a vector of strings.

    """
    function names end

    """
        values(entity)

    An ordered list of the numerical values for each variational parameter in `entity`.

    # Implementation

    Must return a vector of some float type.

    """
    function values end

    """
        bind(entity, x̄::AbstractVector)

    Assigns new values for each variational parameter in `entity`.

    # Implementation

    This method should mutate `entity` such that, for example,
        the expression `bind(entity, x̄); ȳ = values(entity); x̄ == ȳ` evaluates true.
    There is no return value.

    """
    function bind end
end

##########################################################################################
#= UTILITIES =#

"""
    TempArrays

Maintain caches for pre-allocated arrays used only temporarily within a function.

Do NOT use this module for arrays
    whose data are accessible outside the function in which they are created.

"""
module TempArrays; include("utils/TempArrays.jl"); end

"""
    Quples

Qubit tuples: simple types to represent couplings within a device.

"""
module Quples; include("utils/Quples.jl"); end
import .Quples: Quple

"""
    LinearAlgebraTools

Implement some frequently-used linear-algebraic operations.

Much of this functionality is available in Julia's standard `LinearAlgebra` library,
    but the implementations here might take advantage of pre-allocated `TempArrays`
    and (sometimes) efficient tensor contractions.

"""
module LinearAlgebraTools; include("utils/LinearAlgebraTools.jl"); end


##########################################################################################
#= ENUMERATIONS =#

"""
    Bases

Enumerates various linear-algebraic bases for representing statevectors and matrices.

"""
module Bases; include("enums/Bases.jl"); end
import .Bases: BasisType, LocalBasis
import .Bases: DRESSED, OCCUPATION, COORDINATE, MOMENTUM

"""
    Operators

Enumerates various categories of Hermitian observable related to a device.

"""
module Operators; include("enums/Operators.jl"); end
import .Operators: OperatorType, StaticOperator
import .Operators: Qubit, Channel, Drive, Hamiltonian, Gradient
import .Operators: IDENTITY, COUPLING, UNCOUPLED, STATIC


##########################################################################################
#= INTEGRATIONS =#

"""
    Integrations

Encapsulations of everything you need to know to integrate over time.

"""
module Integrations; include("integrals/Integrations.jl"); end
import .Integrations: IntegrationType
import .Integrations: nsteps, timeat, stepat, lattice, integrate
import .Integrations: starttime, endtime, duration, stepsize

module TrapezoidalIntegrations; include("integrals/TrapezoidalIntegrations.jl"); end
import .TrapezoidalIntegrations: TrapezoidalIntegration

##########################################################################################
#= SIGNALS =#

"""
    Signals

Time-dependent functions suitable for control signals with variational parameters.

The main motivation of this module
    is to provide a common interface for analytical gradients and optimization.

"""
module Signals; include("signals/Signals.jl"); end
import .Signals: SignalType
import .Signals: valueat, partial

module ParametricSignals; include("signals/ParametricSignals.jl"); end
import .ParametricSignals: ParametricSignal, ConstrainedSignal
import .ParametricSignals: parameters

module CompositeSignals; include("signals/CompositeSignals.jl"); end
import .CompositeSignals: CompositeSignal, WeightedCompositeSignal

module ModulatedSignals; include("signals/ModulatedSignals.jl"); end
import .ModulatedSignals: ModulatedSignal

module WindowedSignals; include("signals/WindowedSignals.jl"); end
import .WindowedSignals: WindowedSignal

module ConstantSignals; include("signals/ConstantSignals.jl"); end
import .ConstantSignals: Constant, ComplexConstant, PolarComplexConstant

module IntervalSignals; include("signals/IntervalSignals.jl"); end
import .IntervalSignals: Interval, ComplexInterval

module StepFunctionSignals; include("signals/StepFunctionSignals.jl"); end
import .StepFunctionSignals: StepFunction

module GaussianSignals; include("signals/GaussianSignals.jl"); end
import .GaussianSignals: Gaussian

module TrigonometricSignals; include("signals/TrigonometricSignals.jl"); end
import .TrigonometricSignals: Sine

module SincSignals; include("signals/SincSignals.jl"); end
import .SincSignals: Sinc

module TanhSignals; include("signals/TanhSignals.jl"); end
import .TanhSignals: Tanh

##########################################################################################
#= DEVICES =#

"""
    Devices

*In silico* representation of quantum devices, in which quantum states evolve in time.

In this package,
    the "static" components (ie. qubit frequencies, couplings, etc.)
    and the "drive" components (ie. control signal, variational parameters, etc.)
    are *all* integrated into a single `DeviceType` object.
All you need to know how a quantum state `ψ` evolves up time `T` is in the device.

"""
module Devices; include("devices/Devices.jl"); end
import .Devices: DeviceType, LocallyDrivenDevice
import .Devices: nqubits, nstates, nlevels, ndrives, ngrades, gradient
import .Devices: operator, propagator, propagate!, expectation, braket
import .Devices: drivequbit, gradequbit
import .Devices: resonancefrequency, drivefrequency, detuningfrequency
import .Devices: drivesignal, set_drivesignal

module TransmonDevices; include("devices/TransmonDevices.jl"); end
import .TransmonDevices: TransmonDevice, FixedFrequencyTransmonDevice

##########################################################################################
#= EVOLUTION ALGORITHMS =#

"""
    Evolutions

Algorithms to run time evolution, and related constructs like gradient signals.

NOTE: the `trapezoidaltimegrid` function is a utility.
    But it is implicitly tied to evolutions. I'm not sure what to do about it.
    It is an awkward space, similar to QubitOperators below.

"""
module Evolutions; include("evols/Evolutions.jl"); end
import .Evolutions: EvolutionType
import .Evolutions: workbasis, evolve, evolve!, gradientsignals

module ToggleEvolutions; include("evols/ToggleEvolutions.jl"); end
import .ToggleEvolutions: TOGGLE

module DirectEvolutions; include("evols/DirectEvolutions.jl"); end
import .DirectEvolutions: DIRECT

#= TODO (mid): Nick thinks all the evolutions should be promoted to the main module.

        I'm not sold.
        I think if a package doesn't really depend on a third party,
            it shouldn't list that third party as a dependency.

That said, we *do* need to get at *least* ODE caught up and with tests...
And of course, Lanczos is basically trivial...

=#

##########################################################################################
#= MORE UTILITIES =#

"""
    QubitOperators

Interfaces arbitrary physical Hilbert space with a strictly binary logical Hilbert space.

NOTE: I don't especially like how this module is organized,
    so consider this code subject to change.

"""
module QubitOperators; include("utils/QubitOperators.jl"); end

##########################################################################################
#= COST FUNCTIONS =#

"""
    CostFunctions

Interfaces time-evolution and penalty functions to interface directly with optimization.

Each distinct cost function is implemented in a sub-module,
    alongside its gradient function.
The sub-module implements a `functions` method which constructs a cost function
    and its corresponding gradient function.
Thus, all you need to run a gradient based optimization is:

    import CostFunctions.MySubModule: functions
    f, g = functions(args...)

"""
module CostFunctions; include("costfns/CostFunctions.jl"); end
import .CostFunctions: CostFunctionType, CompositeCostFunction
import .CostFunctions: cost_function, grad_function, grad_function_inplace
import .CostFunctions: EnergyFunction, ConstrainedEnergyFunction
import .CostFunctions: trajectory_callback

#= ENERGY FUNCTIONS =#
module BareEnergies; include("costfns/BareEnergies.jl"); end
import .BareEnergies: BareEnergy

module ProjectedEnergies; include("costfns/ProjectedEnergies.jl"); end
import .ProjectedEnergies: ProjectedEnergy

module Normalizations; include("costfns/Normalizations.jl"); end
import .Normalizations: Normalization

module NormalizedEnergies; include("costfns/NormalizedEnergies.jl"); end
import .NormalizedEnergies: NormalizedEnergy

#= PENALTY FUNCTIONS =#
module GlobalAmplitudeBounds; include("costfns/GlobalAmplitudeBounds.jl"); end
import .GlobalAmplitudeBounds: GlobalAmplitudeBound

module GlobalFrequencyBounds; include("costfns/GlobalFrequencyBounds.jl"); end
import .GlobalFrequencyBounds: GlobalFrequencyBound

module AmplitudeBounds; include("costfns/AmplitudeBounds.jl"); end
import .AmplitudeBounds: AmplitudeBound

module SoftBounds; include("costfns/SoftBounds.jl"); end
import .SoftBounds: SoftBound

module HardBounds; include("costfns/HardBounds.jl"); end
import .HardBounds: HardBound

module SmoothBounds; include("costfns/SmoothBounds.jl"); end
import .SmoothBounds: SmoothBound

#= TODO (mid): Global RMS penalty on selected parameters. =#
#= TODO (mid): Global RMS penalty on diff of selected parameters. =#

#= TODO (lo): Some way to pre-constrain x̄ BEFORE energy function
    Eg. activator function, see tensorflow tutorial for inspiration. =#




##########################################################################################
#= RECIPES =#

"""
    TemporalLattice(T::AbstractFloat, r::Int)

Semantic shorthand for constructing a `TrapezoidalIntegration`,
    which is the only one supported for, probably, all time. ^_^

# Arguments
- `T::AbstractFloat`: the total pulse duration
- `r::Int`: the number of Trotter steps

"""
function TemporalLattice(T::AbstractFloat, r::Int)
    return TrapezoidalIntegration(zero(T), T, r)
end

"""
    Systematic(TransmonDeviceType, n, pulses; kwargs...)

Standardized constructor for a somewhat realistic transmon device, but of arbitrary size.

This is a linearly coupled device,
    with uniformly-spaced resonance frequencies,
    and with all coupling and anharmonicity constants equal for each qubit.
The actual values of each constant are meant to roughly approximate a typical IBM device.

# Arguments
- `TransmonDeviceType`: the type of the device to be constructed
- `n::Int`: the number of qubits in the device
- `pulses`: a vector of control signals (`Signals.SignalType`), or one to be copied

# Keyword Arguments
- `m::Int`: the number of transmon levels to include in simulations (defaults to 2)
- `F`: the float type to use for device parameters (defaults to `Float64`)

"""
function Systematic(
    TransmonDeviceType::Type{<:TransmonDevices.AbstractTransmonDevice},
    n::Int,
    pulses;
    m=2,
    F=Float64,
)
    # INTERPRET SCALAR `pulses` AS A TEMPLATE TO BE COPIED
    Ω̄ = (pulses isa Signals.SignalType) ? [deepcopy(pulses) for _ in 1:n] : pulses

    # DEFINE STANDARDIZED PARAMETERS
    ω0 = F(2π * 4.80)
    Δω = F(2π * 0.02)
    δ0 = F(2π * 0.30)
    g0 = F(2π * 0.02)

    # ASSEMBLE THE DEVICE
    ω̄ = collect(ω0 .+ (Δω * (1:n)))
    δ̄ = fill(δ0, n)
    ḡ = fill(g0, n-1)
    quples = [Quple(q,q+1) for q in 1:n-1]
    q̄ = 1:n
    ν̄ = copy(ω̄)
    return TransmonDeviceType(ω̄, δ̄, ḡ, quples, q̄, ν̄, Ω̄, m)
end

"""
    FullyTrotterized(signal::Signals.SignalType, T::Real, r::Int)

Break a signal up so that each time-step is parameterized separately.

Usually you'll want to use this with constant signals.

"""
function FullyTrotterized(signal::Signals.SignalType, T::Real, r::Int)
    τ, _, t̄ = Evolutions.trapezoidaltimegrid(T,r)
    starttimes = t̄ .- (τ/2)
    return WindowedSignal(
        [deepcopy(signal) for t in starttimes],
        starttimes,
    )
end

"""
    UniformWindowed(signal::Signals.SignalType, T::Real, W::Int)

Break a signal up into equal-sized windows.

Usually you'll want to use this with constant signals.

"""
function UniformWindowed(signal::Signals.SignalType, T::Real, W::Int)
    starttimes = range(zero(T), T, W+1)[1:end-1]
    return WindowedSignal(
        [deepcopy(signal) for t in starttimes],
        starttimes,
    )
end

end # module CtrlVQE
