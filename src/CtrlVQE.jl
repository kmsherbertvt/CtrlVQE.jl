module CtrlVQE

"""
    TempArrays

Maintain caches for pre-allocated arrays used only temporarily within a function.

Do NOT use this module for arrays
    whose data are accessible outside the function in which they are created.

"""
module TempArrays
    using Memoization: @memoize

    """
        array(::F, shape::Tuple, index=nothing)

    Fetch a temporary array with type `F` and shape `shape`.

    The `index` parameter is an additional unique key,
        allowing the module to cache mulitple arrays of the same type and shape.
    You should pass `index=Symbol(@__MODULE__)` to prevent collisions across modules.
    You may pass a tuple, eg. `index=(Symbol(@__MODULE__), :otherkey)`
        to prevent collisions within a module.

    """
    @memoize function array(::F, shape::Tuple, index=nothing) where {F<:Number}
        # TODO (lo): Thread-safe and hands-off approach to this.
        return Array{F}(undef, shape)
    end

    """
        array(F::Type{<:Number}, shape::Tuple, index=nothing)

    Same as above but passing the type directly, rather than an instance of the type.

    """
    function array(F::Type{<:Number}, shape::Tuple, index=nothing)
        return array(zero(F), shape, index)
    end
end

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


"""
    Bases

Enumerates various linear-algebraic bases for representing statevectors and matrices.

"""
module Bases
    abstract type BasisType end

    """
        Dressed(), aka DRESSED

    The eigenbasis of the static Hamiltonian associated with a `Device`.
    Eigenvectors are ordered to maximize similarity with an identity matrix.
    Phases are fixed so that the diagonal is real.

    """
    struct Dressed      <: BasisType end;       const DRESSED = Dressed()

    abstract type LocalBasis <: BasisType end

    """
        Occupation(), aka OCCUPATION

    The eigenbasis of local number operators ``n̂ ≡ a'a``.
    Generally equivalent to what is called the "Z basis", or the "computational basis".

    """
    struct Occupation   <: LocalBasis end;      const OCCUPATION = Occupation()

    """
        Coordinate(), aka COORDINATE

    The eigenbasis of local quadrature operators ``Q ≡ (a + a')/√2``.

    """
    struct Coordinate   <: LocalBasis end;      const COORDINATE = Coordinate()

    """
        Momentum(), aka MOMENTUM

    The eigenbasis of local quadrature operators ``P ≡ i(a - a')/√2``.

    """
    struct Momentum     <: LocalBasis end;      const MOMENTUM   = Momentum()
end

"""
    Operators

Enumerates various categories of Hermitian observable related to a device.

"""
module Operators
    abstract type OperatorType end
    abstract type StaticOperator <: OperatorType end

    """
        Identity(), aka IDENTITY

    The identity operator.

    """
    struct Identity <: StaticOperator end
    const IDENTITY = Identity()

    """
        Qubit(q)

    The component of the static Hamiltonian which is local to qubit `q`.

    For example, in a transmon device,
        `Qubit(2)` represents a term ``ω_q a_q'a_q - δ_q/2~ a_q'a_q'a_q a_q``.

    """
    struct Qubit <: StaticOperator
        q::Int
    end

    """
        Coupling(), aka COUPLING

    The components of the static Hamiltonian which are non-local to any one qubit.

    For example, in a transmon device,
        `Coupling()` represents the sum ``∑_{p,q} g_{pq} (a_p'a_q + a_q'a_p)``.

    """
    struct Coupling <: StaticOperator end
    const COUPLING = Coupling()

    """
        Uncoupled(), aka UNCOUPLED

    The components of the static Hamiltonian which are local to each qubit.

    This represents the sum of each `Qubit(q)`,
        where `q` iterates over each qubit in the device.

    For example, in a transmon device,
        `Uncoupled()` represents the sum ``∑_q (ω_q a_q'a_q - δ_q/2~ a_q'a_q'a_q a_q)``.

    """
    struct Uncoupled <: StaticOperator end
    const UNCOUPLED = Uncoupled()

    """
        Static(), aka STATIC

    All components of the static Hamiltonian.

    This represents the sum of `Uncoupled()` and `Coupled()`

    """
    struct Static <: StaticOperator end
    const STATIC = Static()

    """
        Channel(i,t)

    An individual drive term (indexed by `i`) at a specific time `t`.

    For example, in a transmon device,
        `Channel(q,t)` might represent ``Ω_q(t) [\\exp(iν_qt) a_q + \\exp(-iν_qt) a_q']``,
        the drive for a single qubit.

    Note that you are free to have multiple channels for each qubit,
        or channels which operate on multiple qubits.

    """
    struct Channel{R<:Real} <: OperatorType
        i::Int
        t::R
    end

    """
        Drive(t)

    The sum of all drive terms at a specific time `t`.

    This represents the sum of each `Qubit(i)`,
        where `i` iterates over each drive term in the device.

    For example, in a transmon device,
        `Drive(t)` might represent ``∑_q Ω_q(t) [\\exp(iν_qt) a_q + \\exp(-iν_qt) a_q']``.

    """
    struct Drive{R<:Real} <: OperatorType
        t::R
    end

    """
        Hamiltonian(t)

    The full Hamiltonian at a specific time `t`.

    This represents the sum of `Static()` and `Drive(t)`.

    """
    struct Hamiltonian{R<:Real} <: OperatorType
        t::R
    end

    """
        Gradient(j,t)

    An individual gradient operator (indexed by `j`) at a specific time `t`.

    The gradient operators appear in the derivation of each gradient signal,
        which are used to calculate analytical gradients of each variational parameter.
    The gradient operators are very closely related to individual channel operators,
        but sufficiently distinct that they need to be treated separately.

    For example, for a transmon device,
        each channel operator ``Ω_q(t) [\\exp(iν_qt) a_q + \\exp(-iν_qt) a_q']``
        is associated with *two* gradient operators:
    - ``\\exp(iν_qt) a_q + \\exp(-iν_qt) a_q'``
    - ``i[\\exp(iν_qt) a_q - \\exp(-iν_qt) a_q']``

    """
    struct Gradient{R<:Real} <: OperatorType
        j::Int
        t::R
    end
end
import .Operators: IDENTITY, COUPLING, UNCOUPLED, STATIC

"""
    Quples

Qubit tuples: simple types to represent couplings within a device.

"""
module Quples
    """
        Quple(q1,q2)

    A (symmetric) coupling between qubits indexed by `q1` and `q2`.

    Note that the order is irrelevant: `Quple(q1,q2) == Quple(q2,q1)`.

    """
    struct Quple
        q1::Int
        q2::Int
        # INNER CONSTRUCTOR: Constrain order so that `Quple(q1,q2) == Quple(q2,q1)`.
        Quple(q1, q2) = q1 > q2 ? new(q2, q1) : new(q1, q2)
    end

    # IMPLEMENT ITERATION, FOR CONVENIENT UNPACKING
    Base.iterate(quple::Quple) = quple.q1, true
    Base.iterate(quple::Quple, state) = state ? (quple.q2, false) : nothing
end
import .Quples: Quple

"""
    LinearAlgebraTools

Implement some frequently-used linear-algebraic operations.

Much of this functionality is available in Julia's standard `LinearAlgebra` library,
    but the implementations here might take advantage of pre-allocated `TempArrays`
    and (sometimes) efficient tensor contractions.

"""
module LinearAlgebraTools; include("LinearAlgebraTools.jl"); end

"""
    Signals

Time-dependent functions suitable for control signals with variational parameters.

The main motivation of this module
    is to provide a common interface for analytical gradients and optimization.

"""
module Signals; include("Signals.jl"); end

module ParametricSignals; include("signals/ParametricSignals.jl"); end
import .ParametricSignals: ParametricSignal, parameters, ConstrainedSignal

module CompositeSignals; include("signals/CompositeSignals.jl"); end
import .CompositeSignals: CompositeSignal

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

"""
    Devices

*In silico* representation of quantum devices, in which quantum states evolve in time.

In this package,
    the "static" components (ie. qubit frequencies, couplings, etc.)
    and the "drive" components (ie. control signal, variational parameters, etc.)
    are *all* integrated into a single `Device` object.
All you need to know how a quantum state `ψ` evolves up time `T` is in the device.

"""
module Devices; include("Devices.jl"); end
import .Devices: nqubits, nstates, nlevels, ndrives, ngrades

module LocallyDrivenDevices; include("devices/LocallyDrivenDevices.jl"); end
import .LocallyDrivenDevices: LocallyDrivenDevice

module TransmonDevices; include("devices/TransmonDevices.jl"); end
import .TransmonDevices: TransmonDevice, FixedFrequencyTransmonDevice


"""
    Evolutions

Algorithms to run time evolution, and related constructs like gradient signals.

"""
module Evolutions; include("Evolutions.jl"); end
import .Evolutions: trapezoidaltimegrid, evolve, evolve!, gradientsignals, Rotate

"""
    QubitOperators

Interfaces arbitrary physical Hilbert space with a strictly binary logical Hilbert space.

NOTE: I don't especially like how this module is organized,
    so consider this code subject to change.

"""
module QubitOperators
    include("QubitOperators.jl")
end

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
module CostFunctions; include("CostFunctions.jl"); end
import .CostFunctions: cost_function, grad_function, grad_function_byvalue
import .CostFunctions: CompositeCostFunction

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




#= RECIPES =#

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
- `pulses`: a vector of control signals (`Signals.AbstractSignal`), or one to be copied

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
    Ω̄ = (pulses isa Signals.AbstractSignal) ? [deepcopy(pulses) for _ in 1:n] : pulses

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
    FullyTrotterized(signal::Signals.AbstractSignal, T::Real, r::Int)

Break a signal up so that each time-step is parameterized separately.

Usually you'll want to use this with constant signals.

"""
function FullyTrotterized(signal::Signals.AbstractSignal, T::Real, r::Int)
    τ, _, t̄ = Evolutions.trapezoidaltimegrid(T,r)
    starttimes = t̄ .- (τ/2)
    return Signals.WindowedSignal(
        [deepcopy(signal) for t in starttimes],
        starttimes,
    )
end

"""
    UniformWindowed(signal::Signals.AbstractSignal, T::Real, W::Int)

Break a signal up into equal-sized windows.

Usually you'll want to use this with constant signals.

"""
function UniformWindowed(signal::Signals.AbstractSignal, T::Real, W::Int)
    starttimes = range(zero(T), T, W+1)[1:end-1]
    return Signals.WindowedSignal(
        [deepcopy(signal) for t in starttimes],
        starttimes,
    )
end

end # module CtrlVQE
