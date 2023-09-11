export OperatorType, StaticOperator
export Qubit, Channel, Drive, Hamiltonian, Gradient
export IDENTITY, COUPLING, UNCOUPLED, STATIC

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