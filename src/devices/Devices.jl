import ..Parameters
export DeviceType, LocallyDrivenDevice
export nqubits, nstates, nlevels, ndrives, ngrades, gradient
export operator, propagator, propagate!, expectation, braket
export drivequbit, gradequbit

import ..LinearAlgebraTools
import ..Bases, ..Operators
import ..Integrations

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

import ..LinearAlgebraTools: MatrixList
const Evolvable = AbstractVecOrMat{<:Complex{<:AbstractFloat}}

import ..Signals: SignalType

using Memoization: @memoize
using LinearAlgebra: I, Diagonal, Hermitian, Eigen, eigen

"""
    DeviceType{F,FΩ}

Super-type for all device objects.

# Type Parameters
- `F`: the float type associated with a device
- `FΩ`: the number type associated with the device's drive signals. May be complex.

# Implementation

Any concrete sub-type `D` must implement all functions in the `Parameters` module.
- In particular, if any static operators in your device depend on variational parameters,
    you should consult the "Note on Caching" below.

In addition, all methods in the following sections must be implemented.
- Counting methods
- Algebra methods
- Operator methods
- Type methods
- Frequency methods
- Signal methods
- Gradient methods

If your device's drive channels are all local,
    you should implement a `LocallyDrivenDevice`,
    which has a few extra requirements.

## Counting methods:

- `nqubits(::D)`: the number of qubits in the device - call this `n`.
- `nlevels(::D)`: the number of physical levels in each "qubit" - call this `m`.
- `ndrives(::D)`: the number of distinct drive channels.
- `ngrades(::D)`: the number of distinct gradient operators.

Each of these methods returns an integer.

## Algebra methods:

- `localloweringoperator(::D)`:
        an `m × m` matrix applying the lowering operator `a` to a single qubit.

This method should define `result=nothing` as a keyword argument;
    when passed, use it as the array to store your result in.

## Operator methods:

- `qubithamiltonian(::D, ā, q::Int)`:
        the static components of the device Hamiltonian local to qubit q.

- `staticcoupling(::D, ā)`:
        the static components of the device Hamiltonian nonlocal to any one qubit.

- `driveoperator(::D, ā, i::Int, t::Real)`:
        the distinct drive operator for channel `i` at time `t`

- `gradeoperator(::D, ā, j::Int, t::Real)`:
        the distinct gradient operator indexed by `j` at time `t`

Each of these methods should define `result=nothing` as a keyword argument;
    when passed, use it as the array to store your result in.

Each of these methods takes a 3darray `ā`;
    the annihilation operator ``a_q`` is given by the matrix `ā[:,:,q]`.
These methods should construct a new matrix as a function of each ``a_q``.
Usually, each `ā[:,:,q]` is defined on the full Hilbert space (ie. `m^n × m^n`),
    but sometimes the code exploits a simple tensor structure
    by passing in local `m × m` operators instead,
    so do not assume a specific size a priori.

The annihilation operators ``a_q`` and their adjoints ``a_q'`` form a complete algebra,
    so it is always possible to express any operator given just `ā`.
For example, the Pauli spin matrices in a two-level system can be expressed
    as ``X=a+a'``, ``Y=i(a-a')``, and ``Z=a'a``.
If you *really* want to write your operators as functions of something other than ``a_q``,
    you may "hack" in a new algebra by implementing new methods
    for this module's `eltype_algebra`, `algebra`, and `localalgebra` functions.

## Type methods:

- `eltype_localloweringoperator(::D)`
- `eltype_qubithamiltonian(::D)`
- `eltype_staticcoupling(::D)`
- `eltype_driveoperator(::D)`
- `eltype_gradeoperator(::D)`

Each of these methods gives the number type of the corresponding operator.
Implement these methods based only on your implementation of the methods,
    ie. they should be independent of the type of `ā`.

## Frequency methods:

- `resonancefrequency(::D, q::Int)`
- `drivefrequency(::D, i::Int)`

Each of these methods gives a scalar.

## Drive signal methods:

- `__get__drivesignals(::D)`

This method exposes a vector of `SignalType`, of length `ndrives`.
It is used by the default implementations of `drivesignal` and `set_drivesignal`.

## Gradient methods:

- `gradient(::D, grid::Integrations.IntegrationType, ϕ̄)`:
        the gradient vector for each variational parameter in the device.

Each partial is generally an integral over at least one gradient signal.
The argument `grid` identifies the temporal lattice on which ϕ̄ is defined.
The argument `ϕ̄` is a 2d array; `ϕ̄[:,:,j]` contains the jth gradient signal
    ``ϕ_j(t)`` evaluated at each point in `grid`.

This method should define `result=nothing` as a keyword argument;
    when passed, use it as the array to store your result in.

## Notes on Caching

This module uses the `Memoization` package to cache some arrays as they are calculated.

This does not apply to any method which depends on an absolute time t,
    though it does apply to methods depending only on a relative time τ.
For example, the propagator for a static Hamiltonian is cached,
    but not one for a drive Hamiltonian.
No caching happens for any method if its `result` keyword argument is used.

Usually, variational parameters only affect time-dependent methods,
    but if any of your device's static operators do depend on a variational parameter,
    you should be careful to empty the cache when `Parameters.bind` is called.

You can completely clear everything in the cache with:

    Memoization.empty_all_caches!()

Alternatively, selectively clear caches for affected functions via:

    Memoization.empty_cache!(fn)

I don't know if it's possible to selectively clear cached values for specific methods.
If it can be done, it would require obtaining the actual `Dict`
    being used as a cache for a particular function,
    figuring out exactly how that cache is indexed,
    and manually removing elements matching your targeted method signature.

"""
abstract type DeviceType{F,FΩ} end

#= TODO (mid):
    Now that abstract devices have a definite typing,
        are all the `eltype_<operator>` methods necessary still?
    Can we, perchance, just assume they are `F`?
    Not QUITE, because they could be type `F` or type `Complex{F}`.
    But we'll have to think through all this carefully;
        maybe there is a way to short-circuit some of them.
=#

"""
    nqubits(device::DeviceType)

The number of qubits in the device.

"""
function nqubits(::DeviceType)
    error("Not Implemented")
    return 0
end

"""
    nlevels(device::DeviceType)

The number of physical levels in each "qubit".

"""
function nlevels(::DeviceType)
    error("Not Implemented")
    return 0
end

"""
    ndrives(device::DeviceType)

The number of distinct drive channels.

"""
function ndrives(::DeviceType)
    error("Not Implemented")
    return 0
end
#= TODO (mid): we could choose to automate from __get__drivesignals =#

"""
    ngrades(device::DeviceType)

The number of distinct gradient operators.

"""
function ngrades(::DeviceType)
    error("Not Implemented")
    return 0
end

"""
    localloweringoperator(device::DeviceType; result=nothing)

The lowering operator ``a`` acting on the Hilbert space of a single physical qubit.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function localloweringoperator(::DeviceType; result=nothing)
    error("Not Implemented")
    return zeros(eltype_localloweringoperator(device), nlevels(device), nlevels(device))
end

"""
    qubithamiltonian(device::DeviceType, ā::MatrixList, q::Int; result=nothing)

The static components of the device Hamiltonian local to qubit `q`.

This method is a function of annihilation operators ``a_q`` given by `ā[:,:,q]`,
    which may be matrices acting on a physical Hilbert space either globally or locally.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function qubithamiltonian(::DeviceType, ā::MatrixList, q::Int; result=nothing)
    error("Not Implemented")
    return zeros(eltype(ā), size(ā)[1:end-1])
end

"""
    staticcoupling(device::DeviceType, ā::MatrixList, q::Int; result=nothing)

The static components of the device Hamiltonian nonlocal to any one qubit.

This method is a function of annihilation operators ``a_q`` given by `ā[:,:,q]`,
    which are matrices acting globally on the physical Hilbert space.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function staticcoupling(::DeviceType, ā::MatrixList; result=nothing)
    error("Not Implemented")
    return zeros(eltype(ā), size(ā)[1:end-1])
end

"""
    driveoperator(device::DeviceType, ā::MatrixList, i::Int, t::Real; result=nothing)

The distinct drive operator for channel `i` at time `t`.

This method is a function of annihilation operators ``a_q`` given by `ā[:,:,q]`,
    which are matrices acting globally on the physical Hilbert space.
If `device` is a `LocallyDrivenDevice`,
    the matrices may also act on a local physical Hilbert space for each individual qubit.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function driveoperator(::DeviceType, ā::MatrixList, i::Int, t::Real; result=nothing)
    error("Not Implemented")
    return zeros(eltype(ā), size(ā)[1:end-1])
end

"""
    gradeoperator(device::DeviceType, ā::MatrixList, j::Int, t::Real; result=nothing)

The distinct gradient operator indexed by `j` at time `t`.

I have defined the "gradient operator" ``Â_j`` as the Hermitian operator
    for which the jth gradient signal is ``ϕ_j = ⟨λ|(iÂ_j)|ψ⟩ + h.t.``.

This method is a function of annihilation operators ``a_q`` given by `ā[:,:,q]`,
    which are matrices acting globally on the physical Hilbert space.
If `device` is a `LocallyDrivenDevice`,
    the matrices may also act on a local physical Hilbert space for each individual qubit.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function gradeoperator(::DeviceType, ā::MatrixList, j::Int, t::Real; result=nothing)
    error("Not Implemented")
    return zeros(eltype(ā), size(ā)[1:end-1])
end


"""
    eltype_localloweringoperator(device::DeviceType)

The number type of a local lowering operator for this device.

"""
function eltype_localloweringoperator(::DeviceType)
    error("Not Implemented")
    return Bool
end

"""
    eltype_qubithamiltonian(device::DeviceType)

The number type of the local static components of the Hamiltonian for this device.

The number type of the algebra `ā` is ignored for the purposes of this method.

"""
function eltype_qubithamiltonian(::DeviceType)
    error("Not Implemented")
    return Bool
end

"""
    eltype_staticcoupling(device::DeviceType)

The number type of the non-local static components of the Hamiltonian for this device.

The number type of the algebra `ā` is ignored for the purposes of this method.

"""
function eltype_staticcoupling(::DeviceType)
    error("Not Implemented")
    return Bool
end

"""
    eltype_driveoperator(device::DeviceType)

The number type of the time-dependent drive channels for this device.

The number type of the algebra `ā` is ignored for the purposes of this method.

"""
function eltype_driveoperator(::DeviceType)
    error("Not Implemented")
    return Bool
end

"""
    eltype_gradeoperator(device::DeviceType)

The number type of the gradient operators for this device.

The number type of the algebra `ā` is ignored for the purposes of this method.

"""
function eltype_gradeoperator(::DeviceType)
    error("Not Implemented")
    return Bool
end

"""
    gradient(::DeviceType, grid::Integrations.IntegrationType, ϕ̄; result=nothing)

The gradient vector of partials for each variational parameter in the device.

Each partial is generally an integral over at least one gradient signal.
The argument `grid` identifies the temporal lattice on which ϕ̄ is defined.
The argument `ϕ̄` is a 2d array; `ϕ̄[:,:,j]` contains the jth gradient signal
    ``ϕ_j(t)`` evaluated at each point in `grid`.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function gradient(device::DeviceType,
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix;
    result=nothing,
)
    error("Not Implemented")
    return zero(Parameters.values(device))
end


"""
    resonancefrequency(device::DeviceType, q::Int)

The energy gap betwen the logical |0⟩ and |1⟩ states of a given qubit index.

"""
function resonancefrequency(device::DeviceType{F,FΩ}, q::Int) where {F,FΩ}
    error("Not Implemented")
    return zero(F)
end

"""
    drivefrequency(device::DeviceType, i::Int)

The drive frequency for a given pulse index.

NOTE: At present, the drive frequency is assumed to be constant.
    We may change it someday to a generic Signal,
        where the typical use-case is to use a real Constant signal.
    In any case it shall be real.

"""
function drivefrequency(device::DeviceType{F,FΩ}, i::Int) where {F,FΩ}
    error("Not Implemented")
    return zero(F)
end

"""
    __get__drivesignals(device::DeviceType)

A vector of drive signals, presumably an attribute of the device object.

"""
function __get__drivesignals(device::DeviceType{F,FΩ}) where {F,FΩ}
    error("Not Implemented")
    return Vector{<:TypeSignal{F,FΩ}}[]
end


#= IMPLEMENTED INTERFACE

All the methods above must be implemented by device sub-types.
All the methods below will work automatically.

=#

"""
    nstates(device::DeviceType)

The total number of states in the physical Hilbert space of the device.

(This is as opposed to `nlevels(device)`,
    the number of states in the physical Hilbert space of a single independent qubit.)

"""
nstates(device::DeviceType) = nlevels(device) ^ nqubits(device)


"""
    globalize(device::DeviceType, op::AbstractMatrix, q::Int; result=nothing)

Extend a local operator `op` acting on qubit `q` into the global Hilbert space.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function globalize(
    device::DeviceType, op::AbstractMatrix{F}, q::Int;
    result=nothing,
) where {F}
    if isnothing(result)
        N = nstates(device)
        result = Matrix{F}(undef, N, N)
    end

    m = nlevels(device)
    n = nqubits(device)
    ops = array(F, (m,m,n), LABEL)
    for p in 1:n
        ops[:,:,p] .= p == q ? op : Matrix(I, m, m)
    end
    return LinearAlgebraTools.kron(ops; result=result)
end

"""
    diagonalize(basis::Bases.BasisType, device::DeviceType)

Compute the vector of eigenvalues `Λ` and the rotation matrix `U` for a given basis.

`U` is an operator acting on the global Hilbert space of the device.

The result is packed into a `LinearAlgebra.Eigen` object,
    but it may be unpacked by `Λ, U = diagonalize(basis, device)`.

    diagonalize(basis::Bases.LocalBasis, device::DeviceType, q::Int)

Same as above, except that `U` acts on the local Hilbert space of qubit `q`.

Note that you can still construct
    the global rotation matrix of a local basis by using the first method.

"""
function diagonalize end

@memoize Dict function diagonalize(::Bases.Dressed, device::DeviceType)
    H0 = operator(Operators.STATIC, device)

    N = size(H0)[1]
    Λ, U = eigen(Hermitian(H0))

    # IMPOSE PERMUTATION
    σ = Vector{Int}(undef,N)
    for i in 1:N
        perm = sortperm(abs.(U[i,:]), rev=true) # STABLE SORT BY "ith" COMPONENT
        perm_ix = 1                             # CAREFULLY HANDLE TIES
        while perm[perm_ix] ∈ @view(σ[1:i-1])
            perm_ix += 1
        end
        σ[i] = perm[perm_ix]
    end
    Λ .= Λ[  σ]
    U .= U[:,σ]

    # IMPOSE PHASE
    for i in 1:N
        U[:,i] .*= U[i,i] < 0 ? -1 : 1          # ASSUMES REAL TYPE
        # U[:,i] .*= exp(-im*angle(U[i,i]))        # ASSUMES COMPLEX TYPE
    end

    # IMPOSE ZEROS
    Λ[abs.(Λ) .< eps(eltype(Λ))] .= zero(eltype(Λ))
    U[abs.(U) .< eps(eltype(U))] .= zero(eltype(U))

    return Eigen(Λ,U)

    # TODO (mid): Each imposition really ought to be a separate function.
    # TODO (mid): Phase imposition should accommodate real or complex H0.
    # TODO (mid): Strongly consider imposing phase on the local bases also.
end

@memoize Dict function diagonalize(basis::Bases.LocalBasis, device::DeviceType)
    F = eltype_localloweringoperator(device)
        # NOTE: May not be correct, if we ever introduce a complex local basis!

    m = nlevels(device)
    n = nqubits(device)
    λ̄ = Array{F,2}(undef, m, n)
    ū = Array{F,3}(undef, m, m, n)
    for q in 1:n
        ΛU = diagonalize(basis, device, q)
        λ̄[:,q] .= ΛU.values
        ū[:,:,q] .= ΛU.vectors
    end
    Λ = LinearAlgebraTools.kron(λ̄)
    U = LinearAlgebraTools.kron(ū)
    return Eigen(Λ, U)
end

@memoize Dict function diagonalize(::Bases.Occupation, device::DeviceType, q::Int)
    F = eltype_localloweringoperator(device)
    m = nlevels(device)
    identity = Matrix{F}(I, m, m)
    return eigen(Hermitian(identity))
end

@memoize Dict function diagonalize(::Bases.Coordinate, device::DeviceType, q::Int)
    a = localloweringoperator(device)
    Q = (a + a') / eltype(a)(√2)
    return eigen(Hermitian(Q))
end

@memoize Dict function diagonalize(::Bases.Momentum, device::DeviceType, q::Int)
    a = localloweringoperator(device)
    P = im*(a - a') / eltype(a)(√2)
    return eigen(Hermitian(P))
end

"""
    basisrotation(tgt::Bases.BasisType, src::Bases.BasisType, device::DeviceType)

Calculate the basis rotation `U` which transforms ``|ψ_{src}⟩ → |ψ_{tgt}⟩ = U|ψ_{src}⟩``.

"""
@memoize Dict function basisrotation(
    tgt::Bases.BasisType,
    src::Bases.BasisType,
    device::DeviceType,
)
    Λ0, U0 = diagonalize(src, device)
    Λ1, U1 = diagonalize(tgt, device)
    # |ψ'⟩ ≡ U0|ψ⟩ rotates |ψ⟩ OUT of `src` Bases.
    # U1'|ψ'⟩ rotates |ψ'⟩ INTO `tgt` Bases.
    return U1' * U0
end

@memoize Dict function basisrotation(
    tgt::Bases.LocalBasis,
    src::Bases.LocalBasis,
    device::DeviceType,
)
    ū = localbasisrotations(tgt, src, device)
    return LinearAlgebraTools.kron(ū)
end

"""
    basisrotation(tgt::Bases.LocalBasis, src::Bases.LocalBasis, device::DeviceType, q::Int)

Same as above, except that `U` acts on the local Hilbert space of qubit `q`.

This is used elsewhere for more efficient rotations exploiting tensor structure.

"""
@memoize Dict function basisrotation(
    tgt::Bases.LocalBasis,
    src::Bases.LocalBasis,
    device::DeviceType,
    q::Int,
)
    Λ0, U0 = diagonalize(src, device, q)
    Λ1, U1 = diagonalize(tgt, device, q)
    # |ψ'⟩ ≡ U0|ψ⟩ rotates |ψ⟩ OUT of `src` Bases.
    # U1'|ψ'⟩ rotates |ψ'⟩ INTO `tgt` Bases.
    return U1' * U0
end

"""
    localbasisrotations(tgt::Bases.LocalBasis, src::Bases.LocalBasis, device::DeviceType)

A matrix list `ū`, where `ū[:,:,q]` is a local basis rotation on qubit `q`.

"""
@memoize Dict function localbasisrotations(
    tgt::Bases.LocalBasis,
    src::Bases.LocalBasis,
    device::DeviceType,
)
    F = eltype_localloweringoperator(device)
        # NOTE: May not be correct, if we ever introduce a complex local basis!

    m = nlevels(device)
    n = nqubits(device)
    ū = Array{F,3}(undef, m, m, n)
    for q in 1:n
        ū[:,:,q] .= basisrotation(tgt, src, device, q)
    end
    return ū
end

"""
    eltype_algebra(device::DeviceType[, basis::Bases.BasisType])

The number type of each annihilation operator ``a_j`` represented in the given basis.

When omitted, the basis defaults to `Bases.OCCUPATION`.

"""
function eltype_algebra(device::DeviceType, ::Bases.BasisType=Bases.OCCUPATION)
    return eltype_localloweringoperator(device)
end

function eltype_algebra(device::DeviceType, ::Bases.Dressed)
    return promote_type(
        eltype_localloweringoperator(device),
        eltype_qubithamiltonian(device),
        eltype_staticcoupling(device),
    )
end


"""
    algebra(device::DeviceType[, basis::Bases.BasisType])

A matrix list `ā`, where `ā[:,:,q]` represents the annihilation operator ``a_q``.

When omitted, the basis defaults to `Bases.OCCUPATION`.

Each `ā[:,:,q]` acts globally on the full Hilbert space of the device,
    even in bases where it acts trivially on states outside the local space of qubit `q`.
To construct local operators, use `localalgebra` instead.

"""
function algebra(
    device::DeviceType,
    basis::Bases.BasisType=Bases.OCCUPATION,
)
    F = eltype_algebra(device, basis)
    U = basisrotation(basis, Bases.OCCUPATION, device)

    n = nqubits(device)
    N = nstates(device)
    ā = Array{F,3}(undef, N, N, n)
    a0 = localloweringoperator(device)
    for q in 1:n
        ā[:,:,q] .= globalize(device, a0, q)
        LinearAlgebraTools.rotate!(U, @view(ā[:,:,q]))
    end
    return ā
end

"""
    localalgebra(device::DeviceType[, basis::Bases.LocalBasis])

A matrix list `ā`, where `ā[:,:,q]` represents the annihilation operator ``a_q``.

When omitted, the basis defaults to `Bases.OCCUPATION`.

Each `ā[:,:,q]` acts locally on the physical Hilbert space of qubit `q`.
Note that you can construct global operators in a local basis by using `algebra` instead.

"""
@memoize Dict function localalgebra(
    device::DeviceType,
    basis::Bases.LocalBasis=Bases.OCCUPATION,
)
    # DETERMINE THE NUMBER TYPE COMPATIBLE WITH ROTATION
    F = eltype_algebra(device, basis)

    m = nlevels(device)
    n = nqubits(device)
    ā = Array{F,3}(undef, m, m, n)
    a0 = localloweringoperator(device)
    for q in 1:nqubits(device)
        ā[:,:,q] .= a0
        u = basisrotation(basis, Bases.OCCUPATION, device, q)
        LinearAlgebraTools.rotate!(u, @view(ā[:,:,q]))
    end
    return ā
end

"""
    Base.eltype(op::Operators.OperatorType, device::DeviceType[, basis::Bases.BasisType])

The number type of the matrix returned by `operator(op, device, basis)`.

When omitted, the basis defaults to Bases.OCCUPATION.

"""
function Base.eltype(op::Operators.OperatorType, device::DeviceType)
    return Base.eltype(op, device, Bases.OCCUPATION)
end

function Base.eltype(op::Operators.Identity, device::DeviceType, basis::Bases.BasisType)
    return Bool
end

function Base.eltype(::Operators.Qubit, device::DeviceType, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_qubithamiltonian(device),
    )
end

function Base.eltype(::Operators.Coupling, device::DeviceType, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_staticcoupling(device),
    )
end

function Base.eltype(::Operators.Channel, device::DeviceType, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_driveoperator(device),
    )
end

function Base.eltype(::Operators.Gradient, device::DeviceType, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_gradeoperator(device),
    )
end

function Base.eltype(::Operators.Uncoupled, device::DeviceType, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_qubithamiltonian(device),
    )
end

function Base.eltype(::Operators.Static, device::DeviceType, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_qubithamiltonian(device),
        eltype_staticcoupling(device),
    )
end

function Base.eltype(::Operators.Drive, device::DeviceType, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_driveoperator(device),
    )
end

function Base.eltype(::Operators.Hamiltonian, device::DeviceType, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_qubithamiltonian(device),
        eltype_staticcoupling(device),
        eltype_driveoperator(device),
    )
end

"""
    operator(op, device[, basis]; kwargs...)

A Hermitian operator describing a `device`, represented in the given `basis`.

For example, to construct the static Hamiltonian of a device in the dressed basis,
    call `operator(Operators.STATIC, device, Bases.DRESSED)`.

# Arguments
- `op::Operators.OperatorType`: which operator to construct (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operator will be represented in.
        Defaults to `Bases.OCCUPATION` when omitted.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    operator(op, device, basis, :cache)

For internal use only.
The extra positional argument enables dispatch to a cached function when appropriate.

"""
function operator(op::Operators.OperatorType, device::DeviceType; kwargs...)
    return operator(op, device, Bases.OCCUPATION; kwargs...)
end

@memoize Dict function operator(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    ::Symbol,
)
    F = eltype(op, device, basis)
    N = nstates(device)
    result = Matrix{F}(undef, N, N)
    return operator(op, device, basis; result=result)
end

@memoize Dict function operator(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType,
    ::Symbol,
)
    return Diagonal(ones(Bool, nstates(device)))
end

function operator(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return operator(op, device, basis, :cache)
    N = nstates(device)
    result .= Matrix(I, N, N)
    return result
end

function operator(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return operator(op, device, basis, :cache)
    ā = algebra(device, basis)
    return qubithamiltonian(device, ā, op.q; result=result)
end

function operator(
    op::Operators.Coupling,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return operator(op, device, basis, :cache)
    ā = algebra(device, basis)
    return staticcoupling(device, ā; result=result)
end

function operator(
    op::Operators.Channel,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    ā = algebra(device, basis)
    return driveoperator(device, ā, op.i, op.t; result=result)
end

function operator(
    op::Operators.Gradient,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    ā = algebra(device, basis)
    return gradeoperator(device, ā, op.j, op.t; result=result)
end

function operator(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return operator(op, device, basis, :cache)
    result .= 0
    for q in 1:nqubits(device)
        result .+= operator(Operators.Qubit(q), device, basis)
    end
    return result
end

function operator(
    op::Operators.Static,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return operator(op, device, basis, :cache)
    result .= 0
    result .+= operator(Operators.UNCOUPLED, device, basis)
    result .+= operator(Operators.COUPLING, device, basis)
    return result
end

@memoize Dict function operator(
    op::Operators.Static,
    device::DeviceType,
    basis::Bases.Dressed,
    ::Symbol,
)
    Λ, U = diagonalize(Bases.DRESSED, device)
    return Diagonal(Λ)
end

function operator(
    op::Operators.Static,
    device::DeviceType,
    ::Bases.Dressed;
    result=nothing,
)
    isnothing(result) && return operator(op, device, Bases.DRESSED, :cache)
    result .= operator(op, device, Bases.DRESSED, :cache)
    return result
end

function operator(
    op::Operators.Drive,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    if isnothing(result)
        N = nstates(device)
        result = Matrix{eltype(op,device,basis)}(undef, N, N)
    end
    result .= 0
    intermediate = array(eltype(result), size(result), (LABEL, :intermediate))

    for i in 1:ndrives(device)
        intermediate = operator(
            Operators.Channel(i, op.t), device, basis;
            result=intermediate,
        )
        result .+= intermediate
    end
    return result
end

function operator(
    op::Operators.Hamiltonian,
    device::DeviceType,
    basis::Bases.BasisType;
    result=nothing,
)
    if isnothing(result)
        N = nstates(device)
        result = Matrix{eltype(op,device,basis)}(undef, N, N)
    end
    result = operator(Operators.Drive(op.t), device, basis; result=result)
    result .+= operator(Operators.STATIC, device, basis)
    return result
end

"""
    localqubitoperators(device[, basis]; kwargs...)

A matrix list `h̄`, where each `h̄[:,:,q]` represents a local qubit hamiltonian.

# Arguments
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operators will be represented in.
        Defaults to `Bases.OCCUPATION` when omitted.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    localqubitoperators(device, basis, :cache)

For internal use only.
The extra positional argument enables dispatch to a cached function when appropriate.

"""
function localqubitoperators(device::DeviceType; kwargs...)
    return localqubitoperators(device, Bases.OCCUPATION; kwargs...)
end

function localqubitoperators(
    device::DeviceType,
    basis::Bases.LocalBasis;
    result=nothing,
)
    isnothing(result) && return localqubitoperators(device, basis, :cache)

    ā = localalgebra(device, basis)
    for q in 1:nqubits(device)
        result[:,:,q] .= qubithamiltonian(device, ā, q)
    end
    return result
end

@memoize Dict function localqubitoperators(
    device::DeviceType,
    basis::Bases.LocalBasis,
    ::Symbol,
)
    F = eltype(Operators.UNCOUPLED, device, basis)
    m = nlevels(device)
    n = nqubits(device)
    result = Array{F,3}(undef, m, m, n)
    return localqubitoperators(device, basis; result=result)
end

"""
    propagator(op, device[, basis], τ; kwargs...)

A unitary propagator describing evolution under a Hermitian operator for a small time τ.

# Arguments
- `op::Operators.OperatorType`: which operator to evolve under (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operator will be represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `τ::Real`: the amount to move forward in time by.
        Note that the propagation is only approximate for time-dependent operators.
        The smaller `τ` is, the more accurate the approximation.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    propagator(op, device, basis, τ, :cache)

For internal use only.
The extra positional argument enables dispatch to a cached function when appropriate.

"""
function propagator(op::Operators.OperatorType, device::DeviceType, τ::Real; kwargs...)
    return propagator(op, device, Bases.OCCUPATION, τ; kwargs...)
end

@memoize Dict function propagator(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real,
    ::Symbol,
)
    N = nstates(device)
    F = LinearAlgebraTools.cis_type(eltype(op,device,basis))
    result = Matrix{F}(undef, N, N)
    return propagator(op, device, basis, τ; result=result)
end

@memoize Dict function propagator(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real,
    ::Symbol,
)
    # NOTE: Select type independent of Identity, which is non-descriptive Bool.
    N = nstates(device)
    F = eltype_staticcoupling(device)
    result = Matrix{LinearAlgebraTools.cis_type(F)}(undef, N, N)
    return propagator(op, device, basis, τ; result=result)
end

function propagator(
    op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real;
    result=nothing,
)
    N = nstates(device)
    H = array(eltype(op, device, basis), (N,N), LABEL)
    H = operator(op, device, basis; result=H)

    isnothing(result) && (result=Matrix{LinearAlgebraTools.cis_type(H)}(undef, size(H)))
    result .= H
    return LinearAlgebraTools.cis!(result, -τ)
end

function propagator(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real;
    result=nothing,
)
    isnothing(result) && return propagator(op, device, basis, τ, :cache)
    result .= operator(op, device, basis, :cache)
    return LinearAlgebraTools.cis!(result, -τ)
end

function propagator(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real;
    result=nothing,
)
    isnothing(result) && return propagator(op, device, basis, τ, :cache)
    result .= operator(op, device, basis, :cache)
    result .*= exp(-im*τ)   # Include global phase.
    return result
end

function propagator(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.LocalBasis,
    τ::Real;
    result=nothing,
)
    isnothing(result) && return propagator(op, device, basis, τ, :cache)
    F = LinearAlgebraTools.cis_type(eltype(op, device, basis))

    m = nlevels(device)
    n = nqubits(device)
    ū = array(F, (m,m,n), LABEL)
    ū = localqubitpropagators(device, basis, τ; result=ū)
    return LinearAlgebraTools.kron(ū; result=result)
end

function propagator(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.LocalBasis,
    τ::Real;
    result=nothing,
)
    isnothing(result) && return propagator(op, device, basis, τ, :cache)
    ā = localalgebra(device, basis)

    h = qubithamiltonian(device, ā, op.q)

    u = Matrix{LinearAlgebraTools.cis_type(h)}(undef, size(h))
    u .= h
    u = LinearAlgebraTools.cis!(u, -τ)
    return globalize(device, u, op.q; result=result)
end

"""
    localqubitpropagators(device[, basis], τ; kwargs...)

A matrix list `ū`, where each `ū[:,:,q]` is a propagator for a local qubit hamiltonian.

# Arguments
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operator will be represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `τ::Real`: the amount to move forward in time by.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

    localqubitpropagators(device, basis, τ, :cache)

For internal use only.
The extra positional argument enables dispatch to a cached function when appropriate.

"""
function localqubitpropagators(device::DeviceType, τ::Real; kwargs...)
    return localqubitpropagators(device, Bases.OCCUPATION, τ; kwargs...)
end

function localqubitpropagators(
    device::DeviceType,
    basis::Bases.LocalBasis,
    τ::Real;
    result=nothing,
)
    isnothing(result) && return localqubitpropagators(device, basis, τ, :cache)

    result = localqubitoperators(device, basis; result=result)
    for q in 1:nqubits(device)
        LinearAlgebraTools.cis!(@view(result[:,:,q]), -τ)
    end
    return result
end

@memoize Dict function localqubitpropagators(
    device::DeviceType,
    basis::Bases.LocalBasis,
    τ::Real,
    ::Symbol,
)
    F = LinearAlgebraTools.cis_type(eltype(Operators.UNCOUPLED, device, basis))
    m = nlevels(device)
    n = nqubits(device)
    result = Array{F,3}(undef, m, m, n)
    return localqubitpropagators(device, basis, τ, result=result)
end

"""
    propagate!(op, device[, basis], τ, ψ)

Propagate a state `ψ` by a small time `τ` under the Hermitian `op` describing a `device`.

# Arguments
- `op::Operators.OperatorType`: which operator to evolve under (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the state `ψ` is represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `τ::Real`: the amount to move forward in time by.
        Note that the propagation is only approximate for time-dependent operators.
        The smaller `τ` is, the more accurate the approximation.
- `ψ`: Either a vector or a matrix, defined over the full Hilbert space of the device.

"""
function propagate!(
    op::Operators.OperatorType, device::DeviceType, τ::Real, ψ::Evolvable;
    kwargs...
)
    return propagate!(op, device, Bases.OCCUPATION, τ, ψ; kwargs...)
end

function propagate!(
    op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real,
    ψ::Evolvable,
)
    N = nstates(device)
    U = array(LinearAlgebraTools.cis_type(eltype(op, device, basis)), (N,N), LABEL)
    U = propagator(op, device, basis, τ; result=U)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function propagate!(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real,
    ψ::Evolvable,
)
    U = propagator(op, device, basis, τ, :cache)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function propagate!(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real,
    ψ::Evolvable,
)
    isa(ψ, AbstractMatrix) && return ψ  # CONJUGATION CANCELS ACCUMULATED PHASE

    ψ .*= exp(-im*τ)   # Include global phase.
    return ψ
end

function propagate!(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.LocalBasis,
    τ::Real,
    ψ::Evolvable,
)
    F = LinearAlgebraTools.cis_type(eltype(op, device, basis))
    m = nlevels(device)
    n = nqubits(device)
    ū = array(F, (m,m,n), LABEL)
    ū = localqubitpropagators(device, basis, τ; result=ū)
    ū = localqubitpropagators(device, basis, τ)
    return LinearAlgebraTools.rotate!(ū, ψ)
end

function propagate!(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.LocalBasis,
    τ::Real,
    ψ::Evolvable,
)
    F = LinearAlgebraTools.cis_type(eltype(op, device, basis))
    ā = localalgebra(device, basis)

    m = nlevels(device)
    n = nqubits(device)
    ops = array(F, (m,m,n), LABEL)
    for p in 1:n
        if p == op.q
            qubithamiltonian(device, ā, op.q; result=@view(ops[:,:,p]))
            LinearAlgebraTools.cis!(@view(ops[:,:,p]), -τ)
        else
            ops[:,:,p] .= Matrix(I, m, m)
        end
    end
    return LinearAlgebraTools.rotate!(ops, ψ)
end

"""
    evolver(op, device[, basis], t; kwargs...)

A unitary propagator describing evolution under a Hermitian operator for an time t.

This function is identical to `propagator`,
    except that the argument `t` is considered an absolute time so it is never cached,
    and that it is undefined for time-dependent operators.
It exists solely to perform rotating-frame rotations at every time-step
    without worrying about over-caching.

# Arguments
- `op::Operators.OperatorType`: which operator to evolve under (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operator will be represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `t::Real`: the amount to move forward in time by.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

"""
function evolver(op::Operators.OperatorType, device::DeviceType, t::Real; kwargs...)
    return evolver(op, device, Bases.OCCUPATION, t; kwargs...)
end

function evolver(
    op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    t::Real;
    result=nothing
)
    error("Not implemented for non-static operator.")
end

function evolver(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    t::Real;
    result=nothing
)
    H = operator(op, device, basis, :cache)
    isnothing(result) && (result=Matrix{LinearAlgebraTools.cis_type(H)}(undef, size(H)))
    result .= H
    return LinearAlgebraTools.cis!(result, -t)
end

function evolver(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType,
    t::Real;
    result=nothing,
)
    # NOTE: Select type independent of Identity, which is non-descriptive Bool.
    F = eltype_staticcoupling(device)
    Im = operator(op, device, basis, :cache)
    isnothing(result) && (result=Matrix{LinearAlgebraTools.cis_type(F)}(undef, size(Im)))
    result .= Im
    result .*= exp(-im*t)   # Include global phase.
    return result
end

function evolver(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.LocalBasis,
    t::Real;
    result=nothing
)
    F = LinearAlgebraTools.cis_type(eltype(op, device, basis))
    m = nlevels(device)
    n = nqubits(device)
    ū = array(F, (m,m,n), LABEL)
    ū = localqubitevolvers(device, basis, t; result=ū)
    return LinearAlgebraTools.kron(ū; result=result)
end

function evolver(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.LocalBasis,
    t::Real;
    result=nothing
)
    ā = localalgebra(device, basis)
    h = qubithamiltonian(device, ā, op.q)

    u = Matrix{LinearAlgebraTools.cis_type(h)}(undef, size(h))
    u .= h
    u = LinearAlgebraTools.cis!(u, -t)
    return globalize(device, u, op.q; result=result)
end

"""
    localqubitevolvers(device[, basis], τ; kwargs...)

A matrix list `ū`, where each `ū[:,:,q]` is a propagator for a local qubit hamiltonian.

This function is identical to `localqubitevolvers`,
    except that the argument `t` is considered an absolute time so it is never cached.

# Arguments
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operator will be represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `τ::Real`: the amount to move forward in time by.
        Note that the propagation is only approximate for time-dependent operators.
        The smaller `τ` is, the more accurate the approximation.

# Keyword Arguments
- `result`: a pre-allocated array of compatible type and shape, used to store the result.

"""
function localqubitevolvers(device::DeviceType, t::Real; kwargs...)
    return localqubitevolvers(device, Bases.OCCUPATION, t; kwargs...)
end

function localqubitevolvers(
    device::DeviceType,
    basis::Bases.LocalBasis,
    t::Real;
    result=nothing
)
    F = LinearAlgebraTools.cis_type(eltype(Operators.UNCOUPLED, device, basis))

    m = nlevels(device)
    n = nqubits(device)
    isnothing(result) && (result = Array{F,3}(undef, m, m, n))
    result = localqubitoperators(device, basis; result=result)
    for q in 1:n
        LinearAlgebraTools.cis!(@view(result[:,:,q]), -t)
    end
    return result
end

"""
    evolve!(op, device[, basis], t, ψ)

Propagate a state `ψ` by a time `t` under the Hermitian `op` describing a `device`.

This function is identical to `propagate!`,
    except that the cache is not used for intermediate propagator matrices,
    and that it is undefined for time-dependent operators.
Look to the `Evolutions` module for algorithms compatible with time-dependence!

# Arguments
- `op::Operators.OperatorType`: which operator to evolve under (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the state `ψ` is represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `t::Real`: the amount to move forward in time by.
- `ψ`: Either a vector or a matrix, defined over the full Hilbert space of the device.

"""
function evolve!(op::Operators.OperatorType, device::DeviceType, t::Real, ψ::Evolvable)
    return evolve!(op, device, Bases.OCCUPATION, t, ψ)
end

function evolve!(op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    t::Real,
    ψ::Evolvable,
)
    error("Not implemented for non-static operator.")
end

function evolve!(
    op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    t::Real,
    ψ::Evolvable,
)
    N = nstates(device)
    U = array(LinearAlgebraTools.cis_type(eltype(op, device, basis)), (N,N), LABEL)
    U = evolver(op, device, basis, t; result=U)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function evolve!(
    op::Operators.Identity,
    device::DeviceType,
    basis::Bases.BasisType,
    τ::Real,
    ψ::Evolvable,
)
    isa(ψ, AbstractMatrix) && return ψ  # CONJUGATION CANCELS ACCUMULATED PHASE

    ψ .*= exp(-im*τ)   # Include global phase.
    return ψ
end

function evolve!(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.LocalBasis,
    t::Real,
    ψ::Evolvable,
)
    F = LinearAlgebraTools.cis_type(eltype(op, device, basis))
    m = nlevels(device)
    n = nqubits(device)
    ū = array(F, (m,m,n), LABEL)
    ū = localqubitevolvers(device, basis, t; result=ū)
    return LinearAlgebraTools.rotate!(ū, ψ)
end

function evolve!(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.LocalBasis,
    t::Real,
    ψ::Evolvable,
)
    F = LinearAlgebraTools.cis_type(eltype(op, device, basis))
    ā = localalgebra(device, basis)

    m = nlevels(device)
    n = nqubits(device)
    ops = array(F, (m,m,n), LABEL)
    for p in 1:n
        if p == op.q
            qubithamiltonian(device, ā, op.q; result=@view(ops[:,:,p]))
            LinearAlgebraTools.cis!(@view(ops[:,:,p]), -t)
        else
            ops[:,:,p] .= Matrix(I, m, m)
        end
    end
    return LinearAlgebraTools.rotate!(ops, ψ)
end

"""
    expectation(op, device[, basis], ψ)

The expectation value of an operator describing a `device` with respect to the state `ψ`.

If ``A`` is the operator specified by `op`, this method calculates ``⟨ψ|A|ψ⟩``.

# Arguments
- `op::Operators.OperatorType`: which operator to estimate (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the state `ψ` is represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `ψ`: A statevector defined over the full Hilbert space of the device.

"""
function expectation(op::Operators.OperatorType, device::DeviceType, ψ::AbstractVector)
    return expectation(op, device, Bases.OCCUPATION, ψ)
end

function expectation(
    op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    ψ::AbstractVector,
)
    return braket(op, device, basis, ψ, ψ)
end

"""
    braket(op, device[, basis], ψ1, ψ2)

The braket of an operator describing a `device` with respect to states `ψ1` and `ψ2`.

If ``A`` is the operator specified by `op`, this method calculates ``⟨ψ1|A|ψ2⟩``.

# Arguments
- `op::Operators.OperatorType`: which operator to estimate (eg. static, drive, etc.).
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the state `ψ` is represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `ψ1`, `ψ2`: Statevectors defined over the full Hilbert space of the device.

"""
function braket(
    op::Operators.OperatorType,
    device::DeviceType,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    return braket(op, device, Bases.OCCUPATION, ψ1, ψ2)
end

function braket(op::Operators.OperatorType,
    device::DeviceType,
    basis::Bases.BasisType,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    N = nstates(device)
    H = array(eltype(op, device, basis), (N,N), LABEL)
    H = operator(op, device, basis; result=H)
    return LinearAlgebraTools.braket(ψ1, H, ψ2)
end

function braket(op::Operators.StaticOperator,
    device::DeviceType,
    basis::Bases.BasisType,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    H = operator(op, device, basis, :cache)
    return LinearAlgebraTools.braket(ψ1, H, ψ2)
end

function braket(
    op::Operators.Uncoupled,
    device::DeviceType,
    basis::Bases.LocalBasis,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    return sum(
        braket(Operators.Qubit(q), device, basis, ψ1, ψ2) for q in 1:nqubits(device)
    )
end

function braket(
    op::Operators.Qubit,
    device::DeviceType,
    basis::Bases.LocalBasis,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    ā = localalgebra(device, basis)
    h = qubithamiltonian(device, ā, op.q)

    m = nlevels(device)
    n = nqubits(device)
    ops = array(eltype(h), (m,m,n), LABEL)
    for p in 1:n
        ops[:,:,p] .= p == op.q ? h : Matrix(I, m, m)
    end
    return LinearAlgebraTools.braket(ψ1, ops, ψ2)
end

##########################################################################################
#=  =#

"""
    detuningfrequency(device::DeviceType, i::Int, q::Int)

Calculate the difference between drive frequency and qubit resonance.

"""
function detuningfrequency(device::DeviceType, i::Int, q::Int)
    return drivefrequency(device, i) - resonancefrequency(device, q)
end

"""
    drivesignal(device::DeviceType, i::Int)

The time-dependent signal for a given pulse index.

"""
function drivesignal(device::DeviceType{F,FΩ}, i::Int) where {F,FΩ}
    return __get__drivesignals(device)[i]
end

"""
    set_drivesignal(device::DeviceType, i::Int, signal::SignalType)

Change the time-dependent signal for a given pulse index.

"""
function set_drivesignal(
    device::DeviceType{F,FΩ},
    i::Int,
    signal::SignalType{F,FΩ},
) where {F,FΩ}
    __get__drivesignals(device)[i] = signal
end





"""
    LocallyDrivenDevice

Super-type for device objects whose drive channels act locally on individual qubits.

Inherit from this type if your `driveoperator` and `gradeoperator` methods
    depend only on a single annihilation operator `ā[:,:,q]`.
This enables more efficient propagation methods which exploit a tensor product structure.

# Implementation

Any concrete sub-type `D` must implement
    *everything* required in the `DeviceType` interface,
    so consult the documentation for `DeviceType` carefully.

In addition, the following methods must be implemented:
- `drivequbit(::D, i::Int)`: index of the qubit on which channel `i` is applied.
- `gradequbit(::D, j::Int)`: index of the qubit associated with the jth gradient operator.

It's usually trivial to infer the channel index i associated with each gradient operator,
    in which case `gradequbit(device, j) = drivequbit(device, i)`,
    but this is left as an implementation detail.

"""
abstract type LocallyDrivenDevice{F,FΩ} <: DeviceType{F,FΩ} end

"""
    drivequbit(device, i::Int)

Index of the qubit on which channel `i` is applied.

"""
function drivequbit(::LocallyDrivenDevice, i::Int)
    error("Not Implemented")
    return 0
end

"""
    gradequbit(device, j::Int)

Index of the qubit associated with the jth gradient operator.

"""
function gradequbit(::LocallyDrivenDevice, j::Int)
    error("Not Implemented")
    return 0
end

"""
    localdriveoperators(device[, basis], t; kwargs...)

A matrix list `v̄`, where `v̄[:,:,q]` represents a sum of all drives acting on qubit `q`.

# Arguments
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operators will be represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `t::Real`: the time each drive operator is evaluated at.

"""
function localdriveoperators(device::LocallyDrivenDevice, t::Real; kwargs...)
    return localdriveoperators(device, Bases.OCCUPATION, t; kwargs...)
end

function localdriveoperators(
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    t::Real;
    result=nothing,
)
    F = eltype(Operators.Drive(t), device, basis)
    ā = localalgebra(device, basis)

    m = nlevels(device)
    n = nqubits(device)
    isnothing(result) && (result = Array{F,3}(undef, m, m, n))
    result .= 0
    for i in 1:n
        q = drivequbit(device, i)
        result[:,:,q] .+= driveoperator(device, ā, i, t)
    end
    return result
end

"""
    localdrivepropagators(device[, basis], τ, t; kwargs...)

A matrix list `ū`, where `ū[:,:,q]` is the propagator for a local drive term.

# Arguments
- `device::DeviceType`: which device is being described.
- `basis::Bases.BasisType`: which basis the operators will be represented in.
        Defaults to `Bases.OCCUPATION` when omitted.
- `τ::Real`: the amount to move forward in time by.
        Note that the propagation is only approximate for time-dependent operators.
        The smaller `τ` is, the more accurate the approximation.
- `t::Real`: the time each drive operator is evaluated at.

"""
function localdrivepropagators(device::LocallyDrivenDevice, τ::Real, t::Real; kwargs...)
    return localdrivepropagators(device, Bases.OCCUPATION, τ, t; kwargs...)
end

function localdrivepropagators(
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    τ::Real,
    t::Real;
    result=nothing,
)
    F = eltype(Operators.Drive(t), device, basis)

    m = nlevels(device)
    n = nqubits(device)
    isnothing(result) && (result = Array{F,3}(undef, m, m, n))
    result = localdriveoperators(device, basis, t; result=result)
    for q in 1:n
        LinearAlgebraTools.cis!(@view(result[:,:,q]), -τ)
    end
    return result
end

function propagator(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    τ::Real;
    result=nothing,
)
    F = LinearAlgebraTools.cis_type(eltype(op, device, basis))
    m = nlevels(device)
    n = nqubits(device)
    ū = array(F, (m,m,n), LABEL)
    ū = localdrivepropagators(device, basis, τ, op.t; result=ū)
    return LinearAlgebraTools.kron(ū; result=result)
end

function propagator(
    op::Operators.Channel,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    τ::Real;
    result=nothing,
)
    F = LinearAlgebraTools.cis_type(eltype(op, device, basis))
    ā = localalgebra(device, basis)
    q = drivequbit(device, op.i)

    m = nlevels(device)
    u = array(F, (m, m), LABEL)
    u .= driveoperator(device, ā, op.i, op.t)
    u = LinearAlgebraTools.cis!(u, -τ)
    return globalize(device, u, q; result=result)
end

function propagate!(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    τ::Real,
    ψ::Evolvable,
)
    F = LinearAlgebraTools.cis_type(eltype(op, device, basis))
    m = nlevels(device)
    n = nqubits(device)
    ū = array(F, (m,m,n), LABEL)
    ū = localdrivepropagators(device, basis, τ, op.t; result=ū)
    return LinearAlgebraTools.rotate!(ū, ψ)
end

function propagate!(
    op::Operators.Channel,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    τ::Real,
    ψ::Evolvable,
)
    F = LinearAlgebraTools.cis_type(eltype(op, device, basis))
    ā = localalgebra(device, basis)
    q = drivequbit(device, op.i)

    m = nlevels(device)
    n = nqubits(device)
    ops = array(F, (m,m,n), LABEL)
    for p in 1:n
        if p == q
            driveoperator(device, ā, op.i, op.t; result=@view(ops[:,:,p]))
            LinearAlgebraTools.cis!(@view(ops[:,:,p]), -τ)
        else
            ops[:,:,p] .= Matrix(I, m, m)
        end
    end
    return LinearAlgebraTools.rotate!(ops, ψ)
end

function braket(
    op::Operators.Drive,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    return sum(
        braket(
            Operators.Channel(i, op.t),
            device, basis, ψ1, ψ2
        ) for i in 1:ndrives(device)
    )
end

function braket(
    op::Operators.Channel,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    F = eltype(op, device, basis)
    ā = localalgebra(device, basis)
    q = drivequbit(device, op.i)

    m = nlevels(device)
    n = nqubits(device)
    ops = array(F, (m,m,n), LABEL)
    for p in 1:n
        if p == q
            driveoperator(device, ā, op.i, op.t; result=@view(ops[:,:,p]))
        else
            ops[:,:,p] .= Matrix(I, m, m)
        end
    end
    return LinearAlgebraTools.braket(ψ1, ops, ψ2)
end

function braket(
    op::Operators.Gradient,
    device::LocallyDrivenDevice,
    basis::Bases.LocalBasis,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    F = eltype(op, device, basis)
    ā = localalgebra(device, basis)
    q = gradequbit(device, op.j)

    m = nlevels(device)
    n = nqubits(device)
    ops = array(F, (m,m,n), LABEL)
    for p in 1:n
        if p == q
            gradeoperator(device, ā, op.j, op.t; result=@view(ops[:,:,p]))
        else
            ops[:,:,p] .= Matrix(I, m, m)
        end
    end
    return LinearAlgebraTools.braket(ψ1, ops, ψ2)
end