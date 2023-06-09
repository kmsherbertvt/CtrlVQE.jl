# TODO (hi): add exports
using Memoization: @memoize

import LinearAlgebra: I, Diagonal, Hermitian, Eigen, eigen
import ..Bases, ..Operators, ..LinearAlgebraTools

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

import ..LinearAlgebraTools: MatrixList
const Evolvable = AbstractVecOrMat{<:Complex{<:AbstractFloat}}


"""
    Device

Super-type for all device objects.

# Implementation

Any concrete sub-type `D` must implement all functions in the `Parameters` module.
- In particular, if any static operators in your device depend on variational parameters,
    you should consult the "Note on Caching" below.

In addition, all methods in the following sections must be implemented.
- Counting methods
- Algebra methods
- Operator methods
- Type methods
- Gradient methods

If your device's drive channels are all local,
    you should implement a `LocallyDrivenDevice`,
    which has a few extra requirements.

## Counting methods (each returns an integer):

- `nqubits(::D)`: the number of qubits in the device - call this `n`.
- `nlevels(::D)`: the number of physical levels in each "qubit" - call this `m`.
- `ndrives(::D)`: the number of distinct drive channels.
- `ngrades(::D)`: the number of distinct gradient operators.

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
    for this module's `algebra` and `localalgebra` functions.

## Type methods:

- `eltype_localloweringoperator(::D)`
- `eltype_qubithamiltonian(::D)`
- `eltype_staticcoupling(::D)`
- `eltype_driveoperator(::D)`
- `eltype_gradeoperator(::D)`

Each of these methods gives the number type of the corresponding operator.
Implement these methods based only on your implementation of the methods,
    ie. they should be independent of the type of `ā` (or assume `ā` has type Bool).

## Gradient methods:

- `gradient(::D, τ̄, t̄, ϕ̄)`:
        the gradient vector for each variational parameter in the device.
        Each partial is generally an integral over at least one gradient signal:
        The arguments `τ̄` and `t̄` are time spacings and time grids,
            as given by `Evolutions.trapezoidaltimegrid`.
        The argument `ϕ̄` is a 2d array; `ϕ̄[:,:,j]` contains the jth gradient signal
            ``ϕ_j(t)`` evaluated at each point in `t̄`.

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
abstract type Device end

"""
    nqubits(device)

The number of qubits in the device.

"""
function nqubits(::Device)
    error("Not Implemented")
    return 0
end

"""
    nlevels(device)

The number of physical levels in each "qubit".

"""
function nlevels(::Device)
    error("Not Implemented")
    return 0
end

"""
    ndrives(device)

The number of distinct drive channels.

"""
function ndrives(::Device)
    error("Not Implemented")
    return 0
end

"""
    ngrades(device)

The number of distinct gradient operators.

"""
function ngrades(::Device)
    error("Not Implemented")
    return 0
end

"""
    localloweringoperator(device::Device; result=nothing)

The lowering operator ``a`` acting on the Hilbert space of a single physical qubit.

Optionally, pass a pre-allocated array of compatible type and shape as `result`.

"""
function localloweringoperator(::Device; result=nothing)
    error("Not Implemented")
    return zeros(eltype_localloweringoperator(device), nlevels(device), nlevels(device))
end

# TODO
eltype_qubithamiltonian(::Device)::Type{<:Number} = error("Not Implemented")
function qubithamiltonian(::Device,
    ā::MatrixList,
    q::Int;
    result=nothing,
)::AbstractMatrix
    return error("Not Implemented")
end

eltype_staticcoupling(::Device)::Type{<:Number} = error("Not Implemented")
function staticcoupling(::Device,
    ā::MatrixList;
    result=nothing,
)::AbstractMatrix
    return error("Not Implemented")
end

eltype_driveoperator(::Device)::Type{<:Number} = error("Not Implemented")
function driveoperator(::Device,
    ā::MatrixList,
    i::Int,
    t::Real;
    result=nothing,
)::AbstractMatrix
    return error("Not Implemented")
end

eltype_gradeoperator(::Device)::Type{<:Number} = error("Not Implemented")
function gradeoperator(::Device,
    ā::MatrixList,
    j::Int,
    t::Real;
    result=nothing,
)::AbstractMatrix
    # Returns Hermitian Â such that ϕ = ⟨λ|(𝑖Â)|ψ⟩ + h.t.
    return error("Not Implemented")
end

function gradient(::Device,
    τ̄::AbstractVector,
    t̄::AbstractVector,
    ϕ̄::AbstractMatrix;
    result=nothing,
)::AbstractVector
    return error("Not Implemented")
end





"""
    eltype_localloweringoperator(device::Device)

The number type of a local lowering operator for this device.

"""
function eltype_localloweringoperator(::Device)
    error("Not Implemented")
    return Bool
end




# UTILITIES

@memoize Dict nstates(device::Device) = nlevels(device) ^ nqubits(device)

function globalize(
    device::Device, op::AbstractMatrix{F}, q::Int;
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








# BASIS ROTATIONS

@memoize Dict function diagonalize(::Bases.Dressed, device::Device)
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

@memoize Dict function diagonalize(basis::Bases.LocalBasis, device::Device)
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

@memoize Dict function diagonalize(::Bases.Occupation, device::Device, q::Int)
    F = eltype_localloweringoperator(device)
    m = nlevels(device)
    identity = Matrix{F}(I, m, m)
    return eigen(Hermitian(identity))
end

@memoize Dict function diagonalize(::Bases.Coordinate, device::Device, q::Int)
    a = localloweringoperator(device)
    Q = (a + a') / eltype(a)(√2)
    return eigen(Hermitian(Q))
end

@memoize Dict function diagonalize(::Bases.Momentum, device::Device, q::Int)
    a = localloweringoperator(device)
    P = im*(a - a') / eltype(a)(√2)
    return eigen(Hermitian(P))
end

@memoize Dict function basisrotation(
    tgt::Bases.BasisType,
    src::Bases.BasisType,
    device::Device,
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
    device::Device,
)
    ū = localbasisrotations(tgt, src, device)
    return LinearAlgebraTools.kron(ū)
end

@memoize Dict function basisrotation(
    tgt::Bases.LocalBasis,
    src::Bases.LocalBasis,
    device::Device,
    q::Int,
)
    Λ0, U0 = diagonalize(src, device, q)
    Λ1, U1 = diagonalize(tgt, device, q)
    # |ψ'⟩ ≡ U0|ψ⟩ rotates |ψ⟩ OUT of `src` Bases.
    # U1'|ψ'⟩ rotates |ψ'⟩ INTO `tgt` Bases.
    return U1' * U0
end

@memoize Dict function localbasisrotations(
    tgt::Bases.LocalBasis,
    src::Bases.LocalBasis,
    device::Device,
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





#= ALGEBRAS =#

function eltype_algebra(device::Device, ::Bases.BasisType=Bases.OCCUPATION)
    return eltype_localloweringoperator(device)
end

function eltype_algebra(device::Device, ::Bases.Dressed)
    return promote_type(
        eltype_localloweringoperator(device),
        eltype_qubithamiltonian(device),
        eltype_staticcoupling(device),
    )
end

@memoize Dict function algebra(
    device::Device,
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

@memoize Dict function localalgebra(
    device::Device,
    basis::Bases.BasisType=Bases.OCCUPATION,
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


#= TYPE FUNCTIONS =#

function Base.eltype(op::Operators.OperatorType, device::Device)
    return Base.eltype(op, device, Bases.OCCUPATION)
end

function Base.eltype(op::Operators.Identity, device::Device, basis::Bases.BasisType)
    return Bool
end

function Base.eltype(::Operators.Qubit, device::Device, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_qubithamiltonian(device),
    )
end

function Base.eltype(::Operators.Coupling, device::Device, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_staticcoupling(device),
    )
end

function Base.eltype(::Operators.Channel, device::Device, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_driveoperator(device),
    )
end

function Base.eltype(::Operators.Gradient, device::Device, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_gradeoperator(device),
    )
end

function Base.eltype(::Operators.Uncoupled, device::Device, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_qubithamiltonian(device),
    )
end

function Base.eltype(::Operators.Static, device::Device, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_qubithamiltonian(device),
        eltype_staticcoupling(device),
    )
end

function Base.eltype(::Operators.Drive, device::Device, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_driveoperator(device),
    )
end

function Base.eltype(::Operators.Hamiltonian, device::Device, basis::Bases.BasisType)
    return promote_type(
        eltype_algebra(device, basis),
        eltype_qubithamiltonian(device),
        eltype_staticcoupling(device),
        eltype_driveoperator(device),
    )
end

#= HERMITIAN OPERATORS =#

function operator(op::Operators.OperatorType, device::Device; kwargs...)
    return operator(op, device, Bases.OCCUPATION; kwargs...)
end

@memoize Dict function operator(
    op::Operators.StaticOperator,
    device::Device,
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
    device::Device,
    basis::Bases.BasisType,
    ::Symbol,
)
    return Diagonal(ones(Bool, nstates(device)))
end

function operator(
    op::Operators.Identity,
    device::Device,
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
    device::Device,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return operator(op, device, basis, :cache)
    ā = algebra(device, basis)
    return qubithamiltonian(device, ā, op.q; result=result)
end

function operator(
    op::Operators.Coupling,
    device::Device,
    basis::Bases.BasisType;
    result=nothing,
)
    isnothing(result) && return operator(op, device, basis, :cache)
    ā = algebra(device, basis)
    return staticcoupling(device, ā; result=result)
end

function operator(
    op::Operators.Channel,
    device::Device,
    basis::Bases.BasisType;
    result=nothing,
)
    ā = algebra(device, basis)
    return driveoperator(device, ā, op.i, op.t; result=result)
end

function operator(
    op::Operators.Gradient,
    device::Device,
    basis::Bases.BasisType;
    result=nothing,
)
    ā = algebra(device, basis)
    return gradeoperator(device, ā, op.j, op.t; result=result)
end

function operator(
    op::Operators.Uncoupled,
    device::Device,
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
    device::Device,
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
    device::Device,
    basis::Bases.Dressed,
    ::Symbol,
)
    Λ, U = diagonalize(Bases.DRESSED, device)
    return Diagonal(Λ)
end

function operator(
    op::Operators.Static,
    device::Device,
    ::Bases.Dressed;
    result=nothing,
)
    isnothing(result) && return operator(op, device, Bases.DRESSED, :cache)
    result .= operator(op, device, Bases.DRESSED, :cache)
    return result
end

function operator(
    op::Operators.Drive,
    device::Device,
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
    device::Device,
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


function localqubitoperators(device::Device; kwargs...)
    return localqubitoperators(device, Bases.OCCUPATION; kwargs...)
end

function localqubitoperators(
    device::Device,
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
    device::Device,
    basis::Bases.LocalBasis,
    ::Symbol,
)
    F = eltype(Operators.UNCOUPLED, device, basis)
    m = nlevels(device)
    n = nqubits(device)
    result = Array{F,3}(undef, m, m, n)
    return localqubitoperators(device, basis; result=result)
end




#= PROPAGATORS =#



function propagator(op::Operators.OperatorType, device::Device, τ::Real; kwargs...)
    return propagator(op, device, Bases.OCCUPATION, τ; kwargs...)
end

@memoize Dict function propagator(
    op::Operators.StaticOperator,
    device::Device,
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
    device::Device,
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
    device::Device,
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
    device::Device,
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
    device::Device,
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
    device::Device,
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
    device::Device,
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




function localqubitpropagators(device::Device, τ::Real; kwargs...)
    return localqubitpropagators(device, Bases.OCCUPATION, τ; kwargs...)
end

function localqubitpropagators(
    device::Device,
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
    device::Device,
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



#= MUTATING PROPAGATION =#

function propagate!(
    op::Operators.OperatorType, device::Device, τ::Real, ψ::Evolvable;
    kwargs...
)
    return propagate!(op, device, Bases.OCCUPATION, τ, ψ; kwargs...)
end

function propagate!(
    op::Operators.OperatorType,
    device::Device,
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
    device::Device,
    basis::Bases.BasisType,
    τ::Real,
    ψ::Evolvable,
)
    U = propagator(op, device, basis, τ, :cache)
    return LinearAlgebraTools.rotate!(U, ψ)
end

function propagate!(
    op::Operators.Identity,
    device::Device,
    basis::Bases.BasisType,
    τ::Real,
    ψ::Evolvable,
)
    ψ .*= exp(-im*τ)   # Include global phase.
    return ψ
end

function propagate!(
    op::Operators.Uncoupled,
    device::Device,
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
    device::Device,
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



#= PROPAGATORS FOR ARBITRARY TIME (static only) =#


function evolver(op::Operators.OperatorType, device::Device, t::Real; kwargs...)
    return evolver(op, device, Bases.OCCUPATION, t; kwargs...)
end

function evolver(
    op::Operators.OperatorType,
    device::Device,
    basis::Bases.BasisType,
    t::Real;
    result=nothing
)
    error("Not implemented for non-static operator.")
end

function evolver(
    op::Operators.StaticOperator,
    device::Device,
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
    device::Device,
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
    device::Device,
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
    device::Device,
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



function localqubitevolvers(device::Device, t::Real; kwargs...)
    return localqubitevolvers(device, Bases.OCCUPATION, t; kwargs...)
end

function localqubitevolvers(
    device::Device,
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



#= MUTATING EVOLUTION FOR ARBITRARY TIME (static only) =#

function evolve!(op::Operators.OperatorType, device::Device, t::Real, ψ::Evolvable)
    return evolve!(op, device, Bases.OCCUPATION, t, ψ)
end

function evolve!(op::Operators.OperatorType,
    device::Device,
    basis::Bases.BasisType,
    t::Real,
    ψ::Evolvable,
)
    error("Not implemented for non-static operator.")
end

function evolve!(
    op::Operators.StaticOperator,
    device::Device,
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
    device::Device,
    basis::Bases.BasisType,
    τ::Real,
    ψ::Evolvable,
)
    ψ .*= exp(-im*τ)   # Include global phase.
    return ψ
end

function evolve!(
    op::Operators.Uncoupled,
    device::Device,
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
    device::Device,
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





#= SCALAR MATRIX OPERATIONS =#

function expectation(op::Operators.OperatorType, device::Device, ψ::AbstractVector)
    return expectation(op, device, Bases.OCCUPATION, ψ)
end

function expectation(
    op::Operators.OperatorType,
    device::Device,
    basis::Bases.BasisType,
    ψ::AbstractVector,
)
    return braket(op, device, basis, ψ, ψ)
end

function braket(
    op::Operators.OperatorType,
    device::Device,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    return braket(op, device, Bases.OCCUPATION, ψ1, ψ2)
end

function braket(op::Operators.OperatorType,
    device::Device,
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
    device::Device,
    basis::Bases.BasisType,
    ψ1::AbstractVector,
    ψ2::AbstractVector,
)
    H = operator(op, device, basis, :cache)
    return LinearAlgebraTools.braket(ψ1, H, ψ2)
end

function braket(
    op::Operators.Uncoupled,
    device::Device,
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
    device::Device,
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












