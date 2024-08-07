
"""
    DeviceType{F}

Super-type for all device objects.

# Type Parameters
- `F`: the float type associated with a device

# Implementation

Any concrete sub-type `D` must implement all functions in the `Parameters` module.
- In particular, if any static operators in your device depend on variational parameters,
    you should consult the "Note on Caching" below.

In addition, all methods in the following sections must be implemented.
- Counting methods
- Algebra methods
- Operator methods
- Gradient methods

If your device's drive channels are all local (acting on one qubit at a time),
    you should implement a `LocallyDrivenDevice`,
    which has a few extra requirements.

## Counting methods:

- `nqubits(::D)`: the number of qubits in the device - call this `n`.
- `nlevels(::D)`: the number of physical levels in each "qubit" - call this `m`.
- `ndrives(::D)`: the number of distinct drive channels.
- `ngrades(::D)`: the number of distinct gradient operators.
- `noperators(::D)`: the number of operators used to define an algebra for each "qubit".

Each of these methods returns an integer.

## Algebra methods:

- `localalgebra(::D)`: a 4d array ā
    where ā[:,:,σ,q] is the σ'th algebraic operator on qubit q, in the bare basis.

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
When `result` is not passed, you should return a new array of type `Complex{eltype(D)}`.
Aside from this, do your best to minimize allocations.

Each of these methods takes a 4darray `ā`, indexed as described in `localalgebra`.
For example, for an algebra defined in terms of bosonic ladder operators,
    ``\\hat a_q`` might be accessed with `ā[:,:,1,q]`
    and ``\\hat a_q^\\dagger`` with `ā[:,:,1,q]'`.
For an algebra defined in terms of Pauli operators,
    ``X_q`` might be accessed with `ā[:,:,1,q]`,
    ``Y_q`` with `ā[:,:,2,q]`, and ``Z_q`` with `ā[:,:,3,q]`.

Usually, each `ā[:,:,σ,q]` is defined on the full Hilbert space (ie. `m^n × m^n`),
    but sometimes the code exploits a simple tensor structure
    by passing in local `m × m` operators instead,
    so do not assume a specific size a priori.
Do NOT modify these operators, as they are usually drawn from a cache.

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

Usually, variational parameters only affect time-dependent methods,
    but if any of your device's static operators do depend on a variational parameter,
    you should be careful to empty the cache when `Parameters.bind!` is called.

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
abstract type DeviceType{F} end


"""
    ndrives(device::DeviceType)

The number of distinct drive channels.

"""
function ndrives(::DeviceType)
    error("Not Implemented")
    return 0
end

"""
    ngrades(device::DeviceType)

The number of distinct gradient operators.

"""
function ngrades(::DeviceType)
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
    nqubits(device::DeviceType)

The number of qubits in the device.

"""
function nqubits(::DeviceType)
    error("Not Implemented")
    return 0
end

"""
    ngenerators(device::DeviceType)

The number of operators used to define an algebra for each "qubit".

"""
function noperators(::DeviceType)
    error("Not Implemented")
    return 0
end

"""
    localalgebra(::DeviceType; result=nothing)

Construct a 4d array ā representing all operators defining an algebra for all qubits.

The matrix ā[:,:,σ,q] is a local operator acting on the space of a single qubit
    (meaning it is an `m⨯m` matrix, if `m` is the result of `nlevels(device)`).

The array is stored in `result` or, if not provided, returned from a cache.
If `result` is not provided, the array is of type `Complex{eltype(device)}`.

# Implementation

To use the cache, simply include as the first line:

    isnothing(result) && return _localalgebra(device)

If there is nothing yet in the cache,
    the `_localalgebra` function will simply call your method again,
    but with an empty array passed as `result`.

"""
function localalgebra(::DeviceType; result=nothing)
    error("Not Implemented")
    return Array{Complex{eltype(device)}}(undef, 0, 0, 0)
end

@memoize Dict function _localalgebra(device::DeviceType)
    m = nlevels(device)
    n = nqubits(device)
    o = noperators(device)
    result = Array{Complex{eltype(device)}}(undef, m, m, o, n)
    return localalgebra(device; result=result)
end







"""
    qubithamiltonian(device::DeviceType, ā, q::Int; result=nothing)

The static components of the device Hamiltonian local to qubit `q`.

This method is a function of algebraic operators given by `ā[:,:,σ,q]`,
    constructed by either the `localalgebra` or `globalalgebra` methods.

The array is stored in `result` if provided.
If `result` is not provided, the array is of type `Complex{eltype(device)}`.

"""
function qubithamiltonian(::DeviceType, ā, q::Int; result=nothing)
    error("Not Implemented")
    return zeros(eltype(ā), size(ā)[1:2])
end




"""
    staticcoupling(device::DeviceType, ā, q::Int; result=nothing)

The static components of the device Hamiltonian nonlocal to any one qubit.

This method is a function of algebraic operators given by `ā[:,:,σ,q]`,
    constructed the `globalalgebra` method.

The array is stored in `result` if provided.
If `result` is not provided, the array is of type `Complex{eltype(device)}`.

"""
function staticcoupling(::DeviceType, ā; result=nothing)
    error("Not Implemented")
    return zeros(eltype(ā), size(ā)[1:2])
end




"""
    driveoperator(device::DeviceType, ā, i::Int, t::Real; result=nothing)

The distinct drive operator for channel `i` at time `t`.

This method is a function of algebraic operators given by `ā[:,:,σ,q]`,
    constructed the `globalalgebra` method.
If `device` is a `LocallyDrivenDevice`,
    `ā` may also have been constructed from the `localalgebra` method.

The array is stored in `result` if provided.
If `result` is not provided, the array is of type `Complex{eltype(device)}`.

"""
function driveoperator(::DeviceType, ā, i::Int, t::Real; result=nothing)
    error("Not Implemented")
    return zeros(eltype(ā), size(ā)[1:end-1])
end




"""
    gradeoperator(device::DeviceType, ā, j::Int, t::Real; result=nothing)

The distinct gradient operator indexed by `j` at time `t`.

I have defined the "gradient operator" ``Â_j`` as the Hermitian operator
    for which the jth gradient signal is ``ϕ_j = ⟨λ|(iÂ_j)|ψ⟩ + h.t.``.

This method is a function of algebraic operators given by `ā[:,:,σ,q]`,
    constructed the `globalalgebra` method.
If `device` is a `LocallyDrivenDevice`,
    `ā` may also have been constructed from the `localalgebra` method.

The array is stored in `result` if provided.
If `result` is not provided, the array is of type `Complex{eltype(device)}`.

"""
function gradeoperator(::DeviceType, ā, j::Int, t::Real; result=nothing)
    error("Not Implemented")
    return zeros(eltype(ā), size(ā)[1:end-1])
end




"""
    gradient(::DeviceType, grid::Integrations.IntegrationType, ϕ̄; result=nothing)

The gradient vector of partials for each variational parameter in the device.

Each partial is generally an integral over at least one gradient signal.
The argument `grid` identifies the temporal lattice on which ϕ̄ is defined.
The argument `ϕ̄` is a 2d array; `ϕ̄[:,j]` contains the jth gradient signal
    ``ϕ_j(t)`` evaluated at each point in `grid`.

The array is stored in `result` if provided.

"""
function gradient(device::DeviceType,
    grid::Integrations.IntegrationType,
    ϕ̄::AbstractMatrix;
    result=nothing,
)
    error("Not Implemented")
    return zero(Parameters.values(device))
end