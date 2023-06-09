import LinearAlgebra: norm
import ..Bases, ..LinearAlgebraTools, ..Devices
import ..Operators: STATIC, Drive, Gradient

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using ..LinearAlgebraTools: MatrixList
using Memoization: @memoize

"""
    trapezoidaltimegrid(T::Real, r::Int)

All the tools needed to integrate over time using a simple trapezoidal rule.

# Arguments
- `T`: the upper bound of the integral (0 is implicitly the lower bound).
        If `T` is negative, `|T|` is used as the lower bound and 0 as the upper bound.

- `r`: the number of time steps.
        Note this is the number of STEPS.
        The number of time POINTS is `r+1`, since they include t=0.

# Returns
- `τ`: the length of each time step, simply ``T/r``.
- `τ̄`: vector of `r+1` time spacings to use as `dt` in integration.
- `t̄`: vector of `r+1` time points

The integral ``∫_0^T f(t)⋅dt`` is evaluated with `sum(f.(t̄) .* τ̄)`.

# Explanation

Intuitively, `τ̄` is a vector giving `τ` for each time point.
But careful! The sum of all `τ̄` must match the length of the integral, ie. `T`.
But there are `r+1` points, and `(r+1)⋅τ > T`. How do we reconcile this?
A "Left Hand Sum" would omit `t=T` from the time points;
    a "Right Hand Sum" would omit `t=0`.
The trapezoidal rule omits half a point from each.

That sounds awfully strange, but it's mathematically sound!
We only integrate through *half* of each boundary time point `t=0` and `t=T`.
Thus, those points, and only those points, have a spacing of `τ/2`.

"""
@memoize Dict function trapezoidaltimegrid(T::Real, r::Int)
    # NOTE: Negative values of T give reversed time grid.
    τ = T / r
    τ̄ = fill(τ, r+1); τ̄[[begin, end]] ./= 2
    t̄ = abs(τ) * (T ≥ 0 ? (0:r) : reverse(0:r))
    return τ, τ̄, t̄
end


"""
    Algorithm

Super-type for all evolution algorithms.

# Implementation

Any concrete sub-type `A` must implement the following methods:
- `evolve!(::A, device, basis, T, ψ; callback=nothing)`
- `evolve!(::A, device, T, ψ; callback=nothing)`

The latter method should simply call the former,
    using the basis which renders the algorithm most efficient.
Please consult the documentation for `evolve!` for details on the implementation.

"""
abstract type Algorithm end

"""
    evolve!([algorithm, ]device, [basis, ]T, ψ; callback=nothing)

Evolve a state `ψ` by time `T` under a `device` Hamiltonian.

This method both mutates and returns `ψ`.

# Arguments
- `algorithm::Algorithm`: dispatches which evolution method to use.
        Defaults to `Rotate(1000)` if omitted.

- `device::Devices.Device`: specifies which Hamiltonian to evolve under.

- `basis::Bases.BasisType`: which basis `ψ` is represented in.
        ALSO determines the basis in which calculations are carried out.
        The default *depends on the algorithm*, so be sure to transform `ψ` accordingly.
        For `Rotate` (the default algorithm), the default basis is `Bases.OCCUPATION`.

- `T::Real`: the total amount of time to evolve by.
        The evolution is implicitly assumed to start at time `t=0`.

- `ψ`: the initial statevector, defined on the full Hilbert space of the device.

# Keyword Arguments
- `callback`: a function which is called at each iteration of the time evolution.
        The function is passed three arguments:
        - `i`: indexes the iteration
        - `t`: the current time point
        - `ψ`: the current statevector

"""
function evolve! end



"""
    evolve([algorithm, ]device, [basis, ]T, ψ0; result=nothing, kwargs...)

Evolve a state `ψ0` by time `T` under a `device` Hamiltonian, without mutating `ψ0`.

This method simply copies `ψ0` (to `result` if provided, or else to a new array),
    then calls the mutating function `evolve!` on the copy.
Please see `evolve!` for detailed documentation.

"""
function evolve end

function evolve(
    device::Devices.Device,
    T::Real,
    ψ0::AbstractVector;
    result=nothing,
    kwargs...
)
    F = LinearAlgebraTools.cis_type(ψ0)
    result === nothing && (result = Vector{F}(undef, length(ψ0)))
    result .= ψ0
    return evolve!(device, T, result; kwargs...)
end

function evolve(
    device::Devices.Device,
    basis::Bases.BasisType,
    T::Real,
    ψ0::AbstractVector;
    result=nothing,
    kwargs...
)
    F = LinearAlgebraTools.cis_type(ψ0)
    result === nothing && (result = Vector{F}(undef, length(ψ0)))
    result .= ψ0
    return evolve!(device, basis, T, result; kwargs...)
end

function evolve(
    algorithm::Algorithm,
    device::Devices.Device,
    T::Real,
    ψ0::AbstractVector;
    result=nothing,
    kwargs...
)
    F = LinearAlgebraTools.cis_type(ψ0)
    result === nothing && (result = Vector{F}(undef, length(ψ0)))
    result .= ψ0
    return evolve!(algorithm, device, T, result; kwargs...)
end

function evolve(
    algorithm::Algorithm,
    device::Devices.Device,
    basis::Bases.BasisType,
    T::Real,
    ψ0::AbstractVector;
    result=nothing,
    kwargs...
)
    F = LinearAlgebraTools.cis_type(ψ0)
    result === nothing && (result = Vector{F}(undef, length(ψ0)))
    result .= ψ0
    return evolve!(algorithm, device, basis, T, result; kwargs...)
end




"""
    Rotate(r)

A Trotterization method (using `r` steps) alternately propagating static and drive terms.

The default basis for this algorithm is `Bases.OCCUPATION`.

"""
struct Rotate <: Algorithm
    r::Int
end

function evolve!(args...; kwargs...)
    return evolve!(Rotate(1000), args...; kwargs...)
end

function evolve!(algorithm::Rotate, device::Devices.Device, args...; kwargs...)
    return evolve!(algorithm, device, Bases.OCCUPATION, args...; kwargs...)
end

function evolve!(
    algorithm::Rotate,
    device::Devices.Device,
    basis::Bases.BasisType,
    T::Real,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
    callback=nothing,
)
    r = algorithm.r
    τ, τ̄, t̄ = trapezoidaltimegrid(T, r)

    # REMEMBER NORM FOR NORM-PRESERVING STEP
    A = norm(ψ)

    # FIRST STEP: NO NEED TO APPLY STATIC OPERATOR
    callback !== nothing && callback(1, t̄[1], ψ)
    ψ = Devices.propagate!(Drive(t̄[1]),  device, basis, τ̄[1], ψ)

    # RUN EVOLUTION
    for i in 2:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        ψ = Devices.propagate!(STATIC, device, basis, τ, ψ)
        ψ = Devices.propagate!(Drive(t̄[i]),  device, basis, τ̄[i], ψ)
    end

    # ENFORCE NORM-PRESERVING TIME EVOLUTION
    ψ .*= A / norm(ψ)

    return ψ
end





"""
    Direct(r)

A Trotterization method (using `r` steps) calculating drive terms in the rotation-frame.

The default basis for this algorithm is `Bases.DRESSED`,
    since the rotating-frame evolution ``U_t ≡ exp(-itH_0)`` happens at each step.

This algorithm exponentiates the matrix ``U_t' V(t) U_t`` at each time step,
    so it is not terribly efficient.

"""
struct Direct <: Algorithm
    r::Int
end

function evolve!(algorithm::Direct, device::Devices.Device, args...; kwargs...)
    return evolve!(algorithm, device, Bases.DRESSED, args...; kwargs...)
end

function evolve!(
    algorithm::Direct,
    device::Devices.Device,
    basis::Bases.BasisType,
    T::Real,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
    callback=nothing,
)
    r = algorithm.r
    τ, τ̄, t̄ = trapezoidaltimegrid(T, r)

    # REMEMBER NORM FOR NORM-PRESERVING STEP
    A = norm(ψ)

    # # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
    # U = Devices.evolver(STATIC, device, basis, 0)
    # V = Devices.operator(Drive(0), device, basis)
    # # PROMOTE `V` SO THAT IT CAN BE ROTATED IN PLACE AND EXPONENTIATED
    # F = Complex{real(promote_type(eltype(U), eltype(V)))}
    # V = convert(Matrix{F}, copy(V))

    # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
    N = Devices.nstates(device)
    U_TYPE = LinearAlgebraTools.cis_type(eltype(STATIC, device, basis))
    V_TYPE = LinearAlgebraTools.cis_type(eltype(Drive(0), device, basis))
    U = array(U_TYPE, (N,N), (LABEL, :intermediate))
    V = array(V_TYPE, (N,N), LABEL)

    # RUN EVOLUTION
    for i in 1:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        U = Devices.evolver(STATIC, device, basis, t̄[i]; result=U)
        V = Devices.operator(Drive(t̄[i]), device, basis; result=V)
        V = LinearAlgebraTools.rotate!(U', V)
        V = LinearAlgebraTools.cis!(V, -τ̄[i])
        ψ = LinearAlgebraTools.rotate!(V, ψ)
    end

    # ROTATE OUT OF INTERACTION PICTURE
    ψ = Devices.evolve!(STATIC, device, basis, T, ψ)

    # ENFORCE NORM-PRESERVING TIME EVOLUTION
    ψ .*= A / norm(ψ)

    return ψ
end



"""
    gradientsignals(device[, basis], T, ψ0, r, O; kwargs...)

The gradient signals associated with a given `device` Hamiltonian, and an observable `O`.

Gradient signals are used to calculate analytical derivatives of a control pulse.

# Arguments
- `device::Devices.Device`: specifies which Hamiltonian to evolve under.
        Also identifies each of the gradient operators used to calculate gradient signals.

- `basis::Bases.BasisType`: which basis `ψ` is represented in.
        ALSO determines the basis in which calculations are carried out.
        Defaults to `Bases.OCCUPATION`.

- `T::Real`: the total duration of the pulse.

- `ψ0`: the initial statevector, defined on the full Hilbert space of the device.

- `r::Int`: the number of time-steps to evaluate ``ϕ_j(t)`` for.

- `O`: a Hermitian observable, represented as a matrix.
    Gradients are calculated with respect to the expectation `⟨O⟩` at time `T`.

# Keyword Arguments
- `result`: an (optional) pre-allocated array to store gradient signals

- `evolution`: the evolution algorithm used to initialize the co-state `|λ⟩`.
        The computation of the gradient signals always uses a `Rotate`-like algorithm,
            but it begins with a plain-old time evolution.
        This keyword argument controls how to do that initial time evolution only.
        It defaults to `Rotate(r)`.

- `callback`: a function called at each iteration of the gradient signal calculation.
        The function is passed three arguments:
        - `i`: indexes the iteration
        - `t`: the current time point
        - `ψ`: the current statevector

# Returns
A vector list `ϕ̄`, where each `ϕ̄[:,j]` is the gradient signal ``ϕ_j(t)``
    evaluated on a time grid given by `trapezoidaltimegrid(T,r)`.


# Explanation
A gradient signal ``ϕ_j(t)`` is defined with respect to a gradient operator ``Â_j``,
    an observable ``Ô``, a time-dependent state `|ψ(t)⟩`, and total pulse duration `T`.

Let us define the expectation value ``E(T) ≡ ⟨ψ(T)|Ô|ψ(T)⟩``.

Define the co-state ``|λ(t)⟩`` as the (un-normalized) statevector
    which satisfies ``E(T)=⟨λ(t)|ψ(t)⟩`` for any time `t∊[0,T]`.
The gradient signal is defined as ``ϕ_j(t) ≡ ⟨λ(t)|(iÂ_j)|ψ(t)⟩ + h.t.``.


    gradientsignals(device[, basis], T, ψ0, r, Ō; kwargs...)

When the matrix argument `O` is replaced by a matrix list `Ō`,
    each `Ō[:,:,k]` represents a different Hermitian observable ``Ô_k``.
In this case, a different set of gradient signals is computed for *each* ``Ô_k``.

# Returns
A 3d array `ϕ̄`, where each `ϕ̄[:,j,k]` is the gradient signal ``ϕ_j(t)``
    defined with respect to the observable ``Ô_k``.

# Explanation
Multiple sets of gradient signals may be useful
    if you want to compute gradients with respect to multiple observables.
For example, gradients with respect to a normalized molecular energy
    include contributions from both a molecular Hamiltonian and a leakage operator.
This method enables such calculations using only a single "pass" through time.

"""
function gradientsignals(device::Devices.Device, args...; kwargs...)
    return gradientsignals(device, Bases.OCCUPATION, args...; kwargs...)
end

function gradientsignals(
    device::Devices.Device,
    basis::Bases.BasisType,
    T::Real,
    ψ0::AbstractVector,
    r::Int,
    O::AbstractMatrix;
    result=nothing,
    kwargs...
)
    # `O` AND `result` GIVEN AS 2D ARRAYS BUT MUST BE 3D FOR DELEGATION
    result !== nothing && (result = reshape(result, size(result)..., 1))
    O = reshape(O, size(O)..., 1)

    # PERFORM THE DELEGATION
    result = gradientsignals(device, basis, T, ψ0, r, O; result=result, kwargs...)

    # NOW RESHAPE `result` BACK TO 2D ARRAY
    result = reshape(result, size(result, 1), size(result, 2))
    return result
end

function gradientsignals(
    device::Devices.Device,
    basis::Bases.BasisType,
    T::Real,
    ψ0::AbstractVector,
    r::Int,
    Ō::MatrixList;
    result=nothing,
    evolution=Rotate(r),
    callback=nothing,
)
    τ, τ̄, t̄ = trapezoidaltimegrid(T, r)

    # PREPARE SIGNAL ARRAYS ϕ̄[i,j,k]
    if result === nothing
        F = real(LinearAlgebraTools.cis_type(ψ0))
        result = Array{F}(undef, r+1, Devices.ngrades(device), length(Ō))
    end

    # PREPARE STATE AND CO-STATES
    ψTYPE = LinearAlgebraTools.cis_type(ψ0)
    ψ = array(ψTYPE, size(ψ0), LABEL); ψ .= ψ0
    ψ = evolve!(evolution, device, basis, T, ψ)

    λ̄ = array(ψTYPE, (size(ψ0,1), size(Ō,3)), LABEL)
    for k in axes(Ō,3)
        λ̄[:,k] .= ψ
        LinearAlgebraTools.rotate!(@view(Ō[:,:,k]), @view(λ̄[:,k]))
    end

    # LAST GRADIENT SIGNALS
    callback !== nothing && callback(r+1, t̄[r+1], ψ)
    for k in axes(Ō,3)
        λ = @view(λ̄[:,k])
        for j in 1:Devices.ngrades(device)
            z = Devices.braket(Gradient(j, t̄[end]), device, basis, λ, ψ)
            result[r+1,j,k] = 2 * imag(z)   # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
        end
    end

    # ITERATE OVER TIME
    for i in reverse(1:r)
        # COMPLETE THE PREVIOUS TIME-STEP AND START THE NEXT
        ψ = Devices.propagate!(Drive(t̄[i+1]), device, basis, -τ/2, ψ)
        ψ = Devices.propagate!(STATIC, device, basis, -τ, ψ)
        ψ = Devices.propagate!(Drive(t̄[i]),   device, basis, -τ/2, ψ)
        for k in axes(Ō,3)
            λ = @view(λ̄[:,k])
            Devices.propagate!(Drive(t̄[i+1]), device, basis, -τ/2, λ)
            Devices.propagate!(STATIC, device, basis, -τ, λ)
            Devices.propagate!(Drive(t̄[i]),   device, basis, -τ/2, λ)
        end

        # CALCULATE GRADIENT SIGNAL BRAKETS
        callback !== nothing && callback(i, t̄[i], ψ)
        for k in axes(Ō,3)
            λ = @view(λ̄[:,k])
            for j in 1:Devices.ngrades(device)
                z = Devices.braket(Gradient(j, t̄[i]), device, basis, λ, ψ)
                result[i,j,k] = 2 * imag(z) # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
            end
        end
    end

    return result
end

