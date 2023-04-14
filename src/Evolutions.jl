import LinearAlgebra: norm
import ..Bases, ..LinearAlgebraTools, ..Devices
import ..Operators: STATIC, Drive, Gradient

import ..TempArrays: array
const LABEL = Symbol(@__MODULE__)

using ..LinearAlgebraTools: List

function trapezoidaltimegrid(T::Real, r::Int)
    # NOTE: Negative values of T give reversed time grid.
    τ = T / r
    τ̄ = fill(τ, r+1); τ̄[[begin, end]] ./= 2
    t̄ = abs(τ) * (T ≥ 0 ? (0:r) : reverse(0:r))
    return τ, τ̄, t̄
end

abstract type Algorithm end


#= Non-mutating `evolve` function. =#

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
    # `result` IS GIVEN AS A 2D ARRAY BUT MUST BE 3D FOR DELEGATION
    result !== nothing && (result = reshape(result, size(result, 1), size(result, 2), 1))

    # PERFORM THE DELEGATION
    result = gradientsignals(device, basis, T, ψ0, r, [O]; result=result, kwargs...)

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
    Ō::List{<:AbstractMatrix};
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
    ψ = Vector{LinearAlgebraTools.cis_type(ψ0)}(undef, length(ψ0))
    ψ .= ψ0
    ψ = evolve!(evolution, device, basis, T, ψ)
    λ̄ = [LinearAlgebraTools.rotate!(O, copy(ψ)) for O in Ō]

    # TODO (hi): HEY! Can't we use temp arrays for ψ and λ̄? Just need to be careful with index.

    #= TODO (hi): Check closely the accuracy of first and last Φ values.

        Do we need to half-evolve V here?
        There is something beautifully symmetric about *not* doing so.
        Every drive propagation has exactly τ/2.
        And the first and last gradient points correspond
            to the true beginning and end of time evolution,
            which feels right.

        BUT I was doing half-evolution before,
            and the first/last Φ seemed to match finite difference exactly.
        So, it might be objectively wrong to change that...

        If so, must use τ̄[i]/2 instead of τ/2 below, for all Device propagation.
        (And also add in a half-evolution before the first gradient point.)
    =#

    # LAST GRADIENT SIGNALS
    callback !== nothing && callback(r+1, t̄[r+1], ψ)
    for (k, λ) in enumerate(λ̄)
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
        for λ in λ̄
            Devices.propagate!(Drive(t̄[i+1]), device, basis, -τ/2, λ)
            Devices.propagate!(STATIC, device, basis, -τ, λ)
            Devices.propagate!(Drive(t̄[i]),   device, basis, -τ/2, λ)
        end

        # CALCULATE GRADIENT SIGNAL BRAKETS
        callback !== nothing && callback(i, t̄[i], ψ)
        for (k, λ) in enumerate(λ̄)
            for j in 1:Devices.ngrades(device)
                z = Devices.braket(Gradient(j, t̄[i]), device, basis, λ, ψ)
                result[i,j,k] = 2 * imag(z) # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
            end
        end
    end

    return result
end

