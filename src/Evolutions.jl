import DifferentialEquations, KrylovKit
#= TODO: These should be considered optional dependencies
            and certainly do not need to be installed every time we run on ARC!!!
=#
import ..Bases, ..Operators, ..LinearAlgebraTools, ..Devices


function evolve(device::Devices.Device, ψ0::AbstractVector, args...; kwargs...)
    ψ = copy(ψ0)
    return evolve!(device, ψ, args...; kwargs...)
end

abstract type EvolutionAlgorithm end


struct Rotate <: EvolutionAlgorithm end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Bases.BasisType}, # TODO: Default
    T::Real;
    kwargs...
)
    return evolve!(device, ψ, T, Rotate; kwargs...)
end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Bases.BasisType}, # TODO: Default
    T::Real,
    ::Type{Rotate};
    r::Int=1000,
    callback=nothing
)
    # CONSTRUCT TIME GRID
    τ = T / r
    τ̄ = fill(τ, r + 1)
    τ̄[[begin, end]] ./= 2
    t̄ = τ * (0:r)

    # FIRST STEP: NO NEED TO APPLY STATIC OPERATOR
    callback !== nothing && callback(0, t̄[1], ψ)
    ψ = Devices.propagate!(Operators.Drive, device, t̄[1], τ̄[1], ψ, basis)

    # RUN EVOLUTION
    for i in 2:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        ψ = Devices.propagate!(Operators.Static, device, τ̄[i],       ψ, basis)
        ψ = Devices.propagate!(Operators.Drive,  device, t̄[i], τ̄[i], ψ, basis)
    end

    return ψ
end


function gradientsignals(
    device::Devices.Device,
    ψ0::AbstractVector,
    basis::Type{<:Bases.BasisType}, # TODO: Default
    T::Real,
    r::Int,
    Ō::AbstractVector{<:AbstractMatrix};
    callback=nothing
)
    # CONSTRUCT TIME GRID
    τ = T / r
    τ̄ = fill(τ, r + 1)
    τ̄[[begin, end]] ./= 2
    t̄ = τ * (0:r)

    # PREPARE SIGNAL ARRAYS ϕ̄[k,j,i]
    ϕ̄ = Array{eltype(real.(ψ0))}(undef, length(Ō), ngrades(device), r+1)

    # PREPARE STATE AND CO-STATES
    ψ = copy(ψ0)
    λ̄ = [copy(ψ0) for k in eachindex(Ō)]
    for k in eachindex(Ō)
        λ = evolve!(device, λ, basis,  T, Rotate; r=r)
        λ = LinearAlgebraTools.rotate!(Ō[k], λ)    # NOTE: O is not unitary.
        λ = evolve!(device, λ, basis, -T, Rotate; r=r)
        push!(λ̄, λ)
    end

    # FIRST STEP: NO NEED TO APPLY STATIC OPERATOR
    callback !== nothing && callback(0, t̄[0], ψ)
    ψ = Devices.propagate!(Operators.Drive, device, t̄[1], τ̄[1], ψ, basis)
    for λ in λ̄
        Devices.propagate!(Operators.Drive, device, t̄[1], τ̄[1], λ, basis)
    end

    # FIRST GRADIENT SIGNALS
    for (k, λ) in enumerate(λ̄)
        for j in 1:ngrades(device)
            z = Devices.braket(Operators.Gradient, device, basis, λ, ψ, j, t̄[1])
            ϕ̄[k,j,i] = 2 * imag(z)  # ϕ̄[k,j,i] = -𝑖z + 𝑖z̄
        end
    end

    # ITERATE OVER TIME
    for i in 2:r+1
        # CONTINUE TIME EVOLUTION
        callback !== nothing && callback(i, t̄[i], ψ)
        ψ = Devices.propagate!(Operators.Static, device, τ̄[i],       ψ, basis)
        ψ = Devices.propagate!(Operators.Drive,  device, t̄[i], τ̄[i], ψ, basis)
        for λ in λ̄
            Devices.propagate!(Operators.Static, device, τ̄[i],       λ, basis)
            Devices.propagate!(Operators.Drive,  device, t̄[i], τ̄[i], λ, basis)
        end

        # CALCULATE GRADIENT SIGNAL BRAKETS
        for (k, λ) in enumerate(λ̄)
            for j in 1:ngrades(device)
                z = Devices.braket(Operators.Gradient, device, basis, λ, ψ, j, t̄[i])
                ϕ̄[k,j,i] = 2 * imag(z)  # ϕ̄[k,j,i] = -𝑖z + 𝑖z̄
            end
        end
    end

    return ϕ̄
end

















struct ODE <: EvolutionAlgorithm end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Bases.BasisType}, # TODO: Default dressed
    T::Real,
    ::Type{ODE};
    callback=nothing
)
    # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
    U = Devices.evolver(Operators.Static, device, basis, 0)
    V = Devices.operator(Operators.Drive, device, basis, 0)

    # DELEGATE TO `DifferentialEquations`
    i = Ref(0)
    p = (device, basis, U, V, callback, i)
    schrodinger = DifferentialEquations.ODEProblem(_interaction!, ψ, (0.0, T), p)
    solution = solve(schrodinger, save_everystep=false)
    ψ .= solution.u[end]

    return ψ
end

function _interaction!(du, u, p, t)
    device, basis, U, V, callback, i = p

    callback !== nothing && callback(i[], t, u)
    i[] += 1

    # H(t) = exp(𝑖t⋅H0) V(t) exp(-𝑖t⋅H0)
    U .= Devices.evolver(Operators.Static, device, basis, t)
    V .= Devices.operator(Operators.Drive, device, basis, t)
    V = LinearAlgebraTools.rotate!(U', V)

    # ∂ψ/∂t = -𝑖 H(t) ψ
    V .*= -im
    mul!(du, V, u)
end





struct Direct <: EvolutionAlgorithm end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Bases.BasisType}, # TODO: Default dressed
    T::Real,
    ::Type{Direct};
    r::Int=1000,
    callback=nothing
)
    # CONSTRUCT TIME GRID
    τ = T / r
    τ̄ = fill(τ, r + 1)
    τ̄[[begin, end]] ./= 2
    t̄ = τ * (0:r)

    # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
    U = Devices.evolver(Operators.Static, device, basis, 0)
    V = Devices.operator(Operators.Drive, device, basis, 0)

    # RUN EVOLUTION
    for i in 1:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        U .= Devices.evolver(Operators.Static, device, basis, t̄[i])
        V .= Devices.operator(Operators.Drive, device, basis, t̄[i])
        V = LinearAlgebraTools.rotate!(U', V)
        V .*= -im * τ̄[i]
        V = LinearAlgebraTools.exponentiate!(V)
        ψ = LinearAlgebraTools.rotate!(V, ψ)
    end

    return ψ
end






struct Lanczos <: EvolutionAlgorithm end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Bases.BasisType}, # TODO: Default dressed
    T::Real,
    ::Type{Lanczos};
    r::Int=1000,
    callback=nothing
)
    # CONSTRUCT TIME GRID
    τ = T / r
    τ̄ = fill(τ, r + 1)
    τ̄[[begin, end]] ./= 2
    t̄ = τ * (0:r)

    # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
    U = Devices.evolver(Operators.Static, device, basis, 0)
    V = Devices.operator(Operators.Drive, device, basis, 0)

    # RUN EVOLUTION
    for i in 1:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        U .= Devices.evolver(Operators.Static, device, basis, t̄[i])
        V .= Devices.operator(Operators.Drive, device, basis, t̄[i])
        V = LinearAlgebraTools.rotate!(U', V)
        V .*= -im * τ̄[i]
        V = LinearAlgebraTools.exponentiate!(V)
        ψ .= KrylovKit.exponentiate(V, -im * τ̄[i], ψ)[1]
    end

    return ψ
end