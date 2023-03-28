import ..Bases, ..Operators, ..LinearAlgebraTools, ..Devices



abstract type EvolutionAlgorithm end


#= Non-mutating `evolve` function. =#

function evolve(
    device::Devices.Device,
    T::Real,
    ψ0::AbstractVector{<:Complex{<:AbstractFloat}};
    kwargs...
)
    ψ = convert(Array{LinearAlgebraTools.cis_type(ψ0)}, ψ0)
    return evolve!(device, T, ψ; kwargs...)
end

function evolve(
    device::Devices.Device,
    basis::Type{<:Bases.BasisType},
    T::Real,
    ψ0::AbstractVector{<:Complex{<:AbstractFloat}};
    kwargs...
)
    ψ = convert(Array{LinearAlgebraTools.cis_type(ψ0)}, ψ0)
    return evolve!(device, basis, T, ψ; kwargs...)
end

function evolve(
    algorithm::Type{<:EvolutionAlgorithm},
    device::Devices.Device,
    T::Real,
    ψ0::AbstractVector{<:Complex{<:AbstractFloat}};
    kwargs...
)
    ψ = convert(Array{LinearAlgebraTools.cis_type(ψ0)}, ψ0)
    return evolve!(algorithm, device, T, ψ; kwargs...)
end

function evolve(
    algorithm::Type{<:EvolutionAlgorithm},
    device::Devices.Device,
    basis::Type{<:Bases.BasisType},
    T::Real,
    ψ0::AbstractVector{<:Complex{<:AbstractFloat}};
    kwargs...
)
    ψ = convert(Array{LinearAlgebraTools.cis_type(ψ0)}, ψ0)
    return evolve!(algorithm, device, basis, T, ψ; kwargs...)
end







struct Rotate <: EvolutionAlgorithm end

function evolve!(args...; kwargs...)
    return evolve!(Rotate, args...; kwargs...)
end

function evolve!(::Type{Rotate}, device::Devices.Device, args...; kwargs...)
    return evolve!(Rotate, device, Bases.Occupation, args...; kwargs...)
end

function evolve!(::Type{Rotate},
    device::Devices.Device,
    basis::Type{<:Bases.BasisType},
    T::Real,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
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
        ψ = Devices.propagate!(Operators.Drive,  device, basis, τ̄[1], ψ, t̄[1])

    # RUN EVOLUTION
    for i in 2:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        ψ = Devices.propagate!(Operators.Static, device, basis, τ̄[i], ψ)
        ψ = Devices.propagate!(Operators.Drive,  device, basis, τ̄[i], ψ, t̄[i])
    end

    return ψ
end






struct Direct <: EvolutionAlgorithm end

function evolve!(::Type{Direct}, device::Devices.Device, args...; kwargs...)
    return evolve!(Direct, device, Bases.Dressed, args...; kwargs...)
end

function evolve!(::Type{Direct},
    device::Devices.Device,
    basis::Type{<:Bases.BasisType},
    T::Real,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
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
    # PROMOTE `V` SO THAT IT CAN BE ROTATED IN PLACE AND EXPONENTIATED
    F = Complex{real(promote_type(eltype(U), eltype(V)))}
    V = convert(Matrix{F}, V)

    # RUN EVOLUTION
    for i in 1:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        U .= Devices.evolver(Operators.Static, device, basis, t̄[i])
        V .= Devices.operator(Operators.Drive, device, basis, t̄[i])
        V = LinearAlgebraTools.rotate!(U', V)
        V = LinearAlgebraTools.cis!(V, -τ̄[i])
        ψ = LinearAlgebraTools.rotate!(V, ψ)
    end

    return ψ
end




function gradientsignals(device::Devices.Device, args...; kwargs...)
    return gradientsignals(device, Bases.Occupation, args...; kwargs...)
end

function gradientsignals(
    device::Devices.Device,
    basis::Type{<:Bases.BasisType},
    T::Real,
    ψ0::AbstractVector,
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
    F = real(LinearAlgebraTools.cis_type(ψ0))
    ϕ̄ = Array{F}(undef, length(Ō), ngrades(device), r+1)

    # PREPARE STATE AND CO-STATES
    ψ = convert(Array{LinearAlgebraTools.cis_type(ψ0)}, ψ0)
    λ̄ = [convert(Array{LinearAlgebraTools.cis_type(ψ0)}, ψ0) for k in eachindex(Ō)]
    for k in eachindex(Ō)
        λ̄[k] = evolve!(Rotate, device, basis,  T, λ̄[k]; r=r)
        λ̄[k] = LinearAlgebraTools.rotate!(Ō[k], λ̄[k])    # NOTE: O is not unitary.
        λ̄[k] = evolve!(Rotate, device, basis, -T, λ̄[k]; r=r)
    end

    # FIRST STEP: NO NEED TO APPLY STATIC OPERATOR
    callback !== nothing && callback(0, t̄[0], ψ)
    ψ = Devices.propagate!(Operators.Drive, device, basis, τ̄[1], ψ, t̄[1])
    for λ in λ̄
        Devices.propagate!(Operators.Drive, device, basis, τ̄[1], λ, t̄[1])
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
        ψ = Devices.propagate!(Operators.Static, device, basis, τ̄[i], ψ)
        ψ = Devices.propagate!(Operators.Drive,  device, basis, τ̄[i], ψ, t̄[i])
        for λ in λ̄
            Devices.propagate!(Operators.Static, device, basis, τ̄[i], λ)
            Devices.propagate!(Operators.Drive,  device, basis, τ̄[i], λ, t̄[i])
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

