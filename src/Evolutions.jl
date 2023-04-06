import ..Bases, ..Operators, ..LinearAlgebraTools, ..Devices


function trapezoidaltimegrid(T::Real, r::Int)
    # NOTE: Negative values of T give reversed time grid.
    τ = T / r
    τ̄ = fill(τ, r+1); τ̄[[begin, end]] ./= 2
    t̄ = abs(τ) * (T ≥ 0 ? (0:r) : reverse(0:r))
    return τ, τ̄, t̄
end



abstract type EvolutionAlgorithm end


#= Non-mutating `evolve` function. =#

function evolve(
    device::Devices.Device,
    T::Real,
    ψ0::AbstractVector;
    kwargs...
)
    ψ = convert(Array{LinearAlgebraTools.cis_type(ψ0)}, copy(ψ0))
    return evolve!(device, T, ψ; kwargs...)
end

function evolve(
    device::Devices.Device,
    basis::Bases.BasisType,
    T::Real,
    ψ0::AbstractVector;
    kwargs...
)
    ψ = convert(Array{LinearAlgebraTools.cis_type(ψ0)}, copy(ψ0))
    return evolve!(device, basis, T, ψ; kwargs...)
end

function evolve(
    algorithm::EvolutionAlgorithm,
    device::Devices.Device,
    T::Real,
    ψ0::AbstractVector;
    kwargs...
)
    ψ = convert(Array{LinearAlgebraTools.cis_type(ψ0)}, copy(ψ0))
    return evolve!(algorithm, device, T, ψ; kwargs...)
end

function evolve(
    algorithm::EvolutionAlgorithm,
    device::Devices.Device,
    basis::Bases.BasisType,
    T::Real,
    ψ0::AbstractVector;
    kwargs...
)
    ψ = convert(Array{LinearAlgebraTools.cis_type(ψ0)}, copy(ψ0))
    return evolve!(algorithm, device, basis, T, ψ; kwargs...)
end







struct Rotate <: EvolutionAlgorithm end; const ROTATE = Rotate()

function evolve!(args...; kwargs...)
    return evolve!(ROTATE, args...; kwargs...)
end

function evolve!(::Rotate, device::Devices.Device, args...; kwargs...)
    return evolve!(ROTATE, device, Bases.OCCUPATION, args...; kwargs...)
end

function evolve!(::Rotate,
    device::Devices.Device,
    basis::Bases.BasisType,
    T::Real,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
    r::Int=1000,
    callback=nothing
)
    τ, τ̄, t̄ = trapezoidaltimegrid(T, r)

    # FIRST STEP: NO NEED TO APPLY STATIC OPERATOR
    callback !== nothing && callback(0, t̄[1], ψ)
    ψ = Devices.propagate!(Operators.DRIVE,  device, basis, τ̄[1], ψ, t̄[1])

    # RUN EVOLUTION
    for i in 2:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        ψ = Devices.propagate!(Operators.STATIC, device, basis, τ, ψ)
        ψ = Devices.propagate!(Operators.DRIVE,  device, basis, τ̄[i], ψ, t̄[i])
    end

    return ψ
end






struct Direct <: EvolutionAlgorithm end; const DIRECT = Direct()

function evolve!(::Direct, device::Devices.Device, args...; kwargs...)
    return evolve!(DIRECT, device, Bases.DRESSED, args...; kwargs...)
end

function evolve!(::Direct,
    device::Devices.Device,
    basis::Bases.BasisType,
    T::Real,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
    r::Int=1000,
    callback=nothing
)
    τ, τ̄, t̄ = trapezoidaltimegrid(T, r)

    # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
    U = Devices.evolver(Operators.STATIC, device, basis, 0)
    V = Devices.operator(Operators.DRIVE, device, basis, 0)
    # PROMOTE `V` SO THAT IT CAN BE ROTATED IN PLACE AND EXPONENTIATED
    F = Complex{real(promote_type(eltype(U), eltype(V)))}
    V = convert(Matrix{F}, copy(V))

    # RUN EVOLUTION
    for i in 1:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        U .= Devices.evolver(Operators.STATIC, device, basis, t̄[i])
        V .= Devices.operator(Operators.DRIVE, device, basis, t̄[i])
        V = LinearAlgebraTools.rotate!(U', V)
        V = LinearAlgebraTools.cis!(V, -τ̄[i])
        ψ = LinearAlgebraTools.rotate!(V, ψ)
    end

    # ROTATE OUT OF INTERACTION PICTURE
    ψ = Devices.evolve!(Operators.STATIC, device, basis, T, ψ)

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
    Ō::AbstractVector{<:AbstractMatrix};
    callback=nothing
)
    τ, τ̄, t̄ = trapezoidaltimegrid(T, r)

    # PREPARE SIGNAL ARRAYS ϕ̄[k,j,i]
    F = real(LinearAlgebraTools.cis_type(ψ0))
    ϕ̄ = Array{F}(undef, r+1, Devices.ngrades(device), length(Ō))

    # PREPARE STATE AND CO-STATES
    ψ = convert(Array{LinearAlgebraTools.cis_type(ψ0)}, copy(ψ0))
    λ̄ = [convert(Array{LinearAlgebraTools.cis_type(ψ0)}, copy(ψ0)) for k in eachindex(Ō)]
    for k in eachindex(Ō)
        λ̄[k] = evolve!(ROTATE, device, basis,  T, λ̄[k]; r=r)
        λ̄[k] = LinearAlgebraTools.rotate!(Ō[k], λ̄[k])    # NOTE: O is not unitary.
        λ̄[k] = evolve!(ROTATE, device, basis, -T, λ̄[k]; r=r)
    end

    # START THE FIRST STEP
    ψ = Devices.propagate!(Operators.DRIVE, device, basis, τ̄[1]/2, ψ, t̄[1])
    for λ in λ̄
        Devices.propagate!(Operators.DRIVE, device, basis, τ̄[1]/2, λ, t̄[1])
    end

    # FIRST GRADIENT SIGNALS
    callback !== nothing && callback(1, t̄[1], ψ)
    for (k, λ) in enumerate(λ̄)
        for j in 1:Devices.ngrades(device)
            z = Devices.braket(Operators.GRADIENT, device, basis, λ, ψ, j, t̄[1])
            ϕ̄[1,j,k] = 2 * imag(z)  # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
        end
    end

    # ITERATE OVER TIME
    for i in 2:r+1
        # COMPLETE THE PREVIOUS TIME-STEP AND START THE NEXT
        ψ = Devices.propagate!(Operators.DRIVE,  device, basis, τ̄[i-1]/2, ψ, t̄[i-1])
        ψ = Devices.propagate!(Operators.STATIC, device, basis, τ, ψ)
        ψ = Devices.propagate!(Operators.DRIVE,  device, basis, τ̄[i]/2, ψ, t̄[i])
        for λ in λ̄
            Devices.propagate!(Operators.DRIVE,  device, basis, τ̄[i-1]/2, λ, t̄[i-1])
            Devices.propagate!(Operators.STATIC, device, basis, τ, λ)
            Devices.propagate!(Operators.DRIVE,  device, basis, τ̄[i]/2, λ, t̄[i])
        end

        # CALCULATE GRADIENT SIGNAL BRAKETS
        callback !== nothing && callback(i, t̄[i], ψ)
        for (k, λ) in enumerate(λ̄)
            for j in 1:Devices.ngrades(device)
                z = Devices.braket(Operators.GRADIENT, device, basis, λ, ψ, j, t̄[i])
                ϕ̄[i,j,k] = 2 * imag(z)  # ϕ̄[i,j,k] = -𝑖z + 𝑖z̄
            end
        end
    end

    # NOTE: I'd like to finish the last time-step, but there's no reason to.

    return ϕ̄
end

