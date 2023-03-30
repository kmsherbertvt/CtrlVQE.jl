module LanczosEvolutions

import KrylovKit

using CtrlVQE: LinearAlgebraTools, Devices, Evolutions

struct Lanczos <: Evolutions.EvolutionAlgorithm end

function Evolutions.evolve!(::Type{Lanczos}, device::Devices.Device, args...; kwargs...)
    return Evolutions.evolve!(Lanczos, device, Bases.Dressed, args...; kwargs...)
end

function Evolutions.evolve!(::Type{Lanczos},
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
    # PROMOTE `V` SO THAT IT CAN BE ROTATED IN PLACE
    F = promote_type(eltype(U), eltype(V))
    V = convert(Matrix{F}, V)

    # RUN EVOLUTION
    for i in 1:r+1
        callback !== nothing && callback(i, t̄[i], ψ)
        U .= Devices.evolver(Operators.Static, device, basis, t̄[i])
        V .= Devices.operator(Operators.Drive, device, basis, t̄[i])
        V = LinearAlgebraTools.rotate!(U', V)
        ψ .= KrylovKit.exponentiate(V, -im * τ̄[i], ψ)[1]
    end

    # ROTATE OUT OF INTERACTION PICTURE
    ψ = Devices.evolve!(Operators.Static, device, basis, T, ψ)

    return ψ
end

end # module LanczosEvolutions
