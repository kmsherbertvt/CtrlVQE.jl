module ODEEvolutions

import DifferentialEquations

using CtrlVQE: LinearAlgebraTools, Devices, Evolutions

struct ODE <: Evolutions.EvolutionAlgorithm end

function Evolutions.evolve!(::Type{ODE}, device::Devices.Device, args...; kwargs...)
    return Evolutions.evolve!(ODE, device, Bases.Dressed, args...; kwargs...)
end

function Evolutions.evolve!(::Type{ODE},
    device::Devices.Device,
    basis::Type{<:Bases.BasisType},
    T::Real,
    ψ::AbstractVector{<:Complex{<:AbstractFloat}};
    callback=nothing
)
    # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
    U = Devices.evolver(Operators.Static, device, basis, 0)
    V = Devices.operator(Operators.Drive, device, basis, 0)
    # PROMOTE `V` SO THAT IT CAN BE ROTATED IN PLACE
    F = promote_type(eltype(U), eltype(V))
    V = convert(Matrix{F}, V)

    # DELEGATE TO `DifferentialEquations`
    i = Ref(0)
    p = (device, basis, U, V, callback, i)
    schrodinger = DifferentialEquations.ODEProblem(_interaction!, ψ, (0.0, T), p)
    solution = solve(schrodinger, save_everystep=false)
    ψ .= solution.u[end]

    # ROTATE OUT OF INTERACTION PICTURE
    ψ = Devices.evolve!(Operators.Static, device, basis, T, ψ)

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

end # module ODEEvolutions
