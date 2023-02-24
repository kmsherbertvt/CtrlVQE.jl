import DifferentialEquations, KrylovKit

import ..Basis, ..Temporality, ..Devices, ..LinearAlgebraTools


function evolve(device::Devices.Device, ψ0::AbstractVector, args...; kwargs...)
    ψ = copy(ψ0)
    return evolve!(device, ψ, args...; kwargs...)
end

abstract type Mode end
preferredbasis(::Mode) = error("Not Implemented")

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Basis.AbstractBasis},
    T::Real,
    mode::Type{<:Mode}=Rotate;
    kwargs...
)
    R = Devices.basisrotation(basis, preferredbasis(mode), device)
    ψ = LinearAlgebraTools.rotate!(R, ψ)
    ψ = evolve!(device, ψ, T, mode; kwargs...)
    ψ = LinearAlgebraTools.rotate!(R', ψ)
    return ψ
end


struct Rotate <: Mode end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Basis.AbstractBasis},
    T::Real;
    kwargs...
)
    return evolve!(device, ψ, T, Rotate; kwargs...)
end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Basis.AbstractBasis},
    T::Real;
    kwargs...
)
    R = Devices.basisrotation(basis, preferredbasis(mode), device)
    ψ = LinearAlgebraTools.rotate!(R, ψ)
    ψ = evolve!(device, ψ, T, Rotate; kwargs...)
    ψ = LinearAlgebraTools.rotate!(R', ψ)
    return ψ
end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Basis.AbstractBasis}=Basis.Occupation,
    T::Real,
    ::Type{Rotate};
    r::Int=1000,
    callback=nothing,
)
    τ = T / r

    t = 0
    callback !== nothing && callback(0, t, ψ)
    ψ = Devices.propagate!(Temporality.Driven, device, t, τ/2, ψ, basis)

    for i in 1:r-1
        t += τ
        callback !== nothing && callback(i, t, ψ)
        ψ = Devices.propagate!(Temporality.Static, device,    τ, ψ, basis)
        ψ = Devices.propagate!(Temporality.Driven, device, t, τ, ψ, basis)
    end

    t += τ
    callback !== nothing && callback(r, t, ψ)
    ψ = Devices.propagate!(Temporality.Static, device,    τ,   ψ, basis)
    ψ = Devices.propagate!(Temporality.Driven, device, t, τ/2, ψ, basis)

    ψ ./= norm(ψ)
    return ψ
end






struct ODE <: Mode end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Basis.AbstractBasis}=Basis.Dressed,
    T::Real,
    ::Type{ODE};
    callback=nothing,
)
    # ALLOCATE MEMORY FOR INTERACTION HAMILTONIAN
    U = Devices.staticevolver(Temporality.Static, device, 0, basis)
    V = Devices.  hamiltonian(Temporality.Driven, device, 0, basis)

    # DELEGATE TO `DifferentialEquations`
    i = Ref(0)
    p = (device, basis, U, V, callback, i)
    schrodinger = DifferentialEquations.ODEProblem(_interaction!, ψ, (0.0, T), p)
    solution = solve(schrodinger, save_everystep=false)
    ψ .= solution.u[end]

    # RENORMALIZE
    ψ ./= norm(ψ)
    return ψ
end

function _interaction!(du, u, p, t)
    device, basis, U, V, callback, i = p

    callback !== nothing && callback(i[], t, u)
    i[] += 1

    # H(t) = exp(𝑖t⋅H0) V(t) exp(-𝑖t⋅H0)
    U .= Devices.staticevolver(Temporality.Static, device, t, basis)
    V .= Devices.  hamiltonian(Temporality.Driven, device, t, basis)
    V = LinearAlgebraTools.rotate!(U', V)

    # ∂ψ/∂t = -𝑖 H(t) ψ
    V .*= -im
    mul!(du, V, u)
end





struct Direct <: Mode end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Basis.AbstractBasis}=Basis.Dressed,
    T::Real,
    ::Type{Direct};
    r::Int=1000,
    callback=nothing,
)
    τ = T / r

    t = 0
    callback !== nothing && callback(0, t, ψ)
    U = Devices.staticevolver(Temporality.Static, device, t, basis)
    V = Devices.  hamiltonian(Temporality.Driven, device, t, basis)
    V = LinearAlgebraTools.rotate!(U', V)
    V .*= -im * τ/2
    V = LinearAlgebraTools.exponentiate!(V)
    ψ = LinearAlgebraTools.rotate!(V, ψ)

    for i in 1:r-1
        t += τ
        callback !== nothing && callback(i, t, ψ)
        U .= Devices.staticevolver(Temporality.Static, device, t, basis)
        V .= Devices.  hamiltonian(Temporality.Driven, device, t, basis)
        V = LinearAlgebraTools.rotate!(U', V)
        V .*= -im * τ
        V = LinearAlgebraTools.exponentiate!(V)
        ψ = LinearAlgebraTools.rotate!(V, ψ)
    end

    t += τ
    callback !== nothing && callback(r, t, ψ)
    U .= Devices.staticevolver(Temporality.Static, device, t, basis)
    V .= Devices.  hamiltonian(Temporality.Driven, device, t, basis)
    V = LinearAlgebraTools.rotate!(U', V)
    V .*= -im * τ/2
    V = LinearAlgebraTools.exponentiate!(V)
    ψ = LinearAlgebraTools.rotate!(V, ψ)

    ψ ./= norm(ψ)
    return ψ
end






struct Lanczos <: Mode end

function evolve!(
    device::Devices.Device,
    ψ::AbstractVector,
    basis::Type{<:Basis.AbstractBasis}=Basis.Dressed,
    T::Real,
    ::Type{Lanczos};
    r::Int=1000,
    callback=nothing,
)
    τ = T / r

    t = 0
    callback !== nothing && callback(0, t, ψ)
    U = Devices.staticevolver(Temporality.Static, device, t, basis)
    V = Devices.  hamiltonian(Temporality.Driven, device, t, basis)
    V = LinearAlgebraTools.rotate!(U', V)
    ψ .= KrylovKit.exponentiate(V, -im * τ/2, ψ)[1]

    for i in 1:r-1
        t += τ
        callback !== nothing && callback(i, t, ψ)
        U .= Devices.staticevolver(Temporality.Static, device, t, basis)
        V .= Devices.  hamiltonian(Temporality.Driven, device, t, basis)
        V = LinearAlgebraTools.rotate!(U', V)
        ψ .= KrylovKit.exponentiate(V, -im * τ, ψ)[1]
    end

    t += τ
    callback !== nothing && callback(r, t, ψ)
    U .= Devices.staticevolver(Temporality.Static, device, t, basis)
    V .= Devices.  hamiltonian(Temporality.Driven, device, t, basis)
    V = LinearAlgebraTools.rotate!(U', V)
    ψ .= KrylovKit.exponentiate(V, -im * τ/2, ψ)[1]

    ψ ./= norm(ψ)
    return ψ
end