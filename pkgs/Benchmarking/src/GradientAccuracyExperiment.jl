import Random
import FiniteDifferences
import LinearAlgebra: qr, Diagonal, norm, diag, Hermitian
using BenchmarkTools: @benchmark, Trial
import Experiments
import CtrlVQE
import AnalyticSquarePulse
import ..Nmn

const S = CtrlVQE.Signals.ComplexConstant
const D = CtrlVQE.Devices.TransmonDevice

const algorithms = [
    CtrlVQE.Evolutions.ROTATE,
    CtrlVQE.Evolutions.DIRECT,
]

const floats = [
    Float64,
    Float32,
    Float16,
]

struct Control{F} <: Experiments.Control
    float::Int          # INDEXES WHICH FLOAT TYPE
    m::Int              # NUMBER OF LEVELS ON TRANSMON (2 or 3)
    n::Int              # NUMBER OF QUBITS
    seed::Int           # RANDOM SEED, FOR GENERATING HERMITIAN
    λ::F                # RMS EIGENVALUE OF RANDOMLY GENERATED HERMITIAN
    T::F                # PULSE DURATION
    Ω::F                # MAGNITUDE OF CONSTANT PULSE
    Δ::F                # ANHARMONICITY
    r::Int              # TROTTER STEPS FOR EVOLUTION (NEED NOT EXCEED ~2000)
end

function Control(
    F::Type{<:AbstractFloat},
    m::Int,
    n::Int,
    seed::Int,
    λ::Real,
    T::Real,
    Ω::Real,
    Δ::Real,
    r::Int,
)
    float = findfirst(F .== floats)
    return Control(float, m, n, seed, F(λ), F(T), F(Ω), F(Δ), r)
end

struct Independent <: Experiments.Independent
    rG::Int                 # NUMBER OF TIME STEPS FOR GRADIENT SIGNAL
end

struct Dependent{F} <: Experiments.Dependent
    # ANALYTICAL GRADIENTS
    A0_RMS::F
    B0_RMS::F               # MORE-OR-LESS WHAT WE CARE ABOUT
    ν0_RMS::F
    # FINITE DIFFERENCES
    AΔ_RMS::F
    BΔ_RMS::F               # BY DESIGN, THESE *SHOULD* BE ZERO
    νΔ_RMS::F
    # ELEMENT-WISE ERRORS
    εA_RMS::F
    εB_RMS::F               # PRESUMABLY THE SAME AS ..0, IF ..Δ IS ZERO
    εν_RMS::F
end

struct Setup{F,P} <: Experiments.Setup
    observable::Matrix{Complex{F}}  # AN OBSERVABLE MINIMIZED BY OUR EVOLUTION
    device::D{F,S{P}}       # THE DEVICE OBJECT BEING EVOLVED
    ψ0::Vector{Complex{F}}  # THE INITIAL STATEVECTOR
    ψ::Vector{Complex{F}}   # THE EVOLVED STATEVECTOR
    gΔ::Vector{F}           # FINITE DIFFERENCE
end

struct Result{F} <: Experiments.Result
    ϕ̄::Array{F,3}           # GRADIENT SIGNALS
    g0::Vector{F}           # ANALYTICAL GRADIENT
end




function Experiments.initialize(expmt::Control{F}) where {F}
    n = expmt.n
    pulse = CtrlVQE.Signals.ComplexConstant(zero(F), zero(F))
    device = CtrlVQE.SystematicTransmonDevice(expmt.m, n, pulse)

    # ASSIGN THE 3n PARAMETERS
    q̄ = 0:n-1   # This isn't the drive qubits! Just helping to construct correct x̄.
    φ = F(2π/n) .* q̄
    Ā = expmt.Ω .* cos.(φ)
    B̄ = expmt.Ω .* sin.(φ)
    ν̄ = device.ω̄ - ((-1).^q̄ * expmt.Δ)  # Switch back and forth between ν=ω±Δ

    x̄0 = Vector{F}(undef, 3n)
    x̄0[1:2:2n] .= Ā
    x̄0[2:2:2n] .= B̄
    x̄0[1+2n:3n].= ν̄
    CtrlVQE.Parameters.bind(device, x̄0)

    #######################################
    # PREPARE THE INITIAL STATE, AND EVOLVE IT
    ψ0 = zeros(Complex{F}, expmt.m^n); ψ0[2] = 1
    ψ = CtrlVQE.Evolutions.evolve(device, expmt.T, ψ0; r=expmt.r)

    #######################################
    # CONSTRUCT THE OBSERVABLE

    # FIRST WE GENERATE A RANDOM ORTHOGONAL MATRIX, WITH ONE DEGREE OF FREEDOM REMOVED...
    Random.seed!(expmt.seed)
    N = expmt.m ^ n
    Q, R = qr(randn(F, N-1, N-1))
    Q = Q * Diagonal(sign.(diag(R)))

    # ADD IN THAT FIRST DIMENSION TO ENSURE THE EVOLVED STATE IS AN EIGENSTATE
    U = zeros(Complex{F}, N, N)
    U[:, 1] .= ψ
    U[2:N, 2:N] .= Q

    # APPLY GRAM-SCHMIDT ORTHOGONALIZATION
    for i in 2:N
        u = U[:,i]
        for j in 1:i-1
            u .-= (U[:,j]'*U[:,i]) .* U[:,j]
        end
        U[:,i] .= u ./ norm(u)
    end
    # NOTE: Technically this fails if ψ has no support on |0̄⟩ state...

    # NEXT WE GENERATE A UNIFORM RANDOM EIGENSPECTRUM, CONSTRAINED BY RMS
    Λ = 2*rand(N) .- 1      # [-1, 1)
    Λ ./= norm( Λ ./ (expmt.λ * √N) )
    Λ = sort(Λ)     # SORT SO LOWEST EIGENVALUE TAKES THE FIRST INDEX

    # NOW WE COMBINE THE TWO
    observable = U * Diagonal(Λ) * U'
    # observable .= Hermitian(observable) # KNOCK OFF SOME ROUND-OFF ERROR


    #######################################
    # RUN THE FINITE DIFFERENCE
    function f(x̄)
        CtrlVQE.Parameters.bind(device, x̄)
        ψx = CtrlVQE.Evolutions.evolve(device, expmt.T, ψ0; r=expmt.r)
        return real(ψx'*observable*ψx)
    end
    gΔ = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, x̄0)[1]
    gΔ = identity.(gΔ)

    CtrlVQE.Parameters.bind(device, x̄0)
    return Setup(observable, device, ψ0, ψ, gΔ)
end

function Experiments.mapindex(expmt::Control, i::Integer)
    return Independent(2^i)
end

function Experiments.runtrial(
    expmt::Control{F},
    setup::Setup,
    xvars::Independent,
) where {F}
    O = setup.observable
    τ, τ̄, t̄ = CtrlVQE.Evolutions.trapezoidaltimegrid(expmt.T, xvars.rG)
    ϕ̄ = CtrlVQE.Evolutions.gradientsignals(setup.device, expmt.T, setup.ψ0, xvars.rG, [O])
    g0 = CtrlVQE.Devices.gradient(setup.device, τ̄, t̄, ϕ̄[:,:,1])
    return Result(ϕ̄, g0)
end

function Experiments.synthesize(
    expmt::Control,
    setup::Setup,
    xvars::Independent,
    result::Result,
)
    n = expmt.n
    g0, gΔ = result.g0, setup.gΔ
    ε = result.g0 - setup.gΔ

    # DEFINE SLICES FOR EACH PARAMETER TYPE
    A = 1:2:2n
    B = 2:2:2n
    ν = 1+2n:3n

    rms(x̄) = √(sum(x̄.^2)/length(x̄))
    return Dependent(
        rms(g0[A]), rms(g0[B]), rms(g0[ν]),
        rms(gΔ[A]), rms(gΔ[B]), rms(gΔ[ν]),
        rms( ε[A]), rms( ε[B]), rms( ε[ν]),
    )
end