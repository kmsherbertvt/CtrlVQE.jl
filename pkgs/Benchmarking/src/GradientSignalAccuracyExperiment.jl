import Random
import FiniteDifferences
import LinearAlgebra: qr, Diagonal, norm, diag

using BenchmarkTools: @benchmark, Trial
import Experiments
import CtrlVQE
import ..Nmn

const S = CtrlVQE.Signals.Composite
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
    seed::Int           # RANDOM SEED, FOR GENERATING HERMITIAN
    λ::F                # RMS EIGENVALUE OF RANDOMLY GENERATED HERMITIAN
    T::F                # TOTAL DURATION OF PULSE
    Ω::F                # MAGNITUDE OF CONSTANT PULSE
    Δ::F                # DETUNING
end

function Control(
    F::Type{<:AbstractFloat},
    m::Int,
    seed::Int,
    λ::Real,
    T::Real,
    Ω::Real,
    Δ::Real,
)
    float = findfirst(F .== floats)
    return Control(float, m, seed, F(λ), F(T), F(Ω), F(Δ))
end

struct Independent <: Experiments.Independent
    r::Int                  # NUMBER OF TROTTER STEPS
end

struct Dependent{F} <: Experiments.Dependent
    α1_RMS::F
    β1_RMS::F
    α2_RMS::F
    β2_RMS::F
    Δν1::F
    Δν2::F
    εα1_RMS::F
    εβ1_RMS::F
    εα2_RMS::F
    εβ2_RMS::F
    εΔν1::F
    εΔν2::F
end

struct Setup{F} <: Experiments.Setup
    observable::Matrix{F}   # A RANDOMLY CONSTRUCTED OBSERVABLE
    ψ0::Vector{Complex{F}}  # THE INITIAL STATEVECTOR
end

struct Result{F} <: Experiments.Result
    device::D{F,S}       # THE DEVICE OBJECT BEING EVOLVED
    ϕ̄::Array{F,3}           # GRADIENT SIGNALS
    g0::Vector{F}           # ANALYTICAL GRADIENT
    gΔ::Vector{F}           # FINITE DIFFERENCE
end




function Experiments.initialize(expmt::Control{F}) where {F}
    n = 2                   # THIS EXPERIMENT IS ALWAYS WITH 2 QUBITS
    N = expmt.m ^ n         # TOTAL HILBERT SPACE

    #######################################
    # CONSTRUCT THE OBSERVABLE

    # FIRST WE GENERATE A RANDOM ORTHOGONAL MATRIX
    Random.seed!(expmt.seed)
    Q, R = qr(randn(F, N, N))
    Q = Q * Diagonal(sign.(diag(R)))

    # NEXT WE GENERATE A UNIFORM RANDOM EIGENSPECTRUM, CONSTRAINED BY RMS
    Λ = 2*rand(N) .- 1      # [-1, 1)
    Λ ./= norm( Λ ./ (expmt.λ * √N) )

    # NOW WE COMBINE THE TWO
    observable = Q * Diagonal(Λ) * Q'

    ########################################
    # PREPARE THE INITIAL STATE
    ψ0 = zeros(Complex{F}, expmt.m^n); ψ0[2] = 1

    return Setup(observable, ψ0)
end

function Experiments.mapindex(expmt::Control, i::Integer)
    r = 2^i
    return Independent(r)
end

function Experiments.runtrial(
    expmt::Control{F},
    setup::Setup,
    xvars::Independent,
) where {F}
    n, r = 2, xvars.r
    O = setup.observable
    τ, τ̄, t̄ = CtrlVQE.Evolutions.trapezoidaltimegrid(expmt.T, r)

    ########################################
    # CONSTRUCT THE DEVICE

    # FIRST WE MAKE THE PULSE TEMPLATE, A FULLY-TROTTERIZED COMPLEX SIGNAL
    pulse = CtrlVQE.Signals.Composite([
        CtrlVQE.Signals.Constrained(
            CtrlVQE.Signals.ComplexInterval(zero(F), zero(F), t-τ/2, t+τ/2),
            :s1, :s2,
        ) for t in t̄
    ]...)

    # NEXT WE MAKE THE DEVICE
    device = CtrlVQE.SystematicTransmonDevice(F, expmt.m, n, pulse)

    # ASSIGN THE PULSE PARAMETERS
    x̄0 = CtrlVQE.Parameters.values(device)
    A1 = 1:2:(2r+2)         # THE SLICE FOR THE FIRST  PULSE'S REAL COEFFICIENTS
    B2 = (2r+4):2:(4r+4)    # THE SLICE FOR THE SECOND PULSE'S IMAG COEFFICIENTS
    x̄0[A1] .= expmt.Ω                       #  FIRST PULSE: A=Ω, B=0
    x̄0[B2] .= expmt.Ω                       # SECOND PULSE: A=0, B=Ω
    x̄0[end-1] = device.ω̄[1] - expmt.Δ       #  FIRST PULSE: ν ≡ ω - Δ
    x̄0[end]   = device.ω̄[2] + expmt.Δ       # SECOND PULSE: ν ≡ ω + Δ
    CtrlVQE.Parameters.bind(device, x̄0)

    ########################################
    # ANALYTICAL GRADIENT
    ϕ̄ = CtrlVQE.Evolutions.gradientsignals(device, expmt.T, setup.ψ0, r, [O])
    g0 = CtrlVQE.Devices.gradient(device, τ̄, t̄, ϕ̄[:,:,1])

    ########################################
    # FINITE DIFFERENCE
    function f(x̄)
        CtrlVQE.Parameters.bind(device, x̄)
        ψ = CtrlVQE.Evolutions.evolve(device, expmt.T, setup.ψ0; r=r)
        return real(ψ'*O*ψ)
    end
    gΔ = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, x̄0)[1]
    gΔ = identity.(gΔ)

    return Result(device, ϕ̄, g0, gΔ)
end

function Experiments.synthesize(::Control,
    setup::Setup,
    xvars::Independent,
    result::Result,
)
    g0 = result.g0
    ε = result.g0 - result.gΔ

    # DEFINE SLICES FOR EACH PARAMETER TYPE
    r = xvars.r
    A1 = 1:2:(2r+2)         # DRIVE 1, REAL PART
    B1 = 2:2:(2r+2)    # DRIVE 1, IMAG PART
    A2 = (2r+3):2:(4r+4)    # DRIVE 2, REAL PART
    B2 = (2r+4):2:(4r+4)    # DRIVE 2, IMAG PART

    rms(x̄) = √(sum(x̄.^2)/length(x̄))
    return Dependent(
        rms(g0[A1]),        # α1_RMS
        rms(g0[B1]),        # β1_RMS
        rms(g0[A2]),        # α2_RMS
        rms(g0[B2]),        # β2_RMS
        g0[end-1],          # Δν1
        g0[end],            # Δν2
        rms(ε[A1]),         # εα1_RMS
        rms(ε[B1]),         # εβ1_RMS
        rms(ε[A2]),         # εα2_RMS
        rms(ε[B2]),         # εβ2_RMS
        ε[end-1],           # εΔν1
        ε[end],             # εΔν2
    )
end