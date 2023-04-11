using BenchmarkTools: @benchmark, Trial
import Experiments
import CtrlVQE
import AnalyticSquarePulse
import ..Nmn

const S = CtrlVQE.Signals.Constant
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
    alg::Int            # INDEXES WHICH EVOLUTION ALGORITHM
    m::Int              # NUMBER OF LEVELS ON TRANSMON (2 or 3)
    Ω::F                # MAGNITUDE OF CONSTANT PULSE
    φ::F                # PHASE OF CONSTANT PULSE (take negative numbers as real)
    jmax::Int           # NUMBER OF DETUNINGS TO TEST (linear up to 1 GHz)
    kmax::Int           # MAX TROTTER STEPS IS r=2^kmax
end

function Control(
    F::Type{<:AbstractFloat},
    A::CtrlVQE.Evolutions.EvolutionAlgorithm,
    m::Int,
    Ω::Real,
    φ::Real,
    jmax::Int,
    kmax::Int,
)
    float = findfirst(F .== floats)
    alg = findfirst([A == algorithm for algorithm in algorithms])
    return Control(float, alg, m, F(Ω), F(φ), jmax, kmax)
end

struct Independent{F} <: Experiments.Independent
    Δ::F                    # DETUNING
    r::Int                  # NUMBER OF TROTTER STEPS
    T::F                    # TOTAL DURATION OF PULSE
end

struct Dependent <: Experiments.Dependent
    iF::Float64             # INFIDELITY wrt ANALYTICAL SOLUTION
end

struct Setup{F,P} <: Experiments.Setup
    device::D{F,S{P}}       # THE DEVICE OBJECT BEING EVOLVED
    ψ0::Vector{Complex{F}}  # THE INITIAL STATEVECTOR
end

struct Result{F} <: Experiments.Result
    ψA::Vector{Complex{F}}  # THE FINAL STATEVECTOR, ANALYTICAL
    ψx::Vector{Complex{F}}  # THE FINAL STATEVECTOR
end




function Experiments.initialize(expmt::Control{F}) where {F}
    # TECHNICALLY WE HAVE A TYPE INSTABILITY HERE - PULSE MAY BE REAL OR COMPLEX
    pulse = expmt.φ < 0 ? S(expmt.Ω) : S(expmt.Ω * exp(im*expmt.φ))
    device = CtrlVQE.SystematicTransmonDevice(F, expmt.m, 1, pulse)
    ψ0 = zeros(Complex{F}, expmt.m); ψ0[2] = 1
    return Setup(device, ψ0)
end

function Experiments.mapindex(expmt::Control{F}, i::Integer) where {F}
    i, j = divrem(i, expmt.jmax+1)
    i, k = divrem(i, expmt.kmax+1)
    Δ = F(2π * j / expmt.jmax)
    r = 2^k
    T = F(10^i)
    return Independent(Δ,r,T)
end

function Experiments.runtrial(
    expmt::Control{F},
    setup::Setup,
    xvars::Independent,
) where {F}
    A = algorithms[expmt.alg]

    ω, δ, Ω = setup.device.ω̄[1], setup.device.δ̄[1], setup.device.Ω̄[1](0)
    ν = ω - xvars.Δ
    setup.device.ν̄[1] = ν

    ψA = AnalyticSquarePulse.evolve_transmon(ω, δ, Ω, ν, expmt.m, xvars.T, setup.ψ0)
    ψx = CtrlVQE.Evolutions.evolve(A, setup.device, xvars.T, setup.ψ0; r=xvars.r)
    return Result(ψA, ψx)
end

function Experiments.synthesize(::Control,
    setup::Setup,
    xvars::Independent,
    result::Result,
)
    iF = 1 - abs(result.ψA' * result.ψx)^2
    return Dependent(iF)
end