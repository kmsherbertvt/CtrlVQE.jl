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
    alg::Int            # INDEXES WHICH EVOLUTION ALGORITHM
    m::Int              # NUMBER OF LEVELS ON TRANSMON (2 or 3)
    n::Int              # NUMBER OF QUBITS
    T::F                # PULSE DURATION
    Ω::F                # MAGNITUDE OF CONSTANT PULSE
    Δ::F                # ANHARMONICITY
end

function Control(
    F::Type{<:AbstractFloat},
    A::CtrlVQE.Evolutions.EvolutionAlgorithm,
    m::Int,
    n::Int,
    T::Real,
    Ω::Real,
    Δ::Real,
)
    float = findfirst(F .== floats)
    alg = findfirst([A == algorithm for algorithm in algorithms])
    return Control(float, alg, m, n, F(T), F(Ω), F(Δ))
end

struct Independent <: Experiments.Independent
    r::Int                  # NUMBER OF TROTTER STEPS
end

struct Dependent <: Experiments.Dependent
    iF::Float64             # INFIDELITY wrt ANALYTICAL SOLUTION
end

struct Setup{F,P} <: Experiments.Setup
    device::D{F,S{P}}       # THE DEVICE OBJECT BEING EVOLVED
    ψ0::Vector{Complex{F}}  # THE INITIAL STATEVECTOR
end

struct Result{F} <: Experiments.Result
    ψ_::Vector{Complex{F}}  # THE FINAL STATEVECTOR WITH HALVED r
    ψx::Vector{Complex{F}}  # THE FINAL STATEVECTOR
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

    x̄ = Vector{F}(undef, 3n)
    x̄[1:2:2n] .= Ā
    x̄[2:2:2n] .= B̄
    x̄[1+2n:3n].= ν̄
    CtrlVQE.Parameters.bind(device, x̄)

    # PREPARE THE INITIAL STATE

    ψ0 = zeros(Complex{F}, expmt.m^n); ψ0[2] = 1

    return Setup(device, ψ0)
end

function Experiments.mapindex(expmt::Control, i::Integer)
    return Independent(2^i)
end

function Experiments.runtrial(
    expmt::Control{F},
    setup::Setup,
    xvars::Independent,
) where {F}
    A = algorithms[expmt.alg]

    ψ_ = CtrlVQE.Evolutions.evolve(A, setup.device, expmt.T, setup.ψ0; r=xvars.r÷2)
    ψx = CtrlVQE.Evolutions.evolve(A, setup.device, expmt.T, setup.ψ0; r=xvars.r)
    return Result(ψ_, ψx)
end

function Experiments.synthesize(::Control,
    setup::Setup,
    xvars::Independent,
    result::Result,
)
    iF = 1 - abs(result.ψ_' * result.ψx)^2
    return Dependent(iF)
end