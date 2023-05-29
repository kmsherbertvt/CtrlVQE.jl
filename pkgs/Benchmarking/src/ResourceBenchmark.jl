import LinearAlgebra
import Random

import BenchmarkTools: @benchmark, Trial

import Experiments
import CtrlVQE
import ..Nmn, ..algorithms, ..floats

const Device = CtrlVQE.Devices.TransmonDevice

struct Control{F} <: Experiments.Control
    float::Int              # INDEXES WHICH FLOAT TYPE
    alg_ix::Int             # INDEXES WHICH ALGORITHM
    m::Int                  # NUMBER OF LEVELS PER TRANSMON
    n::Int                  # NUMBER OF QUBITS
    seed::Int               # RANDOM SEED USED TO GENERATE HERMITIAN OBSERVABLE
    λ::F                    # RMS EIGENVALUE OF RANDOMLY GENERATED HERMITIAN
    Ω::F                    # AMPLITUDE MODULUS ON THE PULSE
    Δ::F                    # DETUNING FREQUENCY
    kmax_rE::Int            # MAX POWER FOR FOR TROTTER STEPS (EVOLUTION)
    kmax_rG::Int            # MAX POWER FOR FOR TROTTER STEPS (GRADIENT)
end

function Control(
    F::Type{<:AbstractFloat},
    A::Type{<:CtrlVQE.Evolutions.EvolutionAlgorithm},
    m::Int,
    n::Int,
    seed::Int,
    λ::Real,
    Ω::Real,
    Δ::Real,
    kmax_rE::Int,
    kmax_rG::Int,
)
    float = findfirst(F .== floats)
    alg_ix = findfirst([A == algorithm for algorithm in algorithms])
    return Control(float, alg_ix, m, n, seed, F(λ), F(Ω), F(Δ), kmax_rE, kmax_rG)
end

struct Independent{F} <: Experiments.Independent
    rE::Int                 # NUMBER OF TROTTER STEPS
    rG::Int                 # NUMBER OF TROTTER STEPS
    T::F                    # TOTAL DURATION OF PULSE
end

struct Dependent <: Experiments.Dependent
    # EVOLUTION BENCHMARKS
    time_E::Float64         # MINIMUM EXECUTION TIME (ns)
    gctime_E::Float64       # MINIMUM TIME SPENT ON GARBAGE COLLECTION (ns)
    memory_E::Int           # MINIMUM MEMORY CONSUMPTION (bytes)
    allocs_E::Int           # MINIMUM NUMBER OF ALLOCATIONS
    # EVOLUTION BENCHMARKS
    time_G::Float64         # MINIMUM EXECUTION TIME (ns)
    gctime_G::Float64       # MINIMUM TIME SPENT ON GARBAGE COLLECTION (ns)
    memory_G::Int           # MINIMUM MEMORY CONSUMPTION (bytes)
    allocs_G::Int           # MINIMUM NUMBER OF ALLOCATIONS
end

struct Setup{F} <: Experiments.Setup
    observable::Matrix{F}   # THE HERMITIAN OBSERVABLE
    device::Device{F}       # THE DEVICE OBJECT BEING EVOLVED
    ψ0::Vector{Complex{F}}  # THE INITIAL STATEVECTOR
end

struct Result <: Experiments.Result
    benchmark_E::Trial      # THE BENCHMARKING RESULTS
    benchmark_G::Trial      # THE BENCHMARKING RESULTS
end




function Experiments.initialize(expmt::Control{F}) where {F}
    n = expmt.n
    N = expmt.m ^ n

    #######################################
    # CONSTRUCT THE OBSERVABLE

    # FIRST WE GENERATE A RANDOM ORTHOGONAL MATRIX
    Random.seed!(expmt.seed)
    Q, R = LinearAlgebra.qr(randn(F, N, N))
    Q = Q * LinearAlgebra.Diagonal(sign.(LinearAlgebra.diag(R)))

    # NEXT WE GENERATE A UNIFORM RANDOM EIGENSPECTRUM, CONSTRAINED BY RMS
    Λ = 2*rand(N) .- 1      # [-1, 1)
    Λ ./= LinearAlgebra.norm( Λ ./ (expmt.λ * √N) )

    # NOW WE COMBINE THE TWO
    observable = Q * LinearAlgebra.Diagonal(Λ) * Q'

    #######################################
    # PREPARE THE DEVICE
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

    #######################################
    # PREPARE THE INITIAL STATE
    ψ0 = zeros(Complex{F}, expmt.m^n); ψ0[2] = 1

    return Setup(observable, device, ψ0)
end

function Experiments.mapindex(expmt::Control, i::Integer)
    i, rE = divrem(i, expmt.krEmax+1); rE = 2^rE
    i, rG = divrem(i, expmt.krGmax+1); rG = 2^rG
    T = F(10^i)
    return Independent(rE,rG,T)
end

function Experiments.runtrial(
    expmt::Control,
    setup::Setup,
    xvars::Independent,
)
    algorithm = algorithms[expmt.alg_ix](xvars.rE)

    # WARM-UP RUNS
    ψ = CtrlVQE.Evolutions.evolve(algorithm, setup.device, xvars.T, setup.ψ0)
    Φ̄ = CtrlVQE.Evolutions.gradientsignals(
        setup.device, xvars.T, setup.ψ0, xvars.rG, setup.observable; evolution=algorithm
    )

    # PRODUCTION RUNS
    bE = @benchmark CtrlVQE.Evolutions.evolve(
        $algorithm, $setup.device, $xvars.T, $setup.ψ0;
        result=$ψ,
    )
    bG = @benchmark CtrlVQE.Evolutions.gradientsignals(
        $setup.device, $xvars.T, $setup.ψ0, $xvars.rG, $setup.observable;
        result=$Φ̄, evolution=$algorithm,
    )

    return Result(bE, bG)
end

function Experiments.synthesize(::Control,
    setup::Setup,
    xvars::Independent,
    result::Result,
)
    mintrial_E = minimum(result.benchmark_E)
    mintrial_G = minimum(result.benchmark_G)
    return Dependent(
        mintrial_E.time,
        mintrial_E.gctime,
        mintrial_E.memory,
        mintrial_E.allocs,
        mintrial_G.time,
        mintrial_G.gctime,
        mintrial_G.memory,
        mintrial_G.allocs,
    )
end