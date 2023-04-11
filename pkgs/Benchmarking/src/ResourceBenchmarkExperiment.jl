using BenchmarkTools: @benchmark, Trial
import Experiments
import CtrlVQE
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
    T::F                # TOTAL DURATION OF PULSE
    Ωmax::F             # MAXIMUM AMPLITUDE ON THE PULSE
    kmax::Int           # MAX TROTTER STEPS IS r=2^kmax
end

function Control(
    F::Type{<:AbstractFloat},
    A::CtrlVQE.Evolutions.EvolutionAlgorithm,
    T::Real,
    Ωmax::Real,
    kmax::Int,
)
    float = findfirst(F .== floats)
    alg = findfirst([A == algorithm for algorithm in algorithms])
    return Control(float, alg, F(T), F(Ωmax), kmax)
end

struct Independent <: Experiments.Independent
    m::Int                  # NUMBER OF LEVELS PER TRANSMON
    n::Int                  # NUMBER OF QUBITS
    r::Int                  # NUMBER OF TROTTER STEPS
end

struct Dependent <: Experiments.Dependent
    time::Float64           # MINIMUM EXECUTION TIME (ns)
    gctime::Float64         # MINIMUM TIME SPENT ON GARBAGE COLLECTION (ns)
    memory::Int             # MINIMUM MEMORY CONSUMPTION (bytes)
    allocs::Int             # MINIMUM NUMBER OF ALLOCATIONS
end

struct Setup{F} <: Experiments.Setup
    pulse::S{F}             # THE PULSE TO APPLY TO EACH QUBIT
end

struct Result{F} <: Experiments.Result
    device::D{F,S{F}}       # THE DEVICE OBJECT BEING EVOLVED
    ψ0::Vector{Complex{F}}  # THE INITIAL STATEVECTOR
    ψ::Vector{Complex{F}}   # THE FINAL STATEVECTOR
    benchmark::Trial        # THE BENCHMARKING RESULTS
end




function Experiments.initialize(expmt::Control)
    pulse = S(expmt.Ωmax)
    return Setup(pulse)
end

function Experiments.mapindex(expmt::Control, i::Integer)
    i, k = divrem(i, expmt.kmax+1)
    r = 2^k
    N, m, n = Nmn[1+i]
    return Independent(m,n,r)
end

function Experiments.runtrial(
    expmt::Control{F},
    setup::Setup,
    xvars::Independent,
) where {F}
    A = algorithms[expmt.alg]
    device = CtrlVQE.SystematicTransmonDevice(F, xvars.m, xvars.n, setup.pulse)
    ψ0 = zeros(Complex{F}, xvars.m ^ xvars.n); ψ0[2] = 1

    # WARM-UP RUN
    ψ = CtrlVQE.Evolutions.evolve(A, device, expmt.T, ψ0; r=1); ψ .= ψ0

    # PRODUCTION RUN
    T, r = expmt.T, xvars.r
    benchmk = @benchmark CtrlVQE.Evolutions.evolve!($A, $device, $T, $ψ; r=$r)

    return Result(device, ψ0, ψ, benchmk)
end

function Experiments.synthesize(::Control,
    setup::Setup,
    xvars::Independent,
    result::Result,
)
    mintrial = minimum(result.benchmark)
    return Dependent(
        mintrial.time,
        mintrial.gctime,
        mintrial.memory,
        mintrial.allocs,
    )
end