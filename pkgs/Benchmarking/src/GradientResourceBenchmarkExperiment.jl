using BenchmarkTools: @benchmark, Trial
import Experiments
import CtrlVQE
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
    m::Int              # NUMBER OF LEVELS PER TRANSMON
    n::Int              # NUMBER OF QUBITS
    seed::Int           # RANDOM SEED USED TO GENERATE HERMITIAN OBSERVABLE
    T::F                # TOTAL DURATION OF PULSE
    Ω::F                # AMPLITUDE MODULUS ON THE PULSE
    Δ::F                # DETUNING FREQUENCY
end

function Control(
    F::Type{<:AbstractFloat},
    m::Int,
    n::Int,
    seed::Int,
    T::Real,
    Ω::Real,
    Δ::Real,
)
    float = findfirst(F .== floats)
    return Control(float, m, n, seed, F(T), F(Ω), F(Δ))
end

struct Independent <: Experiments.Independent
    r::Int                  # NUMBER OF TROTTER STEPS
end

struct Dependent <: Experiments.Dependent
    time::Float64           # MINIMUM EXECUTION TIME (ns)
    gctime::Float64         # MINIMUM TIME SPENT ON GARBAGE COLLECTION (ns)
    memory::Int             # MINIMUM MEMORY CONSUMPTION (bytes)
    allocs::Int             # MINIMUM NUMBER OF ALLOCATIONS
end

struct Setup{F} <: Experiments.Setup
    observable::Matrix{F}   # THE HERMITIAN OBSERVABLE
    device::D{F,S{F}}       # THE DEVICE OBJECT BEING EVOLVED
    ψ0::Vector{Complex{F}}  # THE INITIAL STATEVECTOR
end

struct Result <: Experiments.Result
    benchmark::Trial        # THE BENCHMARKING RESULTS
end




function Experiments.initialize(expmt::Control{F}) where {F}
    # FORM A HERMITIAN OBSERVABLE
    N = expmt.m ^ expmt.n
    rmatrix = rand(F, N, N)
    observable = rmatrix + rmatrix'

    # PREPARE THE DEVICE
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

    return Setup(observable, device, ψ0)
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
    device, T, ψ0, r, O = setup.device, expmt.T, setup.ψ0, xvars.r, setup.observable

    # WARM-UP RUN
    CtrlVQE.Evolutions.gradientsignals(device, T, ψ0, r, [O])

    # PRODUCTION RUN
    benchmk = @benchmark CtrlVQE.Evolutions.gradientsignals($device, $T, $ψ0, $r, [$O])

    return Result(benchmk)
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