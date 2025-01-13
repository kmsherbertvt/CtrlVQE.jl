import ModularFramework as Modular

import CtrlVQE
import CtrlVQE: LAT
import CtrlVQE: Parameters
import CtrlVQE: Integrations, Devices

import CtrlVQE.Bases: DRESSED
import CtrlVQE.Operators: STATIC
import CtrlVQE: Quple, QUBIT_FRAME
import CtrlVQE: Windowed, Constant, Constrained

import NPZ
import LineSearches
import Optim

import LinearAlgebra: eigen
const Float = Float64

# INCLUDE LOCAL MODULES
include("SmoothWindowedSignals.jl")
include("WindowOverlapPenalties.jl")

# SET UP THE CHEMISTRY
H = NPZ.npzread((@__DIR__)*"/lih30.npy")
                # LiH at 3.0Å, parity mapping with two-qubit tapering.
ψ0 = convert(Vector{Complex{Float}}, LAT.basisvector(16, 4))
ket = [0,0,1,1] # HARTREE FOCK STATE (mapping specific)
n = length(ket) # NUMBER OF QUBITS

REF = real(ψ0' * H * ψ0)        # HARTREE FOCK REFERENCE ENERGY
FCI = eigen(H).values[1]        # FULL CONFIGURATION INTERACTION

# ANSATZ CONSTANTS AND CONSTRAINTS
T = 21.0 # ns           # PULSE DURATION
W = 7                   # NUMBER OF WINDOWS
r = round(Int, 20T)     # NUMBER OF TROTTER STEPS

ΩMAX = 2π*0.020 # GHz   # MAXIMUM AMPLITUDE
σΩ = 2π*0.001 # GHsz    # STEEPNESS OF AMPLITUDE PENALTY
λΩ = 0.001 # Ha         # STRENGTH OF AMPLITUDE PENALTY

s0 = 1.5 # ns           # MINIMUM WINDOW DURATION
σs = 0.1 # ns           # STEEPNESS OF OVERLAP PENALTY (smooth windows only)
λs = 1.0 # Ha           # STRENGTH OF OVERLAP PENALTY

# INITIALIZE SIMULATION OBJECTS
template_α = Constrained(Constant(zero(Complex{Float})), :B)
template_β = Constrained(Constant(zero(Complex{Float})), :A)
template_ν = Constrained(Constant(zero(Float)), :A)

grid = CtrlVQE.TemporalLattice(T, r)

ω = 2π.*[4.82, 4.84, 4.85, 4.88]
A = Modular.TruncatedBosonicAlgebra{2,n}
drift = Modular.TransmonDrift{A}(
    ω, fill(2π.*0.30, n), diff(ω),
    [Quple(q,q+1) for q in 1:n-1],
)

reference = Modular.KetReference(ket, DRESSED)
measurement = Modular.DenseMeasurement(H, DRESSED, STATIC)

# INITIALIZE OPTIMIZATION SETTINGS
linesearch = LineSearches.MoreThuente()
optimizer = Optim.BFGS(linesearch=linesearch)   # STANDARD BFGS
options = Optim.Options(
    show_trace = true,
    show_every = 1,
    f_tol = 0.0,
    g_tol = 1e-6,
    iterations = 10000,
)

##########################################################################################
#= RUN OPTIMIZATIONS =#

for ansatz in [:smooth, :window]
    include("$ansatz/makedevice.jl")

    fw  = CtrlVQE.cost_function(costfn)
    g!w = CtrlVQE.grad!function(costfn)
    xiw = collect(Parameters.values(device))

    optimizationw = Optim.optimize(fw, g!w, xiw, optimizer, options)
    xfw = Optim.minimizer(optimizationw)

    NPZ.npzwrite((@__DIR__)*"/$ansatz/x.npy", xfw)
end

# ##############

# include("smooth/makedevice.jl")

# fs  = CtrlVQE.cost_function(costfn)
# g!s = CtrlVQE.grad!function(costfn)
# xis = collect(Parameters.values(device))

# optimizations = Optim.optimize(fs, g!s, xis, optimizer, options)
# xfs = Optim.minimizer(optimizations)

# NPZ.npzwrite((@__DIR__)*"/smooth/x.npy", xfs)
