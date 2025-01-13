import ModularFramework as Modular

import CtrlVQE
import CtrlVQE: LAT
import CtrlVQE: Parameters
import CtrlVQE: Integrations, Devices

import CtrlVQE.QubitOperations: project
import CtrlVQE.Bases: DRESSED
import CtrlVQE.Operators: STATIC
import CtrlVQE: Quple, QUBIT_FRAME
import CtrlVQE: Windowed, Constant, Constrained

import NPZ

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

# ANSATZ CONSTANTS
T = 21.0 # ns           # PULSE DURATION
W = 7                   # NUMBER OF WINDOWS
r = round(Int, 20T)     # NUMBER OF TROTTER STEPS
s0 = 1.5 # ns           # MINIMUM WINDOW DURATION (for setting σ in smooth windows)

# INITIALIZE SIMULATION OBJECTS
template_α = Constrained(Constant(zero(Complex{Float})), :B)
template_β = Constrained(Constant(zero(Complex{Float})), :A)
template_ν = Constrained(Constant(zero(Float)), :A)

grid = CtrlVQE.TemporalLattice(T, r)

ω = 2π.*[4.82, 4.84, 4.85, 4.88]
make_drift(A) = Modular.TransmonDrift{A}(
    ω, fill(2π.*0.30, n), diff(ω),
    [Quple(q,q+1) for q in 1:n-1],
)
make_drive_window(A, i) = (
    q = ((i-1) >> 1) + 1;
    template = isodd(i) ? template_α : template_β;
    Modular.DipoleDrive{A}(
        q, ω[q],
        Windowed(template, T, W),
        template_ν,
    )
)
make_drive_smooth(A, i) = (
    q = ((i-1) >> 1) + 1;
    template = isodd(i) ? template_α : template_β;
    Modular.DipoleDrive{A}(
        q, ω[q],
        SmoothWindowedSignals.SmoothWindowed(template, T, W, 2s0/π),
        template_ν,
    )
)
reference = Modular.KetReference(ket, DRESSED)

##########################################################################################
#= COMPUTE EACH ANSATZE =#

for m in 2:6
    for ansatz in [:smooth, :window]
        ψfile = (@__DIR__)*"/$ansatz/m$m.npy"
        isfile(ψfile) && continue

        x = NPZ.npzread((@__DIR__)*"/$ansatz/x.npy")
        make_drive = (ansatz == :smooth ? make_drive_smooth : make_drive_window)

        A = Modular.TruncatedBosonicAlgebra{m,n}
        device = Modular.LocalDevice(
            Float, A(),
            make_drift(A),
            [make_drive(A,i) for i in 1:2n],
            Modular.DISJOINT,
        )
        Parameters.bind!(device, x)

        ψ0 = Modular.prepare(reference, device, DRESSED)
        println("Evolving state $ansatz m=$m...")
        ψ = CtrlVQE.evolve(QUBIT_FRAME, device, DRESSED, grid, ψ0)

        # WRITE RESULTS
        NPZ.npzwrite(ψfile, ψ)
    end
end
