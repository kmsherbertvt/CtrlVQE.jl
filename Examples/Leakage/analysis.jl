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
import Plots

import LinearAlgebra: eigen, I
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
#= COMPUTE EACH OBSERVABLE =#

ms = 2:6
E = Dict()
Π = Dict()

for ansatz in [:smooth, :window]
    E[ansatz] = Float[]
    Π[ansatz] = Float[]
    for m in ms
        ψfile = (@__DIR__)*"/$ansatz/m$m.npy"
        ψ = NPZ.npzread(ψfile)

        x = NPZ.npzread((@__DIR__)*"/$ansatz/x.npy")
        make_drive = (ansatz == :smooth ? make_drive_smooth : make_drive_window)

        A = Modular.TruncatedBosonicAlgebra{m,n}
        device = Modular.LocalDevice(
            Float, A(),
            make_drift(A),
            [make_drive(A,i) for i in 1:2n],
            Modular.DISJOINT,
        )

        hamiltonian = Modular.DenseMeasurement(project(    H ,m,n), DRESSED, STATIC)
        projection  = Modular.DenseMeasurement(project(one(H),m,n), DRESSED, STATIC)

        push!(E[ansatz], Modular.measure(hamiltonian, device, DRESSED, ψ, T))
        push!(Π[ansatz], Modular.measure( projection, device, DRESSED, ψ, T))
    end
end


##########################################################################################
#= PLOT LEAKAGE AND NORMALIZED ENERGIES =#

plt = Plots.plot(;
    xlabel = "Levels per Transmon",
    ylabel = "Normalized Energy",
    # ylims = [FCI, REF],
    palette = :roma10,
)
Plots.plot!(plt, ms, E[:window]./Π[:window]; lw=3, color=1, label="Stepped Energy")
Plots.plot!(plt, ms, E[:smooth]./Π[:smooth]; lw=3, color=5, label="Smooth Energy")
Plots.hline!(plt, [REF]; ls=:dot, color=:black, label="Reference Energy")

twin = Plots.twinx(plt)
Plots.plot!(twin;
    ylabel = "Leakage",
    ylims = [0,1],
)
Plots.plot!(twin, ms, 1 .- Π[:window]; lw=3, ls=:dot, color=2, label="Stepped Leakage")
Plots.plot!(twin, ms, 1 .- Π[:smooth]; lw=3, ls=:dot, color=6, label="Smooth Leakage")

Plots.savefig(plt, (@__DIR__)*"/convergence.pdf")