import ModularFramework: LocalDevice, Energy
import ModularFramework: TruncatedBosonicAlgebra
import ModularFramework: TransmonDrift, DipoleDrive
import ModularFramework: DISJOINT
import ModularFramework: KetReference, DenseMeasurement

import CtrlVQE
import CtrlVQE.Bases: DRESSED
import CtrlVQE.Operators: STATIC
import CtrlVQE.QUBIT_FRAME
import CtrlVQE: Constant, Windowed, Constrained

import LinearAlgebra: eigen

Float = Float64

# SET UP THE CHEMISTRY
H = ComplexF64[                 # H2 at 1.5Å, parity mapping with two-qubit tapering.
    -0.3944683030459425                     0.0 0.22953593605970196 0.0
    0.0                 -0.6610488453402209 0.0                     -0.22953593605970196
    0.22953593605970196 0.0                 -0.9108735545943861     0.0
    0.0                 -0.22953593605970196 0.0                    -0.6610488453402209
]
n = 2

ψ0 = Complex{Float}[0, 0, 1, 0] # HARTREE FOCK STATE (mapping specific)
ket0 = [1, 0]                   #   same but in ket form
REF = real(ψ0' * H * ψ0)        # HARTREE FOCK REFERENCE ENERGY
FCI = eigen(H).values[1]        # FULL CONFIGURATION INTERACTION

# SET UP THE PHYSICS
T = 10.0                        # TOTAL DURATION OF PULSE
W = 2                           # NUMBER OF WINDOWS IN PULSE
grid = CtrlVQE.TemporalLattice(T, round(Int, 20T))

ω = 2π.*[4.82, 4.84] # GHz
A = TruncatedBosonicAlgebra{2,2}
drift = TransmonDrift{A}(ω, 2π.*[0.30, 0.30], diff(ω), [CtrlVQE.Quple(1,2)])
drives = [DipoleDrive{A}(
    q, ω[q],
    Windowed(Constant(zero(Complex{Float})), T, W),
    Constrained(Constant(zero(Float)), :A),
) for q in 1:2]
device = LocalDevice(eltype(Δ), A(), drift, drives, DISJOINT)

reference = KetReference(ket0, DRESSED)
measurement = DenseMeasurement(H, DRESSED, STATIC)
costfn = Energy(QUBIT_FRAME, device, grid, reference, measurement)

# TEST SOME KNOWN PULSES
xREF = zeros(2*2*W)                     # ZERO-PULSE SHOULD GIVE HARTREE FOCK ENERGY
@test abs(costfn(xREF) - REF) < 1e-10

xFCI = [                                # PRE-OPTIMIZED PARAMETERS FOR FCI ENERGY
    0.029784841060555706,
    -0.04833626569346621,               # NOTE:
    -0.03987767831836384,               # These parameters were found numerically,
     0.0032901248420285365,             #   using the current version of the code.
    -0.00639228870893881,               # Expect to have to update them
     0.029837465232972194,              #   if the current version of the code
     0.024475817299244966,              #   is found to have errors. :)
    -0.03337472131977085,
]
@test abs(costfn(xFCI) - FCI) < 1e-10