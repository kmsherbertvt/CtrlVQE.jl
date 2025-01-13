import CtrlVQE

import CtrlVQE.Bases: DRESSED
import CtrlVQE.Operators: STATIC
import CtrlVQE.QUBIT_FRAME

import LinearAlgebra: eigen

Float = Float64

# SET UP THE CHEMISTRY
H = ComplexF64[                 # H2 at 1.5Å, parity mapping with two-qubit tapering.
    -0.3944683030459425                     0.0 0.22953593605970196 0.0
    0.0                 -0.6610488453402209 0.0                     -0.22953593605970196
    0.22953593605970196 0.0                 -0.9108735545943861     0.0
    0.0                 -0.22953593605970196 0.0                    -0.6610488453402209
]

ψ0 = Complex{Float}[0, 0, 1, 0] # HARTREE FOCK STATE (mapping specific)
REF = real(ψ0' * H * ψ0)        # HARTREE FOCK REFERENCE ENERGY
FCI = eigen(H).values[1]        # FULL CONFIGURATION INTERACTION

# SET UP THE PHYSICS
T = 10.0                        # TOTAL DURATION OF PULSE
W = 2                           # NUMBER OF WINDOWS IN PULSE
grid = CtrlVQE.TemporalLattice(T, round(Int, 20T))
device = CtrlVQE.CWRTDevice{2}(
    2π.*[4.82, 4.84],               # QUBIT RESONANCE
    2π.*[0.30, 0.30],               # ANHARMONICITIES
    2π.*[0.02],                     # COUPLING STRENGTH
    [CtrlVQE.Quple(1,2)],     # COUPLING MAP
    T, W,
)
costfn = CtrlVQE.DenseObservable(H, ψ0, device, DRESSED, STATIC, grid, QUBIT_FRAME)

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