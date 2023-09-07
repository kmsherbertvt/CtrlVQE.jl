using Test

import Random, LinearAlgebra

import .StandardTests

import CtrlVQE
import CtrlVQE: Operators, Signals, Devices, Evolutions, CostFunctions
import CtrlVQE.Bases: OCCUPATION
import CtrlVQE.Quples: Quple


##########################################################################################
# ENERGY FUNCTION SELF-CONSISTENCY CHECKS


# TRANSMON DEVICE PARAMETERS
ω̄ = 2π * [4.50, 4.52]
δ̄ = 2π * [0.33, 0.34]
ḡ = 2π * [0.020]
quples = [Quple(1, 2)]
q̄ = [1, 2]
ν̄ = 2π * [4.30, 4.80]
Ω̄ = [
    Signals.Constant( 0.020*2π),
    Signals.Constant(-0.020*2π),
]
m = 3
device = Devices.TransmonDevice(ω̄, δ̄, ḡ, quples, q̄, ν̄, Ω̄, m)

# OBSERVABLE AND REFERENCE STATE
N = Devices.nstates(device)

Random.seed!(0)
O0 = LinearAlgebra.Hermitian(rand(ComplexF64, N,N))
ψ0 = zeros(ComplexF64, N); ψ0[1] = 1

# ALGORITHM AND BASIS
T = 10.0
r = 1000
algorithm = Evolutions.Rotate(r)
basis = OCCUPATION

# TEST ENERGY FUNCTIONS!

# (OPERATOR AND COST FUNCTION IMPORTS TO FACILITATE EASY LOOPING AND LABELING)
import CtrlVQE.Operators: Identity, IDENTITY
import CtrlVQE.Operators: Uncoupled, UNCOUPLED
import CtrlVQE.Operators: Static, STATIC
import CtrlVQE: BareEnergy, ProjectedEnergy, NormalizedEnergy

for frame in [IDENTITY, UNCOUPLED, STATIC];
for fn_type in [BareEnergy, ProjectedEnergy, NormalizedEnergy]
    label = "$fn_type - Frame: $(typeof(frame))"
    @testset "$label" begin
        fn = fn_type(
            O0, ψ0, T, device, r;
            algorithm=algorithm, basis=basis, frame=frame,
        )
        StandardTests.validate(fn)
    end
end; end

# TEST NORMALIZATION FUNCTION
@testset "Normalization" begin
    fn = CtrlVQE.Normalization(
        ψ0, T, device, r;
        algorithm=algorithm, basis=basis,
    )
    StandardTests.validate(fn)
end